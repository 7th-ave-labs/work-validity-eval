#!/usr/bin/env python3
"""Cluster O*NET task embeddings and write labeled PCA / UMAP / t-SNE projections.

Recipe (matches the 2026-04-21 rerun notes):
  1. L2-normalize the selected embedding variant (default: task_only).
  2. UMAP-reduce to ``--cluster-umap-dims`` dims (default 15, cosine metric);
     this is the space used for density clustering. PCA on 1536-d L2-normed
     text embeddings concentrates pairwise distances and gives HDBSCAN very
     little contrast, so we use UMAP as the cluster-space reducer.
  3. Run HDBSCAN on that UMAP space with ``prediction_data=True`` so that
     noise points can be soft-assigned to their nearest cluster via
     ``hdbscan.all_points_membership_vectors`` (coverage-preserving).
  4. Run KMeans (``--kmeans-k``) on the same UMAP space as a stability
     cross-check; report ARI/NMI against HDBSCAN on non-noise points.
  5. Summarize each cluster and assign a short cross-family operation label
     for tables and figures.

Use ``--variant`` to switch between the primary ``task_only`` embeddings and
the ``title_plus_task`` sensitivity variant; output filenames are suffixed
accordingly unless an explicit path is given for each artifact.

Inputs (default: task-only variant under ``data/derived/embeddings/``):
  data/derived/embeddings/onet_task_embeddings__task_only.npz
  data/derived/embeddings/onet_task_embeddings__task_only_meta.parquet

Outputs (default suffix: task_only):
  data/derived/onet_task_clusters__task_only.parquet
  data/derived/cluster_summary__task_only.csv
  data/derived/cluster_stability__task_only.json
  data/derived/figs/pca_clusters__task_only.png
  data/derived/figs/umap_clusters__task_only.png
  data/derived/figs/tsne_clusters__task_only.png    (unless --skip-tsne)

The canonical paths ``onet_task_clusters.parquet`` / ``cluster_summary.csv`` /
``cluster_stability.json`` / ``figs/*_clusters.png`` are written alongside the
suffixed copies when ``--variant task_only`` (the primary).
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import hdbscan
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from openai import OpenAI
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
EMB = DERIVED / "embeddings"
FIGS = DERIVED / "figs"
DEFAULT_LABEL_MODEL = os.environ.get("CLUSTER_LABEL_MODEL", "gpt-5.4-mini")

HEURISTIC_OPERATION_LABELS = (
    ("compare", "Compare against standards"),
    ("verify", "Verify records and decisions"),
    ("review", "Review records and cases"),
    ("audit", "Audit compliance and quality"),
    ("analy", "Analyze data and evidence"),
    ("diagnos", "Diagnose conditions"),
    ("inspect", "Inspect and assess conditions"),
    ("investig", "Investigate cases and findings"),
    ("design", "Design technical plans"),
    ("draft", "Draft structured documents"),
    ("write", "Write reports and findings"),
    ("teach", "Teach and instruct people"),
    ("train", "Teach and instruct people"),
    ("advise", "Advise clients and teams"),
    ("counsel", "Advise clients and teams"),
    ("treat", "Treat and support clients"),
    ("plan", "Plan programs and actions"),
    ("coordinate", "Coordinate programs and staff"),
    ("manage", "Manage operations and resources"),
    ("schedule", "Coordinate schedules and services"),
    ("monitor", "Monitor systems and environments"),
    ("test", "Test materials and systems"),
    ("measure", "Measure and calibrate systems"),
    ("repair", "Diagnose and repair systems"),
    ("negot", "Negotiate agreements"),
    ("supervis", "Supervise personnel"),
)

HEURISTIC_OBJECT_HINTS = (
    ("record", "records"),
    ("claim", "claims"),
    ("application", "applications"),
    ("patient", "patients"),
    ("student", "students"),
    ("data", "data"),
    ("report", "reports"),
    ("equipment", "equipment"),
    ("system", "systems"),
    ("program", "programs"),
    ("contract", "contracts"),
    ("budget", "budgets"),
    ("site", "sites"),
)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _cluster_color_map(uniq: list[int]) -> dict[int, tuple]:
    """Enough distinct colors for many clusters (tab20 + tab20b)."""
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    return {c: mcolors.to_rgb(base[i % len(base)]) for i, c in enumerate(uniq)}


def _coerce_variance_target(value: float) -> float:
    if value > 1:
        value = value / 100.0
    return min(max(value, 0.5), 0.99)


def _variant_suffix(variant: str) -> str:
    return f"__{variant}"


def _unique_examples(values: pd.Series, limit: int) -> list[str]:
    seen: list[str] = []
    for value in values.astype(str):
        item = value.strip()
        if item and item not in seen:
            seen.append(item)
        if len(seen) >= limit:
            break
    return seen


def reduce_for_clustering(
    emb_n: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Produce a low-d cluster space and a 2D PCA layout for plotting.

    The cluster space is a UMAP reduction of the L2-normalized embeddings;
    it is used for HDBSCAN and KMeans. The 2D PCA layout is only used for
    the PCA scatter plot and is independent of the cluster space.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    cluster_space = reducer.fit_transform(emb_n)
    pca_xy = PCA(n_components=2, random_state=seed).fit_transform(emb_n)
    info: dict[str, float | int | str] = {
        "reducer": "umap",
        "n_components": int(n_components),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "seed": int(seed),
    }
    return cluster_space, pca_xy, info


def heuristic_cluster_label(sample_tasks: list[str]) -> tuple[str, str]:
    text = " ".join(sample_tasks).lower()
    for needle, label in HEURISTIC_OPERATION_LABELS:
        if needle in text:
            return label, f"Tasks in this cluster center on {label.lower()} across occupations."
    for needle, obj in HEURISTIC_OBJECT_HINTS:
        if needle in text:
            label = f"Work with {obj}"
            return label, f"Tasks in this cluster center on handling {obj} across occupations."
    return "Cross-family task work", "Tasks in this cluster share a broad cross-family operation."


def _parse_cluster_label_response(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("clusters"), list):
            return payload["clusters"]
        for value in payload.values():
            if isinstance(value, list):
                return value
    raise ValueError("Could not parse cluster label response.")


def label_clusters(
    summary_payload: list[dict],
    model: str | None,
) -> tuple[dict[int, dict[str, str]], str]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if model and api_key:
        client = OpenAI(api_key=api_key)
        system = textwrap.dedent(
            """
            You label clusters of O*NET task statements.

            For each cluster, return:
            - cluster_id
            - cluster_label: a 2-6 word cross-family operation label
            - cluster_summary: one sentence describing the shared operation
            - figure_label: a short label suitable for a plot annotation

            Rules:
            - Use operation-first phrasing.
            - Avoid occupation titles, departments, or tool names when a broader operation works.
            - Favor labels that could apply across multiple job families.
            - Keep figure_label under about 26 characters.
            - Return JSON only.
            """
        ).strip()
        user = json.dumps({"clusters": summary_payload}, ensure_ascii=False)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = resp.choices[0].message.content
            parsed = _parse_cluster_label_response(json.loads(raw))
            labeled = {
                int(item["cluster_id"]): {
                    "cluster_label": str(item["cluster_label"]).strip(),
                    "cluster_summary": str(item["cluster_summary"]).strip(),
                    "figure_label": str(item.get("figure_label") or item["cluster_label"]).strip(),
                    "label_source": f"llm:{model}",
                }
                for item in parsed
                if "cluster_id" in item and "cluster_label" in item and "cluster_summary" in item
            }
            if labeled:
                for item in summary_payload:
                    cid = int(item["cluster_id"])
                    if cid not in labeled:
                        label, summary = heuristic_cluster_label(item["sample_tasks"])
                        labeled[cid] = {
                            "cluster_label": label,
                            "cluster_summary": summary,
                            "figure_label": label,
                            "label_source": "heuristic-backfill",
                        }
                return labeled, f"llm:{model}"
        except Exception as exc:  # pragma: no cover
            print(f"[warn] cluster labeling fallback after LLM error: {exc!r}")

    labeled = {}
    for item in summary_payload:
        label, summary = heuristic_cluster_label(item["sample_tasks"])
        labeled[int(item["cluster_id"])] = {
            "cluster_label": label,
            "cluster_summary": summary,
            "figure_label": label,
            "label_source": "heuristic",
        }
    return labeled, "heuristic"


def build_cluster_summary(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    payload: list[dict] = []
    clustered = meta[meta["hdbscan_cluster"] != -1]
    for cid, sub in clustered.groupby("hdbscan_cluster"):
        sample_tasks = _unique_examples(sub["task_text_clean"], limit=5)
        sample_occupations = _unique_examples(sub["occupation_title"], limit=6)
        rows.append(
            {
                "hdbscan_cluster": int(cid),
                "n_tasks": int(len(sub)),
                "n_occupations": int(sub["onetsoc_code"].nunique()),
                "importance_mean": float(sub["importance_mean"].mean()),
                "relevance_mean": float(sub["relevance_pct"].mean()),
                "sample_occupations": "; ".join(sample_occupations[:5]),
                "sample_tasks": " | ".join(sample_tasks[:4])[:800],
            }
        )
        payload.append(
            {
                "cluster_id": int(cid),
                "n_tasks": int(len(sub)),
                "n_occupations": int(sub["onetsoc_code"].nunique()),
                "sample_occupations": sample_occupations,
                "sample_tasks": sample_tasks,
            }
        )
    summary = pd.DataFrame(rows).sort_values("n_tasks", ascending=False).reset_index(drop=True)
    return summary, payload


def plot_projection(
    name: str,
    xy: np.ndarray,
    labels: np.ndarray,
    summary_lookup: dict[int, dict[str, str]],
    out_png: Path,
    title_suffix: str,
    *,
    draw_hulls: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 11), dpi=160)
    noise = labels == -1
    uniq = sorted({int(x) for x in labels if x != -1})
    cmap = _cluster_color_map(uniq)

    if draw_hulls:
        for c in uniq:
            m = labels == c
            pts = xy[m]
            if pts.shape[0] < 3:
                continue
            rgb = cmap[c]
            try:
                hull = ConvexHull(pts)
            except QhullError:
                continue
            verts = pts[hull.vertices]
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=(*rgb, 0.14),
                    edgecolor=(*rgb, 0.58),
                    linewidth=1.2,
                    zorder=1,
                )
            )

    if noise.any():
        ax.scatter(
            xy[noise, 0],
            xy[noise, 1],
            s=4,
            c="#cfcfcf",
            alpha=0.28,
            zorder=2,
            rasterized=True,
        )

    for c in uniq:
        m = labels == c
        rgb = cmap[c]
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=8,
            color=rgb,
            alpha=0.76,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )

    for c in uniq:
        m = labels == c
        cx = float(np.median(xy[m, 0]))
        cy = float(np.median(xy[m, 1]))
        label = summary_lookup.get(c, {}).get("figure_label", f"Cluster {c}")
        wrapped = textwrap.fill(label, width=20)
        ax.text(
            cx,
            cy,
            wrapped,
            fontsize=7.8,
            ha="center",
            va="center",
            color="#111111",
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": "#555555",
                "linewidth": 0.45,
                "alpha": 0.88,
            },
        )

    ax.set_title(
        f"{name.upper()} projection of O*NET task embeddings\n{title_suffix}",
        fontsize=13,
    )
    ax.text(
        0.01,
        0.01,
        "Cluster labels are cross-family operation summaries; gray points are unassigned by HDBSCAN.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4d4d4d",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="task_only",
        choices=("task_only", "title_plus_task", "operation_only"),
        help="Embedding variant to cluster. Drives default input/output paths.",
    )
    p.add_argument(
        "--npz",
        default=None,
        help="Override embedding matrix path (defaults to the variant's npz).",
    )
    p.add_argument("--meta", default=None, help="Override embedding meta path.")
    p.add_argument("--out", default=None, help="Override clusters parquet path.")
    p.add_argument("--summary", default=None, help="Override cluster summary CSV path.")
    p.add_argument("--report", default=None, help="Override cluster_stability JSON path.")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    p.add_argument("--hdbscan-min-samples", type=int, default=10)
    p.add_argument(
        "--cluster-umap-dims",
        type=int,
        default=15,
        help="UMAP dimensions used for density clustering (HDBSCAN + KMeans).",
    )
    p.add_argument(
        "--cluster-umap-neighbors",
        type=int,
        default=30,
        help="n_neighbors for the cluster-space UMAP reducer.",
    )
    p.add_argument(
        "--cluster-umap-min-dist",
        type=float,
        default=0.0,
        help="min_dist for the cluster-space UMAP reducer (0.0 favors density).",
    )
    p.add_argument(
        "--cluster-umap-metric",
        default="cosine",
        help="Metric for the cluster-space UMAP reducer.",
    )
    p.add_argument("--umap-neighbors", type=int, default=30, help="n_neighbors for the 2D UMAP layout.")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="min_dist for the 2D UMAP layout.")
    p.add_argument("--tsne-perplexity", type=int, default=30)
    p.add_argument("--tsne-max-iter", type=int, default=1000)
    p.add_argument(
        "--kmeans-k",
        type=int,
        default=40,
        help="k for the KMeans stability cross-check on the UMAP cluster space.",
    )
    p.add_argument(
        "--cluster-label-model",
        default=DEFAULT_LABEL_MODEL,
        help="Model used to turn cluster summaries into short cross-family labels. "
        "If API credentials are absent or labeling fails, a heuristic fallback is used.",
    )
    p.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE (saves time on very large n).",
    )
    p.add_argument("--no-hulls", action="store_true", help="Disable convex-hull shadows.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    suffix = _variant_suffix(args.variant)
    args.npz = args.npz or str(EMB / f"onet_task_embeddings{suffix}.npz")
    args.meta = args.meta or str(EMB / f"onet_task_embeddings{suffix}_meta.parquet")
    args.out = args.out or str(DERIVED / f"onet_task_clusters{suffix}.parquet")
    args.summary = args.summary or str(DERIVED / f"cluster_summary{suffix}.csv")
    args.report = args.report or str(DERIVED / f"cluster_stability{suffix}.json")

    FIGS.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz)
    emb = z["embeddings"]
    meta = pd.read_parquet(args.meta).copy()
    assert emb.shape[0] == len(meta), (emb.shape, len(meta))

    emb_n = l2_normalize(emb)
    cluster_space, pca_xy, reduction_info = reduce_for_clustering(
        emb_n,
        n_components=args.cluster_umap_dims,
        n_neighbors=args.cluster_umap_neighbors,
        min_dist=args.cluster_umap_min_dist,
        metric=args.cluster_umap_metric,
        seed=args.seed,
    )
    print(
        "UMAP reduction before clustering:",
        f"dims={reduction_info['n_components']}",
        f"n_neighbors={reduction_info['n_neighbors']}",
        f"min_dist={reduction_info['min_dist']}",
        f"metric={reduction_info['metric']}",
    )

    print(f"Running HDBSCAN (min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    hdb_labels = clusterer.fit_predict(cluster_space)
    n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = int((hdb_labels == -1).sum())
    print(f"  HDBSCAN clusters: {n_hdb}; noise: {n_noise} / {len(hdb_labels)}")

    # Soft-assign noise (and every other point) to its most-likely cluster so
    # downstream samplers (propose/review) have full corpus coverage.
    assigned_labels = hdb_labels.copy()
    n_reassigned = 0
    if n_hdb > 0 and n_noise > 0:
        try:
            mem = hdbscan.all_points_membership_vectors(clusterer)
            cluster_ids = sorted(set(int(c) for c in hdb_labels if c != -1))
            if mem is not None and mem.size > 0 and cluster_ids:
                best = mem.argmax(axis=1)
                mapping = np.array(cluster_ids)
                soft = mapping[best]
                noise_mask = hdb_labels == -1
                assigned_labels[noise_mask] = soft[noise_mask]
                n_reassigned = int(noise_mask.sum())
                print(f"  HDBSCAN noise soft-assigned to nearest cluster: {n_reassigned}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] HDBSCAN soft-assignment unavailable ({exc!r}); falling back to hard labels.")

    print(f"Running KMeans (k={args.kmeans_k}) stability cross-check on UMAP cluster space")
    km = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=args.seed)
    km_labels = km.fit_predict(cluster_space)
    mask = hdb_labels != -1
    if mask.sum() > 0:
        ari = float(adjusted_rand_score(hdb_labels[mask], km_labels[mask]))
        nmi = float(normalized_mutual_info_score(hdb_labels[mask], km_labels[mask]))
    else:
        ari = float("nan")
        nmi = float("nan")
    print(f"  HDBSCAN vs KMeans (non-noise): ARI={ari:.3f}, NMI={nmi:.3f}")

    print("Running UMAP (2D) layout from L2-normalized embeddings (cosine)")
    umap_metric = "cosine"
    reducer2d = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=umap_metric,
        random_state=args.seed,
    )
    umap_xy = reducer2d.fit_transform(emb_n)

    tsne_xy: np.ndarray | None = None
    if not args.skip_tsne:
        print("Running t-SNE (2D) on PCA-reduced cluster space")
        n = cluster_space.shape[0]
        perp = int(min(args.tsne_perplexity, max(5, (n - 1) // 3)))
        print(f"  t-SNE perplexity={perp}, max_iter={args.tsne_max_iter}")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
            max_iter=args.tsne_max_iter,
        )
        tsne_xy = tsne.fit_transform(cluster_space)
    else:
        print("Skipping t-SNE (--skip-tsne)")

    meta["hdbscan_cluster"] = hdb_labels
    meta["hdbscan_cluster_assigned"] = assigned_labels
    meta["kmeans_cluster"] = km_labels
    meta["pca_x"] = pca_xy[:, 0]
    meta["pca_y"] = pca_xy[:, 1]
    meta["umap_x"] = umap_xy[:, 0]
    meta["umap_y"] = umap_xy[:, 1]
    if tsne_xy is not None:
        meta["tsne_x"] = tsne_xy[:, 0]
        meta["tsne_y"] = tsne_xy[:, 1]
    else:
        meta["tsne_x"] = np.nan
        meta["tsne_y"] = np.nan

    summary, summary_payload = build_cluster_summary(meta)
    label_lookup, label_source = label_clusters(summary_payload, args.cluster_label_model)
    summary["cluster_label"] = summary["hdbscan_cluster"].map(
        lambda cid: label_lookup[int(cid)]["cluster_label"]
    )
    summary["cluster_summary"] = summary["hdbscan_cluster"].map(
        lambda cid: label_lookup[int(cid)]["cluster_summary"]
    )
    summary["figure_label"] = summary["hdbscan_cluster"].map(
        lambda cid: label_lookup[int(cid)]["figure_label"]
    )
    summary["label_source"] = summary["hdbscan_cluster"].map(
        lambda cid: label_lookup[int(cid)]["label_source"]
    )
    meta["cluster_label"] = meta["hdbscan_cluster"].map(
        lambda cid: label_lookup.get(int(cid), {}).get("cluster_label") if pd.notna(cid) else None
    )
    meta["cluster_summary"] = meta["hdbscan_cluster"].map(
        lambda cid: label_lookup.get(int(cid), {}).get("cluster_summary") if pd.notna(cid) else None
    )
    meta["figure_label"] = meta["hdbscan_cluster"].map(
        lambda cid: label_lookup.get(int(cid), {}).get("figure_label") if pd.notna(cid) else None
    )

    meta.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")

    summary.to_csv(args.summary, index=False)
    print(f"Wrote {args.summary} ({len(summary)} clusters)")

    hulls = not args.no_hulls
    label_arr = meta["hdbscan_cluster"].to_numpy()
    title_suffix = (
        f"HDBSCAN on UMAP-reduced {args.variant} embeddings "
        f"({int(reduction_info['n_components'])} dims, "
        f"metric={reduction_info['metric']})"
    )
    for name, xy in [
        ("pca", pca_xy),
        ("umap", umap_xy),
    ]:
        plot_projection(
            name,
            xy,
            label_arr,
            label_lookup,
            FIGS / f"{name}_clusters{suffix}.png",
            title_suffix,
            draw_hulls=hulls,
        )
        print(f"Wrote {FIGS / f'{name}_clusters{suffix}.png'}")

    if tsne_xy is not None:
        plot_projection(
            "tsne",
            tsne_xy,
            label_arr,
            label_lookup,
            FIGS / f"tsne_clusters{suffix}.png",
            title_suffix,
            draw_hulls=hulls,
        )
        print(f"Wrote {FIGS / f'tsne_clusters{suffix}.png'}")

    report = {
        "variant": args.variant,
        "hdbscan": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "n_clusters": int(n_hdb),
            "n_noise": int(n_noise),
            "n_soft_assigned": int(n_reassigned),
            "n_total": int(len(hdb_labels)),
        },
        "kmeans": {
            "k": int(args.kmeans_k),
            "seed": int(args.seed),
        },
        "ari_hdbscan_vs_kmeans_non_noise": ari,
        "nmi_hdbscan_vs_kmeans_non_noise": nmi,
        "pre_clustering_reduction": reduction_info,
        "cluster_labeling": {
            "source": label_source,
            "model_requested": args.cluster_label_model,
        },
        "layouts": {
            "pca": {
                "n_components": 2,
                "input": "L2-normalized embeddings (variant=" + args.variant + ")",
            },
            "umap": {
                "n_neighbors": args.umap_neighbors,
                "min_dist": args.umap_min_dist,
                "metric": umap_metric,
                "input": "L2-normalized embeddings (variant=" + args.variant + ")",
            },
            "tsne": (
                {
                    "perplexity": int(
                        min(args.tsne_perplexity, max(5, (len(cluster_space) - 1) // 3))
                    ),
                    "max_iter": args.tsne_max_iter,
                    "input_dims": int(reduction_info["n_components"]),
                }
                if tsne_xy is not None
                else None
            ),
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")

    # Also mirror the primary (task_only) artifacts to the canonical un-suffixed
    # paths so legacy consumers keep working without flag changes.
    if args.variant == "task_only":
        import shutil

        canonical_pairs = [
            (args.out, str(DERIVED / "onet_task_clusters.parquet")),
            (args.summary, str(DERIVED / "cluster_summary.csv")),
            (args.report, str(DERIVED / "cluster_stability.json")),
            (str(FIGS / f"pca_clusters{suffix}.png"), str(FIGS / "pca_clusters.png")),
            (str(FIGS / f"umap_clusters{suffix}.png"), str(FIGS / "umap_clusters.png")),
        ]
        if tsne_xy is not None:
            canonical_pairs.append(
                (str(FIGS / f"tsne_clusters{suffix}.png"), str(FIGS / "tsne_clusters.png"))
            )
        for src, dst in canonical_pairs:
            if Path(src).exists():
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
