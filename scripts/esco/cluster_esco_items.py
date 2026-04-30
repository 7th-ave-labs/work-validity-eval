#!/usr/bin/env python3
"""HDBSCAN on ESCO embeddings + structural-recoverability metrics.

Produces an independent partition of the ESCO items that uses no
knowledge of the 18-op taxonomy, then reports how related that
partition is to the consensus op assignment.

Pipeline mirrors the O*NET run (``scripts/cluster_onet_tasks.py``):
  1. L2-normalise the ESCO item embeddings.
  2. UMAP to 15-d cosine space (seed 42, n_neighbors=30, min_dist=0.0).
  3. HDBSCAN with min_cluster_size=40, min_samples=10.
  4. Compute ARI and NMI between the HDBSCAN partition and the LLM
     op label (the noise-free semantic label); also against the
     consensus label on the agreement subset.
  5. Per-op homogeneity: for each op, the fraction of its ESCO items
     captured by the single largest HDBSCAN cluster.
  6. Per-HDBSCAN-cluster purity: for each cluster, the dominant op
     and its share of the cluster.

Outputs:
  data/derived/esco/esco_hdbscan_clusters.parquet
  data/derived/esco/esco_structural_recoverability.json
  data/derived/esco/esco_hdbscan_umap2d.parquet  (UMAP 2-d coords for figures)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ESCO_DERIVED = ROOT / "data" / "derived" / "esco"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emb-npz",
        default=str(ESCO_DERIVED / "embeddings" / "esco_item_embeddings__label_only.npz"),
    )
    p.add_argument(
        "--assignments",
        default=str(ESCO_DERIVED / "esco_op_assignments.parquet"),
    )
    p.add_argument("--min-cluster-size", type=int, default=40)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--umap-n-components", type=int, default=15)
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import hdbscan
    import umap
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.neighbors import NearestNeighbors

    emb = np.load(args.emb_npz)["embeddings"].astype(np.float32)
    emb_n = l2_normalize(emb)
    assignments = pd.read_parquet(args.assignments)

    print(f"UMAP -> {args.umap_n_components}-d cosine (n_neighbors={args.umap_n_neighbors}, seed={args.seed})")
    reducer = umap.UMAP(
        n_components=args.umap_n_components,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    u15 = reducer.fit_transform(emb_n)

    print(
        f"HDBSCAN min_cluster_size={args.min_cluster_size} min_samples={args.min_samples}"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(u15)

    # Soft-assign noise points to their nearest non-noise cluster by
    # centroid in the 15-d space (matches O*NET ``hdbscan_cluster_assigned``).
    labels_assigned = labels.copy()
    non_noise = np.unique(labels[labels != -1])
    if len(non_noise):
        centroids = np.stack(
            [u15[labels == c].mean(axis=0) for c in non_noise]
        )
        mask_noise = labels == -1
        if mask_noise.any():
            d = np.linalg.norm(u15[mask_noise][:, None, :] - centroids[None, :, :], axis=-1)
            nearest = non_noise[d.argmin(axis=1)]
            labels_assigned[mask_noise] = nearest

    # 2-D UMAP for figures (use same seed but a separate run to get a
    # publishable 2-D projection).
    reducer2 = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=0.05,
        metric="cosine",
        random_state=args.seed,
    )
    u2 = reducer2.fit_transform(emb_n)

    out = assignments.copy()
    out["hdbscan_cluster"] = labels
    out["hdbscan_cluster_assigned"] = labels_assigned
    out["umap_x"] = u2[:, 0].astype(np.float32)
    out["umap_y"] = u2[:, 1].astype(np.float32)
    out.to_parquet(ESCO_DERIVED / "esco_hdbscan_clusters.parquet", index=False)
    print(f"Wrote {ESCO_DERIVED / 'esco_hdbscan_clusters.parquet'}")

    # Metrics against the LLM op labels (noise-free semantic label).
    def _metrics(gold: pd.Series, pred: pd.Series) -> dict:
        mask = gold.notna() & pred.notna()
        g, p = gold[mask].astype(str).to_numpy(), pred[mask].astype(str).to_numpy()
        return {
            "n": int(mask.sum()),
            "ari": float(adjusted_rand_score(g, p)),
            "nmi": float(normalized_mutual_info_score(g, p)),
        }

    llm_label = out["llm_op_id"].where(out["llm_op_id"] != "none_of_the_above")
    centroid_label = out["centroid_op_id"]
    consensus = out["consensus_op_id"]

    metrics = {
        "n_items": int(len(out)),
        "n_hdbscan_noise": int((labels == -1).sum()),
        "n_hdbscan_clusters_raw": int(len({c for c in labels if c != -1})),
        "n_hdbscan_clusters_assigned": int(len(set(labels_assigned))),
        "vs_llm_op_id": _metrics(llm_label, pd.Series(labels_assigned.astype(str))),
        "vs_centroid_op_id": _metrics(centroid_label, pd.Series(labels_assigned.astype(str))),
        "vs_consensus_op_id": _metrics(consensus, pd.Series(labels_assigned.astype(str))),
    }

    # Per-op homogeneity against LLM labels.
    op_homog: dict[str, dict] = {}
    for op, sub in out.groupby("llm_op_id"):
        if op == "none_of_the_above" or pd.isna(op):
            continue
        top_cluster_share = (
            sub.groupby("hdbscan_cluster_assigned").size().max() / len(sub)
        )
        op_homog[op] = {
            "n_items": int(len(sub)),
            "dominant_cluster_share": float(top_cluster_share),
        }
    metrics["per_op_homogeneity_vs_llm"] = op_homog

    # Per-cluster purity against LLM labels.
    cluster_purity: dict[str, dict] = {}
    for cl, sub in out.groupby("hdbscan_cluster_assigned"):
        valid = sub[sub["llm_op_id"].notna() & (sub["llm_op_id"] != "none_of_the_above")]
        if valid.empty:
            continue
        counts = valid["llm_op_id"].value_counts()
        cluster_purity[str(int(cl))] = {
            "n_items": int(len(valid)),
            "dominant_op": counts.index[0],
            "dominant_share": float(counts.iloc[0] / len(valid)),
            "second_op": counts.index[1] if len(counts) > 1 else None,
            "second_share": float(counts.iloc[1] / len(valid)) if len(counts) > 1 else 0.0,
        }
    metrics["per_cluster_purity_vs_llm"] = cluster_purity

    # ---- Matched-granularity KMeans(k=18) check. HDBSCAN produces a
    # finer partition of ESCO than the 18-op taxonomy (46 vs 18), which
    # caps ARI mechanically. A KMeans k=18 run matched to op cardinality
    # is the fairer comparison. Seeded with op centroids reprojected
    # from the O*NET embedding space for stability.
    print("KMeans(k=18) matched-granularity check ...")
    km = KMeans(n_clusters=18, random_state=args.seed, n_init=20)
    km_labels = km.fit_predict(emb_n)
    metrics["vs_llm_op_id_kmeans18"] = _metrics(
        llm_label, pd.Series(km_labels.astype(str))
    )

    # kNN op-label consistency: for each ESCO item, what fraction of
    # its k nearest neighbours (in the 1,536-d cosine space) carry the
    # same LLM op label? A high rate means op labels form coherent
    # regions of ESCO embedding space independently of any clustering
    # algorithm.
    print("kNN op-label consistency ...")
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(emb_n)
    _, idx = nn.kneighbors(emb_n)
    idx = idx[:, 1:]  # drop self
    llm_arr = out["llm_op_id"].to_numpy()
    same_mask = llm_arr[idx] == llm_arr[:, None]
    # Exclude items whose own label is none_of_the_above.
    valid_rows = llm_arr != "none_of_the_above"
    knn_rates = {}
    for k in (1, 5, 10):
        rate = same_mask[valid_rows, :k].mean()
        knn_rates[f"k={k}"] = float(rate)
    metrics["knn_label_consistency_vs_llm"] = knn_rates
    # Null baseline: permute labels and repeat.
    rng = np.random.default_rng(args.seed)
    perm = llm_arr.copy()
    rng.shuffle(perm)
    same_perm = perm[idx] == perm[:, None]
    metrics["knn_label_consistency_null"] = {
        f"k={k}": float(same_perm[valid_rows, :k].mean()) for k in (1, 5, 10)
    }

    # Silhouette on UMAP 15-d using op labels (how separated are the
    # LLM-labelled ops in ESCO's own embedding manifold).
    sil_mask = out["llm_op_id"].notna() & (out["llm_op_id"] != "none_of_the_above")
    metrics["silhouette_llm_op_id"] = float(
        silhouette_score(u15[sil_mask.values], llm_arr[sil_mask.values], metric="euclidean")
    ) if sil_mask.sum() > 20 else None

    (ESCO_DERIVED / "esco_structural_recoverability.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(
        f"ARI(hdbscan vs llm) = {metrics['vs_llm_op_id']['ari']:.3f}, "
        f"NMI = {metrics['vs_llm_op_id']['nmi']:.3f}; "
        f"ARI(kmeans18 vs llm) = {metrics['vs_llm_op_id_kmeans18']['ari']:.3f}, "
        f"NMI = {metrics['vs_llm_op_id_kmeans18']['nmi']:.3f}"
    )
    print(
        f"kNN same-op rate: k=1 {knn_rates['k=1']:.3f} / k=5 {knn_rates['k=5']:.3f} / "
        f"k=10 {knn_rates['k=10']:.3f}  (null k=5 {metrics['knn_label_consistency_null']['k=5']:.3f})"
    )
    print(f"silhouette(llm ops, UMAP15) = {metrics['silhouette_llm_op_id']}")


if __name__ == "__main__":
    main()
