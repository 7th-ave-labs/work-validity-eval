"""Build a UMAP "atlas" of the O*NET task_only tasks coloured by the
18 top-level cross-job operations (v5 / pass-7 taxonomy; the
singleton ``assistance`` op is merged into ``protocol_execution``).

Only tasks that pass the round-2 task-level knowledge-work screen
(``data/derived/onet_tasks_coded.parquet``) enter the figures by
default, *plus* any clusters listed in ``FORCE_INCLUDE_CLUSTER_IDS``
(clusters whose round-2 decision was ``exclude`` but that we have
manually added back as knowledge work). Remaining excluded tasks are
carried into the interactive HTML as a separate trace that the viewer
can toggle via a three-way view selector (Included / Excluded / All).

Outputs:
  docs/figures/operation_atlas.png       static 3D figure (included-only)
  docs/figures/operation_atlas_2d.png    static 2D figure (included-only)
  docs/figures/operation_atlas_2d_3d.png side-by-side 2D vs 3D comparison
  docs/figures/operation_atlas.html      interactive Plotly 3D with view toggle

Run:
  python3 scripts/build_operation_atlas.py
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
CLUSTER_PARQUET = REPO / "data" / "derived" / "onet_task_clusters__task_only.parquet"
CODED_PARQUET = REPO / "data" / "derived" / "onet_tasks_coded.parquet"
LABELS_JSON = REPO / "data" / "derived" / "manual_labeling" / "cluster_cross_job_labels.json"
TAX_JSON = REPO / "data" / "derived" / "manual_labeling" / "operation_taxonomy.json"

# Domain-stripped operation phrases produced by normalize_task_operations.py.
# round2 is the refined pass-5 version (8,338 included tasks); the pass-3
# parquet covers all 12,464 tasks and serves as fallback for excluded ones.
NORM_R2_PARQUET = REPO / "data" / "derived" / "task_operations_normalized_round2.parquet"
NORM_PARQUET = REPO / "data" / "derived" / "task_operations_normalized.parquet"

OUT_DIR = REPO / "docs" / "figures"
OUT_HTML = OUT_DIR / "operation_atlas.html"
# Paper's primary figure is the 2D UMAP scatter (clusters are far more
# separable on 2D than on the 3D projection). The 3D solo figure is
# kept as a supplementary view and underpins the interactive HTML.
OUT_PNG = OUT_DIR / "operation_atlas.png"
OUT_PNG_3D = OUT_DIR / "operation_atlas_3d.png"

# Cached 3D UMAP coordinates for the task_only embedding space. Computed
# once on the 1,536-dim embeddings with a pinned seed, then reused.
UMAP3D_PARQUET = REPO / "data" / "derived" / "onet_task_umap3d__task_only.parquet"
EMB_NPZ = REPO / "data" / "derived" / "embeddings" / "onet_task_embeddings__task_only.npz"

# Cluster ids that the round-2 screen marked as excluded but which we
# manually re-include as knowledge work (see cluster audit in the
# paper appendix). Cluster 0 = "hold_scheduled_consultation_availability"
# (maintain regular office hours for advising) sits under ``advising``
# and is cognitive availability scheduling, not physical execution.
FORCE_INCLUDE_CLUSTER_IDS: set[int] = {0}

# Top-level operation merges applied at load time. ``assistance`` held a
# single cluster (79, "support_execution_of_expert_procedures") whose
# member tasks are predominantly clinical / surgical / medical procedures
# performed by physician assistants, nurses, dentists, surgical techs,
# respiratory therapists, etc. Those tasks are a clean fit for the
# existing ``protocol_execution`` op ("follow a prescribed, safety-
# critical procedure exactly"); folding assistance in removes a 1-cluster
# singleton and keeps the colour key compact.
MERGE_TOP_LEVEL: dict[str, str] = {"assistance": "protocol_execution"}

# Base colors for the 18 rendered top-level ops, ordered by a qualitative
# palette so adjacent ops get visually distinct hues.
OP_COLOR_ORDER = [
    "analysis",
    "administration",
    "design",
    "inspection",
    "troubleshooting",
    "record_keeping",
    "teaching",
    "protocol_execution",
    "representation",
    "coordination",
    "creative_production",
    "fabrication",
    "appraisal",
    "advising",
    "investigation",
    "emergency_response",
    "self_study",
    "rule_enforcement",
]
# ``assistance`` was folded into ``protocol_execution`` via MERGE_TOP_LEVEL;
# its palette entry is retained in OP_BASE_COLOR only as a defensive
# fallback in case the merge mapping is edited downstream.

# 19 distinct base colors in a soft pastel palette. Each hue has a
# sibling "deep" anchor so the per-cluster shade ramp still lands in the
# pastel range instead of washing out to white or crushing to black.
OP_BASE_COLOR = {
    "analysis":            "#7FB3D5",  # pastel blue
    "administration":      "#A8D5A2",  # pastel green
    "design":              "#FBC38E",  # pastel orange
    "inspection":          "#C8A2D6",  # pastel purple
    "troubleshooting":     "#F1A7A7",  # pastel red
    "record_keeping":      "#9ADBE0",  # pastel cyan
    "teaching":            "#DADB8A",  # pastel olive
    "protocol_execution":  "#C8A98A",  # pastel brown
    "representation":      "#F2B8D5",  # pastel pink
    "coordination":        "#C9C9C9",  # pastel grey
    "creative_production": "#F7D38C",  # pastel amber
    "fabrication":         "#9BCCE8",  # pastel sky
    "appraisal":           "#B5A4D3",  # pastel violet
    "advising":            "#9DCFC2",  # pastel teal
    "investigation":       "#E6A3A3",  # pastel rose
    "emergency_response":  "#F4B98A",  # pastel tangerine
    "self_study":          "#B6BCC9",  # pastel slate
    "assistance":          "#C8DB90",  # pastel lime
    "rule_enforcement":    "#D5BBA1",  # pastel tan
}


def _shade(hex_color: str, pct: float) -> str:
    """Return a shade of hex_color. pct in [-1, 1]. Negative darkens; positive lightens."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if pct >= 0:
        l = l + (1.0 - l) * pct
    else:
        l = l * (1.0 + pct)
    l = max(0.08, min(0.92, l))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def cluster_colors(sub_ops: list[str], assignments: dict[int, str], ordered_cids: list[int]) -> dict[int, str]:
    """Assign every sub-op cluster the flat base colour of its parent
    top-level op. Earlier versions ran a within-op shade ramp so
    sibling clusters were visually distinct, but it made same-cluster
    points read as darker/lighter variants once depth-shading piled
    on top. Paper figures and the interactive HTML use one colour
    per top-level op; fine-grained cluster identity lives in hover.
    """
    return {cid: OP_BASE_COLOR[assignments[cid]] for cid in ordered_cids}


def drop_cluster_outliers(xs: np.ndarray, ys: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Return a boolean mask keeping points within ``sigma`` robust-sigmas
    of the cluster centroid. Uses the MAD (median absolute deviation) as a
    scale estimate so a handful of very-far points don't inflate sigma.
    """
    if len(xs) < 4:
        return np.ones(len(xs), dtype=bool)
    cx = float(np.median(xs)); cy = float(np.median(ys))
    d = np.hypot(xs - cx, ys - cy)
    mad = float(np.median(np.abs(d - np.median(d))))
    # 1.4826 converts MAD to an estimate of \u03c3 under a gaussian model.
    scale = 1.4826 * mad if mad > 0 else float(np.std(d) + 1e-9)
    thresh = float(np.median(d)) + sigma * scale
    return d <= thresh


def smoothed_hull(
    xs: np.ndarray,
    ys: np.ndarray,
    pad: float = 0.28,
    samples: int = 220,
    smoothness: float = 0.6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a rounded, inflated polygon around (xs, ys).

    Pipeline:
      1. convex hull of the points.
      2. push each hull vertex outward from the centroid by ``pad`` UMAP
         units, so the shadow reads as a region, not a tight shrink-wrap.
      3. periodic cubic B-spline interpolation of the padded hull vertices
         to remove sharp corners and produce a rounded, organic blob.

    Returns ``None`` if there are fewer than four points.
    """
    from scipy.interpolate import splprep, splev  # local import

    if len(xs) < 4:
        return None
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    hx = pts[hull.vertices, 0]
    hy = pts[hull.vertices, 1]

    # Pad each hull vertex outward along the direction from the centroid.
    cx = float(np.mean(hx)); cy = float(np.mean(hy))
    dx = hx - cx; dy = hy - cy
    r = np.hypot(dx, dy)
    r_safe = np.maximum(r, 1e-6)
    hx_pad = cx + dx * (1.0 + pad / r_safe)
    hy_pad = cy + dy * (1.0 + pad / r_safe)

    # Close the polygon (periodic) before spline interpolation.
    hx_closed = np.append(hx_pad, hx_pad[0])
    hy_closed = np.append(hy_pad, hy_pad[0])

    if len(hx_closed) < 5:
        # Too few vertices to spline meaningfully; just return padded hull.
        return hx_closed, hy_closed
    try:
        tck, _ = splprep([hx_closed, hy_closed], s=smoothness, per=True, k=3)
        u_new = np.linspace(0.0, 1.0, samples)
        sx, sy = splev(u_new, tck)
        return np.asarray(sx), np.asarray(sy)
    except Exception:
        return hx_closed, hy_closed


def convex_hull_polygon(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 3:
        return None
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    verts = hull.vertices
    hx = xs[verts]
    hy = ys[verts]
    # close the polygon
    hx = np.append(hx, hx[0])
    hy = np.append(hy, hy[0])
    return hx, hy


def _load_umap3d(row_index: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with ``row_index, umap_x3, umap_y3, umap_z3``
    aligned to the given ``row_index`` order.

    If a cached parquet at ``UMAP3D_PARQUET`` exists, load from it.
    Otherwise, run UMAP(n_components=3) on the 1,536-dim MiniLM-L6
    embedding matrix with a pinned seed and save the cache.
    """
    if UMAP3D_PARQUET.exists():
        cached = pd.read_parquet(UMAP3D_PARQUET)
        print(f"loaded cached 3D UMAP from {UMAP3D_PARQUET.relative_to(REPO)} "
              f"({len(cached)} rows)")
        return cached

    print("computing 3D UMAP (one-time, seeded; caching to "
          f"{UMAP3D_PARQUET.relative_to(REPO)})")
    import umap
    npz = np.load(EMB_NPZ)
    emb = npz["embeddings"]
    emb_row_index = npz["row_index"].astype(int)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(emb)
    out = pd.DataFrame({
        "row_index": emb_row_index,
        "umap_x3": coords[:, 0].astype(np.float32),
        "umap_y3": coords[:, 1].astype(np.float32),
        "umap_z3": coords[:, 2].astype(np.float32),
    })
    UMAP3D_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UMAP3D_PARQUET, index=False)
    return out


def main() -> None:
    df = pd.read_parquet(CLUSTER_PARQUET)
    # Attach 3D UMAP coords (computed from the MiniLM embeddings, seeded).
    umap3d = _load_umap3d(df["row_index"].to_numpy())
    df = df.merge(umap3d, on="row_index", how="left")
    labels_doc = json.loads(LABELS_JSON.read_text())
    tax_doc = json.loads(TAX_JSON.read_text())

    # cluster_id -> (sub_op, top_level, note, n_tasks, n_occupations).
    # ``MERGE_TOP_LEVEL`` remaps the top_level label at load time so
    # downstream coverage stats, colour assignment, legend, and hover
    # cards all see the merged taxonomy without editing the source JSON.
    per_cluster = {
        int(c["cluster_id"]): {
            "sub_op":        c["sub_op"],
            "top_level":     MERGE_TOP_LEVEL.get(c["top_level"], c["top_level"]),
            "note":          c.get("note", ""),
            "n_tasks":       c.get("n_tasks"),
            "n_occupations": c.get("n_occupations"),
        }
        for c in labels_doc["clusters"]
    }

    # attach to tasks
    df = df.copy()
    df["cluster_id"]   = df["hdbscan_cluster_assigned"].astype(int)
    df = df[df["cluster_id"].isin(per_cluster.keys())].copy()
    df["sub_op"]    = df["cluster_id"].map(lambda c: per_cluster[c]["sub_op"])
    df["top_level"] = df["cluster_id"].map(lambda c: per_cluster[c]["top_level"])

    assert len(df) == 12464, f"unexpected row count: {len(df)}"

    # Mark inclusion per the round-2 task-level knowledge-work screen,
    # then fold in FORCE_INCLUDE_CLUSTER_IDS so manually-audited clusters
    # that the round-2 screen got wrong count as included too.
    coded = pd.read_parquet(CODED_PARQUET)
    included_keys = set(zip(coded["onetsoc_code"], coded["task_id"]))
    screen_included = np.array([
        (soc, tid) in included_keys
        for soc, tid in zip(df["onetsoc_code"], df["task_id"])
    ])
    force_mask = df["cluster_id"].isin(FORCE_INCLUDE_CLUSTER_IDS).to_numpy()
    df["included"] = screen_included | force_mask
    n_incl = int(df["included"].sum()); n_excl = int((~df["included"]).sum())
    n_force = int((force_mask & ~screen_included).sum())
    print(
        f"task partition: included={n_incl}, excluded={n_excl}, total={len(df)} "
        f"(force-included clusters {sorted(FORCE_INCLUDE_CLUSTER_IDS)} "
        f"added {n_force} rows)"
    )

    # Attach the domain-stripped operation phrase from normalize_task_operations.py.
    # Prefer the round-2 refined output (8,338 included) and fall back to the
    # pass-3 all-tasks output (12,464) for excluded rows.
    norm_r2 = pd.read_parquet(NORM_R2_PARQUET)[["onetsoc_code", "task_id", "operation_phrase"]].copy()
    norm_r2["task_id"] = norm_r2["task_id"].astype(str)
    norm_r2 = norm_r2.rename(columns={"operation_phrase": "operation_phrase_r2"})
    norm = pd.read_parquet(NORM_PARQUET)[["onetsoc_code", "task_id", "operation_phrase"]].copy()
    norm["task_id"] = norm["task_id"].astype(str)
    norm = norm.rename(columns={"operation_phrase": "operation_phrase_v1"})
    df["task_id"] = df["task_id"].astype(str)
    df = df.merge(norm_r2, on=["onetsoc_code", "task_id"], how="left")
    df = df.merge(norm, on=["onetsoc_code", "task_id"], how="left")
    df["operation_phrase"] = (
        df["operation_phrase_r2"].fillna(df["operation_phrase_v1"]).fillna("")
    )
    df = df.drop(columns=["operation_phrase_r2", "operation_phrase_v1"])
    print(
        f"operation_phrase coverage: "
        f"{int((df['operation_phrase'].str.len() > 0).sum())}/{len(df)} rows"
    )

    # Per-op included-task counts, for figure legends. The 19-op
    # operation_taxonomy.json totals are computed from the full 12,464
    # corpus; once we filter down to included-only, a handful of ops
    # (notably fabrication, emergency_response, some representation /
    # self_study sub-ops) shrink substantially.
    included_df = df[df["included"]].copy()
    excluded_df = df[~df["included"]].copy()

    # Per-cluster MAD-based outlier removal on the 3D UMAP coords: drop
    # points beyond 2 robust sigmas of the cluster centroid so a handful
    # of soft-assigned tails don't dominate the display volume.
    def _drop_outliers_3d(xs, ys, zs, sigma=2.0):
        if len(xs) < 4:
            return np.ones(len(xs), dtype=bool)
        cx = float(np.median(xs)); cy = float(np.median(ys)); cz = float(np.median(zs))
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2)
        mad = float(np.median(np.abs(d - np.median(d))))
        scale = 1.4826 * mad if mad > 0 else float(np.std(d) + 1e-9)
        return d <= float(np.median(d)) + sigma * scale

    included_df["kept"] = False
    for cid, sub in included_df.groupby("cluster_id", sort=False):
        mask = _drop_outliers_3d(
            sub["umap_x3"].to_numpy(),
            sub["umap_y3"].to_numpy(),
            sub["umap_z3"].to_numpy(),
            sigma=2.0,
        )
        included_df.loc[sub.index, "kept"] = mask
    plot_df = included_df[included_df["kept"]].copy()
    n_kept = len(plot_df); n_drop = n_incl - n_kept
    print(f"outlier trim: kept={n_kept}, dropped={n_drop} (of {n_incl} included)")

    # Legend counts report the *raw* included population (pre-outlier-trim)
    # so the figure / caption / examples table all agree on per-op totals.
    op_incl_stats = (
        included_df.groupby("top_level")
        .agg(n_tasks_incl=("task_id", "size"), n_clusters_incl=("cluster_id", "nunique"))
        .to_dict("index")
    )

    # per-cluster colors (shade within op)
    sub_ops = sorted({per_cluster[c]["sub_op"] for c in per_cluster})
    cids_ordered = sorted(per_cluster.keys(), key=lambda c: (per_cluster[c]["top_level"], per_cluster[c]["sub_op"]))
    assignments = {c: per_cluster[c]["top_level"] for c in per_cluster}
    ccolor = cluster_colors(sub_ops, assignments, cids_ordered)

    # one-line / distinct_from lookup for hover info on shadow regions
    op_def = {o["operation_id"]: o for o in tax_doc["top_level_operations"]}

    # ---------- Plotly: interactive 3D HTML ----------
    # Plotly 3D scenes use left-drag to orbit and the scroll-wheel to zoom
    # by default; we pin both explicitly via the config. No hull shadows in
    # 3D -- they read as opaque blobs that obscure the points. Visual grouping
    # comes from the pastel colour families alone.
    fig = go.Figure()

    def _wrap(s: str, n: int = 70) -> str:
        """Soft-wrap a long string with <br> every ~n chars, preserving words
        and escaping <,>,&. Keeps the Plotly hover box a fixed visual width."""
        if not isinstance(s, str):
            s = "" if s is None else str(s)
        s = (s.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
        words = s.split()
        out: list[str] = []
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > n and line:
                out.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line:
            out.append(line)
        return "<br>".join(out)

    # One proxy legend entry per top-level op. Scatter3d legend items
    # toggle the whole (legendgroup) — shared with the task trace below —
    # so clicking an op entry hides/shows all of its tasks at once.
    for op_id in OP_COLOR_ORDER:
        op_meta = op_def.get(op_id, {})
        s = op_incl_stats.get(op_id, {"n_tasks_incl": 0, "n_clusters_incl": 0})
        hover_text = (
            f"<b>{op_id}</b><br>"
            f"{_wrap(op_meta.get('one_line',''), 60)}<br>"
            f"<i>Clusters (incl.):</i> {s['n_clusters_incl']}  "
            f"<i>Tasks (incl.):</i> {s['n_tasks_incl']}<br>"
            f"<i>Clusters (all):</i> {op_meta.get('n_member_clusters','?')}  "
            f"<i>Tasks (all):</i> {op_meta.get('total_n_tasks','?')}"
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=7, color=OP_BASE_COLOR[op_id], symbol="square",
                            line=dict(width=1, color="rgba(0,0,0,0.25)")),
                name=op_id,
                legendgroup=op_id,
                hoverinfo="text",
                text=hover_text,
                showlegend=True,
            )
        )

    # Included tasks: one Scatter3d trace per top-level op.
    for op_id in OP_COLOR_ORDER:
        sub = plot_df[plot_df["top_level"] == op_id]
        if sub.empty:
            continue
        # Pre-wrap the task text once so the hover label width is stable.
        wrapped_tasks = [_wrap(t, 70) for t in sub["task_text_clean"].tolist()]
        wrapped_norm = [_wrap(t, 70) for t in sub["operation_phrase"].tolist()]
        customdata = np.column_stack([
            sub["occupation_title"].astype(str).to_numpy(),
            sub["cluster_id"].astype(int).astype(str).to_numpy(),
            sub["sub_op"].astype(str).to_numpy(),
            sub["top_level"].astype(str).to_numpy(),
            np.asarray(wrapped_tasks, dtype=object),
            np.asarray(wrapped_norm, dtype=object),
        ])
        colors = [ccolor[int(c)] for c in sub["cluster_id"]]
        fig.add_trace(
            go.Scatter3d(
                x=sub["umap_x3"], y=sub["umap_y3"], z=sub["umap_z3"],
                mode="markers",
                marker=dict(
                    size=3.2, color=colors, opacity=0.95,
                    line=dict(width=0),
                ),
                name=op_id + " (tasks)",
                legendgroup=op_id,
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>op:</b> %{customdata[3]}  "
                    "<span style='color:#888'>(sub-op: %{customdata[2]})</span><br>"
                    "<b>job:</b> %{customdata[0]}<br>"
                    "<b>original task:</b><br>%{customdata[4]}<br>"
                    "<b>normalized:</b> %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )

    # Excluded tasks as a single grey trace, hidden by default.
    wrapped_exc = [_wrap(t, 70) for t in excluded_df["task_text_clean"].tolist()]
    wrapped_exc_norm = [_wrap(t, 70) for t in excluded_df["operation_phrase"].tolist()]
    exc_custom = np.column_stack([
        excluded_df["occupation_title"].astype(str).to_numpy(),
        excluded_df["cluster_id"].astype(int).astype(str).to_numpy(),
        excluded_df["sub_op"].astype(str).to_numpy(),
        excluded_df["top_level"].astype(str).to_numpy(),
        np.asarray(wrapped_exc, dtype=object),
        np.asarray(wrapped_exc_norm, dtype=object),
    ])
    fig.add_trace(
        go.Scatter3d(
            x=excluded_df["umap_x3"], y=excluded_df["umap_y3"], z=excluded_df["umap_z3"],
            mode="markers",
            marker=dict(size=2.0, color="#cccccc", opacity=0.45, line=dict(width=0)),
            name=f"Excluded (non-knowledge-work) — {len(excluded_df)} tasks",
            legendgroup="__excluded__",
            visible="legendonly",
            customdata=exc_custom,
            hovertemplate=(
                "<b>EXCLUDED</b> (non-knowledge-work, round-2 screen)<br>"
                "<b>op:</b> %{customdata[3]}  "
                "<span style='color:#888'>(sub-op: %{customdata[2]})</span><br>"
                "<b>job:</b> %{customdata[0]}<br>"
                "<b>original task:</b><br>%{customdata[4]}<br>"
                "<b>normalized:</b> %{customdata[5]}"
                "<extra></extra>"
            ),
        )
    )
    excluded_trace_idx = len(fig.data) - 1
    n_traces = len(fig.data)

    # Three visibility modes for the view-selector buttons. A trace is
    # either a 19-op legend proxy, a per-op task trace, or the single
    # excluded trace at ``excluded_trace_idx``. Legend proxies stay
    # visible in every mode so the colour key remains readable.
    def _is_legend_proxy(trace: go.Scatter3d) -> bool:
        return bool(getattr(trace, "showlegend", False))

    vis_included: list = []
    vis_excluded: list = []
    vis_all: list = []
    for i, tr in enumerate(fig.data):
        if i == excluded_trace_idx:
            vis_included.append("legendonly")
            vis_excluded.append(True)
            vis_all.append(True)
        elif _is_legend_proxy(tr):
            vis_included.append(True); vis_excluded.append(True); vis_all.append(True)
        else:
            vis_included.append(True)
            vis_excluded.append("legendonly")
            vis_all.append(True)

    # Camera: a gentle perspective view so the volume reads as 3D on load.
    scene = dict(
        xaxis=dict(title="", showticklabels=False, showgrid=True, gridcolor="#eeeeee",
                   zeroline=False, showbackground=True, backgroundcolor="#fafafa"),
        yaxis=dict(title="", showticklabels=False, showgrid=True, gridcolor="#eeeeee",
                   zeroline=False, showbackground=True, backgroundcolor="#fafafa"),
        zaxis=dict(title="", showticklabels=False, showgrid=True, gridcolor="#eeeeee",
                   zeroline=False, showbackground=True, backgroundcolor="#fafafa"),
        camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        aspectmode="cube",
        dragmode="orbit",  # left-drag orbits the scene
        bgcolor="#ffffff",
    )

    fig.update_layout(
        title=None,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color="#1f2937"),
        scene=scene,
        legend=dict(
            title=dict(text=f"<b>Top-level operations ({len(OP_COLOR_ORDER)})</b>", font=dict(size=12)),
            orientation="v",
            x=1.005, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.12)", borderwidth=1,
            itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers",
            font=dict(size=11),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        width=1400, height=850,
        margin=dict(l=0, r=280, t=40, b=0),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="rgba(0,0,0,0.2)",
            font=dict(family="Inter, Helvetica, Arial, sans-serif",
                      size=11, color="#1f2937"),
            align="left",
            namelength=-1,  # never auto-truncate
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.0, y=1.02, xanchor="left", yanchor="bottom",
                pad=dict(r=6, t=2, b=2, l=2),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.12)",
                font=dict(size=11),
                buttons=[
                    dict(label=f"Included only ({n_incl}) \u2014 default",
                         method="update", args=[{"visible": vis_included}]),
                    dict(label=f"Excluded only ({n_excl})",
                         method="update", args=[{"visible": vis_excluded}]),
                    dict(label=f"All ({n_incl + n_excl})",
                         method="update", args=[{"visible": vis_all}]),
                ],
            )
        ],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Inject a tiny CSS rule so every Plotly hover label is a fixed width
    # (max-width + word-wrap). Scatter3d doesn't expose a direct width prop,
    # so this is the cleanest way to pin the hover box size.
    fixed_width_css = """
    <style>
      /* pin the Plotly hover tooltip to a stable width */
      .hoverlayer .hovertext { max-width: 360px; }
      .hoverlayer .hovertext text { font-family: Inter, Helvetica, Arial, sans-serif; }
    </style>
    """
    pio.write_html(
        fig, file=OUT_HTML,
        include_plotlyjs="cdn", full_html=True, auto_open=False,
        post_script=None,
        config={
            "scrollZoom": True,              # scroll wheel zooms the 3D scene
            "displaylogo": False,
            "doubleClick": "reset",
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    # Prepend the CSS into the <head> so it's applied on load.
    html_text = OUT_HTML.read_text()
    if "<head>" in html_text and fixed_width_css.strip() not in html_text:
        html_text = html_text.replace("<head>", "<head>" + fixed_width_css, 1)
        OUT_HTML.write_text(html_text)
    print(f"wrote {OUT_HTML.relative_to(REPO)}")

    # ---------- static PNGs (publication quality) ----------
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3d

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
    })

    # The 2D UMAP columns (umap_x, umap_y) were already attached upstream
    # from the clustering parquet. The 2D panel reuses them so both views
    # show the same tasks with the same colours.
    #
    # For the 2D panel we additionally re-run the same MAD outlier trim on
    # the 2D coords so a handful of far-flung soft-assigned tails don't
    # force the axes wide and shrink the main blob.
    plot_df_2d = included_df[included_df["kept"]].copy()
    plot_df_2d["kept2d"] = False
    for cid, sub in plot_df_2d.groupby("cluster_id", sort=False):
        mask = drop_cluster_outliers(
            sub["umap_x"].to_numpy(),
            sub["umap_y"].to_numpy(),
            sigma=2.0,
        )
        plot_df_2d.loc[sub.index, "kept2d"] = mask
    plot_df_2d = plot_df_2d[plot_df_2d["kept2d"]].copy()

    xs3 = plot_df["umap_x3"].to_numpy()
    ys3 = plot_df["umap_y3"].to_numpy()
    zs3 = plot_df["umap_z3"].to_numpy()
    xs2 = plot_df_2d["umap_x"].to_numpy()
    ys2 = plot_df_2d["umap_y"].to_numpy()
    colors3d = [ccolor[int(c)] for c in plot_df["cluster_id"]]
    colors2d = [ccolor[int(c)] for c in plot_df_2d["cluster_id"]]

    def _lim(v, pct=(2, 98), pad_frac=0.03):
        lo, hi = np.percentile(v, pct); pad = pad_frac * (hi - lo)
        return lo - pad, hi + pad

    def _draw_3d(ax):
        # depthshade=False keeps a cluster's colour constant regardless
        # of camera depth so the legend colour matches every point.
        ax.scatter(xs3, ys3, zs3, s=4.0, c=colors3d, alpha=0.88,
                   linewidths=0, depthshade=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_edgecolor("#dcdcdc")
            pane.pane.set_facecolor((1, 1, 1, 1))
            pane.pane.set_alpha(1.0)
        ax.grid(True)
        try:
            ax.xaxis._axinfo["grid"].update({"color": "#f0f0f0", "linewidth": 0.4})
            ax.yaxis._axinfo["grid"].update({"color": "#f0f0f0", "linewidth": 0.4})
            ax.zaxis._axinfo["grid"].update({"color": "#f0f0f0", "linewidth": 0.4})
        except Exception:
            pass
        ax.set_xlim(*_lim(xs3)); ax.set_ylim(*_lim(ys3)); ax.set_zlim(*_lim(zs3))
        ax.view_init(elev=18, azim=35)
        try:
            ax.set_box_aspect((1, 1, 0.9))
        except Exception:
            pass

    def _draw_2d(ax):
        ax.scatter(xs2, ys2, s=6.0, c=colors2d, alpha=0.85, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_color("#dcdcdc"); spine.set_linewidth(0.6)
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#f0f0f0", linewidth=0.4)
        ax.set_xlim(*_lim(xs2)); ax.set_ylim(*_lim(ys2))
        ax.set_aspect("equal")

    def _legend_label(op: str) -> str:
        n_c = op_incl_stats.get(op, {}).get("n_clusters_incl", 0)
        n_t = op_incl_stats.get(op, {}).get("n_tasks_incl", 0)
        return f"{op.replace('_', ' ')}  ({n_c}c · {n_t:,})"

    def _legend_handles():
        return [
            mpatches.Patch(
                facecolor=OP_BASE_COLOR[op],
                edgecolor=OP_BASE_COLOR[op],
                alpha=0.85,
                label=_legend_label(op),
            )
            for op in OP_COLOR_ORDER
        ]

    def _attach_legend(ax, *, bbox=(1.02, 0.5), fontsize=9.0):
        leg = ax.legend(
            handles=_legend_handles(),
            title=f"Top-level operations ({len(OP_COLOR_ORDER)})",
            loc="center left",
            bbox_to_anchor=bbox,
            ncol=1,
            fontsize=fontsize, title_fontsize=fontsize + 1,
            frameon=True, fancybox=False,
            edgecolor="#d0d0d0",
            labelspacing=0.55, handlelength=1.4, handleheight=1.0,
            borderpad=0.7,
        )
        leg.get_frame().set_linewidth(0.6)
        leg.get_frame().set_facecolor("#ffffff")
        leg.get_title().set_fontweight("bold")
        return leg

    # Solo 2D PNG — paper's primary figure (clusters are most separable
    # in 2D; the 3D view compresses mid-space clusters).
    fig2d = plt.figure(figsize=(10, 7.0), dpi=200)
    ax2d = fig2d.add_subplot(111)
    _draw_2d(ax2d)
    _attach_legend(ax2d, bbox=(1.02, 0.5), fontsize=9.0)
    fig2d.tight_layout()
    fig2d.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG.relative_to(REPO)}")
    plt.close(fig2d)

    # Solo 3D PNG — supplementary view; same tasks / colours.
    fig3d = plt.figure(figsize=(11, 7.0), dpi=200)
    ax3d = fig3d.add_subplot(111, projection="3d")
    _draw_3d(ax3d)
    _attach_legend(ax3d, bbox=(1.04, 0.5), fontsize=9.0)
    fig3d.tight_layout()
    fig3d.savefig(OUT_PNG_3D, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG_3D.relative_to(REPO)}")
    plt.close(fig3d)

    # Side-by-side 2D vs 3D comparison with a single shared legend.
    out_png_cmp = OUT_DIR / "operation_atlas_2d_3d.png"
    fig_cmp = plt.figure(figsize=(16, 7.5), dpi=200)
    gs = fig_cmp.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.32], wspace=0.02)
    ax_cmp2d = fig_cmp.add_subplot(gs[0, 0])
    ax_cmp3d = fig_cmp.add_subplot(gs[0, 1], projection="3d")
    ax_legend = fig_cmp.add_subplot(gs[0, 2]); ax_legend.axis("off")
    _draw_2d(ax_cmp2d)
    _draw_3d(ax_cmp3d)
    ax_cmp2d.set_title("(a) 2D UMAP", fontsize=11, loc="left", pad=6)
    ax_cmp3d.set_title("(b) 3D UMAP", fontsize=11, loc="left", pad=6)
    leg = ax_legend.legend(
        handles=_legend_handles(),
        title=f"Top-level operations ({len(OP_COLOR_ORDER)})",
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        fontsize=8.5, title_fontsize=9.5,
        frameon=True, fancybox=False,
        edgecolor="#d0d0d0",
        labelspacing=0.55, handlelength=1.4, handleheight=1.0,
        borderpad=0.7,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_title().set_fontweight("bold")
    fig_cmp.savefig(out_png_cmp, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png_cmp.relative_to(REPO)}")
    plt.close(fig_cmp)


if __name__ == "__main__":
    main()
