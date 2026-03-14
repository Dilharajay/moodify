"""
app.py — AestheteAI Streamlit UI
==================================

Entry point for the web application. Run with:
    streamlit run app.py

This UI is a display layer over pre-computed results produced by the CLI
pipeline. It does not run the pipeline itself — scraping, embedding, and
clustering are all done via:
    python main.py --action pipeline --keyword "your keyword"

Architecture decision: why pre-computed only?
----------------------------------------------
Streamlit is a display framework, not a job runner. The full AestheteAI
pipeline (Playwright scraping + CLIP embeddings + UMAP + HDBSCAN) takes
several minutes and requires significant RAM. Running it inside a Streamlit
session would:
  - Hit Hugging Face Spaces' 16GB memory limit during CLIP inference
  - Timeout Streamlit's 60s default request limit during UMAP
  - Block the UI thread, making the app unresponsive

The correct pattern for ML demo apps is: compute offline, display online.
The Dashboard page shows clearly which pipeline steps are complete and which
still need to be run via CLI.

Page structure:
    Sidebar navigation → 4 pages
    Page 1: Dashboard       — DB stats and pipeline status per keyword
    Page 2: Cluster Gallery — mood board grid for a selected keyword
    Page 3: Scatter Plot    — interactive 2D UMAP cluster visualization
    Page 4: Download Centre — PNG and PDF download buttons per board
"""

import os
import json
import glob
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.database import get_connection, initialize_db, get_stats, migrate_schema_phase4
from utils.database import migrate_schema_phase5
from config import DB_PATH, EMBEDDINGS_DIR


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title   = "AestheteAI",
    page_icon    = "🎨",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS — tighten spacing and style the color swatches inline
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .swatch-row { display: flex; gap: 6px; margin-top: 4px; }
    .swatch { width: 24px; height: 24px; border-radius: 4px;
               border: 1px solid rgba(255,255,255,0.15); }
    .cluster-card { background: #0f0f1a; border-radius: 8px;
                    padding: 12px; margin-bottom: 8px; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #E94560; }
    .stat-label  { font-size: 0.8rem; color: #888; text-transform: uppercase;
                   letter-spacing: 0.08em; }
    .pipeline-ok   { color: #2ECC71; }
    .pipeline-warn { color: #F39C12; }
    .pipeline-none { color: #666; }
</style>
""", unsafe_allow_html=True)


# ─── DB Connection (cached across reruns) ─────────────────────────────────────

@st.cache_resource
def get_db():
    """
    Open and cache the database connection for the lifetime of the Streamlit
    session. st.cache_resource ensures only one connection is created even
    when multiple users are connected simultaneously.
    """
    conn = initialize_db(DB_PATH)
    migrate_schema_phase4(conn)
    migrate_schema_phase5(conn)
    return conn


# ─── Data Loaders (cached with TTL) ───────────────────────────────────────────

@st.cache_data(ttl=30)
def load_stats():
    """Reload stats every 30 seconds so the dashboard stays reasonably fresh."""
    conn = get_db()
    return get_stats(conn)


@st.cache_data(ttl=30)
def load_keywords():
    """Return all distinct keywords that have at least one clustered image."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT DISTINCT keyword
        FROM images
        WHERE cluster_id IS NOT NULL
        ORDER BY keyword
        """
    ).fetchall()
    return [r["keyword"] for r in rows]


@st.cache_data(ttl=30)
def load_all_keywords():
    """Return all distinct keywords in the DB regardless of clustering status."""
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT keyword FROM images ORDER BY keyword"
    ).fetchall()
    return [r["keyword"] for r in rows]


@st.cache_data(ttl=60)
def load_clusters_for_keyword(keyword: str):
    """
    Load all cluster metadata for a keyword.
    Returns a list of dicts with cluster_id, cluster_label, image_count.
    """
    conn = get_db()
    rows = conn.execute(
        """
        SELECT cluster_id, cluster_label, COUNT(*) as image_count
        FROM images
        WHERE keyword    = ?
          AND cluster_id IS NOT NULL
          AND cluster_id != -1
        GROUP BY cluster_id, cluster_label
        ORDER BY cluster_id
        """,
        (keyword,)
    ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=60)
def load_cluster_images(keyword: str, cluster_id: int, limit: int = 12):
    """Load local_path and dominant_colors for up to `limit` images in a cluster."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, local_path, dominant_colors, title
        FROM images
        WHERE keyword    = ?
          AND cluster_id = ?
          AND local_path IS NOT NULL
        LIMIT ?
        """,
        (keyword, cluster_id, limit)
    ).fetchall()
    return [dict(r) for r in rows]


@st.cache_data(ttl=120)
def load_umap_data(keyword: str):
    """
    Load the 2D UMAP projection and cluster labels for the scatter plot.
    Returns (coords_2d, labels, label_names) or (None, None, None) if not found.
    """
    safe_kw  = keyword.strip().lower().replace(" ", "_")
    umap_path = os.path.join(EMBEDDINGS_DIR, f"umap2_{safe_kw}.npy")

    if not os.path.exists(umap_path):
        return None, None, None

    coords_2d = np.load(umap_path)

    conn = get_db()
    rows = conn.execute(
        """
        SELECT cluster_id, cluster_label
        FROM images
        WHERE keyword = ?
          AND image_embedding_path IS NOT NULL
          AND color_histogram IS NOT NULL
        ORDER BY id
        """,
        (keyword,)
    ).fetchall()

    labels      = np.array([r["cluster_id"] if r["cluster_id"] is not None else -1
                             for r in rows])
    label_names = {r["cluster_id"]: r["cluster_label"] or f"cluster {r['cluster_id']}"
                   for r in rows if r["cluster_id"] is not None}

    # Guard against shape mismatch (can happen if DB was updated after matrix was built)
    if len(labels) != len(coords_2d):
        min_len   = min(len(labels), len(coords_2d))
        labels    = labels[:min_len]
        coords_2d = coords_2d[:min_len]

    return coords_2d, labels, label_names


def find_board_files(keyword: str) -> list:
    """
    Find all PNG and PDF board files generated for a keyword.
    Returns a list of dicts with label, png_path, pdf_path.
    """
    safe_kw   = keyword.strip().lower().replace(" ", "_")
    board_dir = os.path.join("output", "boards", safe_kw)

    if not os.path.exists(board_dir):
        return []

    png_files = sorted(glob.glob(os.path.join(board_dir, "*.png")))
    results   = []

    for png_path in png_files:
        base     = os.path.splitext(png_path)[0]
        pdf_path = base + ".pdf"
        name     = os.path.basename(base)
        # Strip the "board_cN_" prefix for a clean label
        label    = name.replace("board_c", "").split("_", 1)[-1].replace("_", " ")
        results.append({
            "label":    label,
            "png_path": png_path,
            "pdf_path": pdf_path if os.path.exists(pdf_path) else None,
        })

    return results


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎨 AestheteAI")
    st.markdown("*Keyword-driven mood board generation*")
    st.divider()

    page = st.radio(
        "Navigate",
        options=["Dashboard", "Cluster Gallery", "Scatter Plot", "Download Centre"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown(
        "<small style='color:#555'>Run the pipeline via CLI:<br>"
        "<code>python main.py --action pipeline<br>--keyword \"dark minimalism\"</code></small>",
        unsafe_allow_html=True,
    )


# ─── Page 1: Dashboard ────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("Dashboard")
    st.markdown("Overview of your database and pipeline progress.")

    stats    = load_stats()
    all_kws  = load_all_keywords()

    # Top-level stat cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='stat-number'>{stats['total']}</div>"
                    f"<div class='stat-label'>Total Images</div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-number'>{stats['downloaded']}</div>"
                    f"<div class='stat-label'>Downloaded</div>",
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-number'>{stats['colors_extracted']}</div>"
                    f"<div class='stat-label'>Colors Extracted</div>",
                    unsafe_allow_html=True)
    with col4:
        conn = get_db()
        n_embedded = conn.execute(
            "SELECT COUNT(*) FROM images WHERE image_embedding_path IS NOT NULL"
        ).fetchone()[0]
        st.markdown(f"<div class='stat-number'>{n_embedded}</div>"
                    f"<div class='stat-label'>Embedded</div>",
                    unsafe_allow_html=True)

    st.divider()

    # Per-keyword pipeline status table
    st.subheader("Pipeline Status by Keyword")

    if not all_kws:
        st.info("No data yet. Run the scraper to get started:\n\n"
                "```bash\npython main.py --action pipeline --keyword \"dark minimalism\"\n```")
    else:
        for kw in all_kws:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                         AS total,
                    SUM(CASE WHEN local_path IS NOT NULL THEN 1 ELSE 0 END) AS downloaded,
                    SUM(CASE WHEN dominant_colors IS NOT NULL THEN 1 ELSE 0 END) AS colored,
                    SUM(CASE WHEN image_embedding_path IS NOT NULL THEN 1 ELSE 0 END) AS embedded,
                    SUM(CASE WHEN cluster_id IS NOT NULL AND cluster_id != -1 THEN 1 ELSE 0 END) AS clustered
                FROM images WHERE keyword = ?
                """,
                (kw,)
            ).fetchone()

            def badge(val, total):
                if total == 0:
                    return "<span class='pipeline-none'>–</span>"
                pct = val / total * 100
                color_cls = "pipeline-ok" if pct >= 90 else \
                            "pipeline-warn" if pct >= 50 else "pipeline-none"
                return f"<span class='{color_cls}'>{val}/{total}</span>"

            with st.expander(f"**{kw}** — {row['total']} images", expanded=False):
                steps = {
                    "Scraped":   (row["total"],    row["total"]),
                    "Downloaded":(row["downloaded"], row["total"]),
                    "Colors":    (row["colored"],   row["total"]),
                    "Embedded":  (row["embedded"],  row["total"]),
                    "Clustered": (row["clustered"], row["total"]),
                }
                cols = st.columns(len(steps))
                for col, (label, (done, total)) in zip(cols, steps.items()):
                    with col:
                        st.markdown(
                            f"<div class='stat-label'>{label}</div>"
                            f"<div style='font-size:1.2rem'>{badge(done, total)}</div>",
                            unsafe_allow_html=True,
                        )


# ─── Page 2: Cluster Gallery ──────────────────────────────────────────────────

elif page == "Cluster Gallery":
    st.title("Cluster Gallery")

    clustered_kws = load_keywords()

    if not clustered_kws:
        st.info("No clustered keywords yet. Run the full pipeline first:\n\n"
                "```bash\npython main.py --action pipeline --keyword \"dark minimalism\"\n```")
    else:
        keyword = st.selectbox("Select keyword", clustered_kws)
        clusters = load_clusters_for_keyword(keyword)

        if not clusters:
            st.warning(f"No clusters found for '{keyword}'.")
        else:
            st.markdown(f"**{len(clusters)} clusters** found for *{keyword}*")
            st.divider()

            for cluster in clusters:
                cid   = cluster["cluster_id"]
                label = cluster["cluster_label"] or f"Cluster {cid}"
                count = cluster["image_count"]

                with st.expander(f"**{label}** — {count} images", expanded=True):
                    images = load_cluster_images(keyword, cid, limit=12)

                    if not images:
                        st.caption("No image files found on disk for this cluster.")
                        continue

                    # Show color palette from the first image as a sample
                    if images[0]["dominant_colors"]:
                        palette = json.loads(images[0]["dominant_colors"])
                        swatches = "".join(
                            f"<div class='swatch' style='background:{c}'></div>"
                            for c in palette
                        )
                        st.markdown(
                            f"<div class='swatch-row'>{swatches}</div>",
                            unsafe_allow_html=True,
                        )

                    # Image grid — 4 columns
                    img_cols = st.columns(4)
                    for i, img_row in enumerate(images):
                        path = img_row["local_path"]
                        if path and os.path.exists(path):
                            with img_cols[i % 4]:
                                try:
                                    st.image(path, use_container_width=True)
                                    if img_row.get("title"):
                                        st.caption(img_row["title"])
                                except Exception:
                                    st.caption("(unreadable)")


# ─── Page 3: Scatter Plot ─────────────────────────────────────────────────────

elif page == "Scatter Plot":
    st.title("Scatter Plot Explorer")
    st.markdown("UMAP 2D projection of image embeddings colored by cluster.")

    clustered_kws = load_keywords()

    if not clustered_kws:
        st.info("No clustered data yet. Run the pipeline first.")
    else:
        keyword = st.selectbox("Select keyword", clustered_kws)
        coords, labels, label_names = load_umap_data(keyword)

        if coords is None:
            st.warning(
                f"No UMAP 2D projection found for '{keyword}'.\n\n"
                f"Make sure `data/embeddings/umap2_{keyword.replace(' ', '_')}.npy` exists.\n\n"
                f"Re-run: `python main.py --action cluster --keyword \"{keyword}\"`"
            )
        else:
            # Plotly color palette — 20 distinct colors
            COLORS = [
                "#E94560", "#0F3460", "#533483", "#2ECC71", "#F39C12",
                "#1ABC9C", "#E74C3C", "#3498DB", "#9B59B6", "#F1C40F",
                "#E67E22", "#2980B9", "#27AE60", "#8E44AD", "#C0392B",
                "#16A085", "#D35400", "#2C3E50", "#95A5A6", "#BDC3C7",
            ]

            fig  = go.Figure()
            uids = sorted(set(labels.tolist()))

            for i, uid in enumerate(uids):
                mask  = labels == uid
                color = "#444444" if uid == -1 else COLORS[i % len(COLORS)]
                name  = "noise" if uid == -1 else label_names.get(uid, f"cluster {uid}")

                fig.add_trace(go.Scatter(
                    x    = coords[mask, 0].tolist(),
                    y    = coords[mask, 1].tolist(),
                    mode = "markers",
                    name = f"{name} ({mask.sum()})",
                    marker = dict(
                        color   = color,
                        size    = 7 if uid != -1 else 4,
                        opacity = 0.85 if uid != -1 else 0.35,
                        line    = dict(width=0),
                    ),
                    hovertemplate = f"<b>{name}</b><extra></extra>",
                ))

            fig.update_layout(
                template      = "plotly_dark",
                paper_bgcolor = "#0D1117",
                plot_bgcolor  = "#0D1117",
                font          = dict(color="#CCCCCC", size=12),
                legend        = dict(bgcolor="#1A1A2E", bordercolor="#333",
                                     borderwidth=1, font=dict(size=11)),
                margin        = dict(l=20, r=20, t=40, b=20),
                title         = dict(
                    text = f'"{keyword}" — UMAP 2D Cluster View',
                    font = dict(size=14, color="#CCCCCC"),
                ),
                xaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
                height = 600,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cluster legend below the chart
            st.divider()
            clusters = load_clusters_for_keyword(keyword)
            if clusters:
                st.subheader("Clusters")
                cols = st.columns(min(len(clusters), 3))
                for i, c in enumerate(clusters):
                    with cols[i % 3]:
                        color = COLORS[i % len(COLORS)]
                        st.markdown(
                            f"<div style='border-left:4px solid {color};"
                            f"padding-left:10px; margin-bottom:8px'>"
                            f"<strong>{c['cluster_label'] or 'cluster ' + str(c['cluster_id'])}</strong><br>"
                            f"<small style='color:#888'>{c['image_count']} images</small></div>",
                            unsafe_allow_html=True,
                        )


# ─── Page 4: Download Centre ──────────────────────────────────────────────────

elif page == "Download Centre":
    st.title("Download Centre")
    st.markdown("Download mood boards as PNG or PDF for each cluster.")

    clustered_kws = load_keywords()

    if not clustered_kws:
        st.info("No mood boards generated yet. Run the full pipeline first.")
    else:
        for kw in clustered_kws:
            boards = find_board_files(kw)

            with st.expander(f"**{kw}** — {len(boards)} board(s)", expanded=True):
                if not boards:
                    st.caption(
                        f"No board files found for '{kw}'. "
                        f"Run: `python main.py --action boards --keyword \"{kw}\"`"
                    )
                    continue

                for board in boards:
                    col_preview, col_label, col_png, col_pdf = st.columns([2, 3, 1, 1])

                    # Thumbnail preview
                    with col_preview:
                        if os.path.exists(board["png_path"]):
                            try:
                                img = Image.open(board["png_path"])
                                img.thumbnail((200, 120))
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.caption("(preview unavailable)")

                    with col_label:
                        st.markdown(f"**{board['label'].title()}**")

                    # PNG download
                    with col_png:
                        if os.path.exists(board["png_path"]):
                            with open(board["png_path"], "rb") as f:
                                st.download_button(
                                    label    = "PNG",
                                    data     = f.read(),
                                    file_name = os.path.basename(board["png_path"]),
                                    mime     = "image/png",
                                    key      = f"png_{board['png_path']}",
                                )

                    # PDF download
                    with col_pdf:
                        if board["pdf_path"] and os.path.exists(board["pdf_path"]):
                            with open(board["pdf_path"], "rb") as f:
                                st.download_button(
                                    label    = "PDF",
                                    data     = f.read(),
                                    file_name = os.path.basename(board["pdf_path"]),
                                    mime     = "application/pdf",
                                    key      = f"pdf_{board['pdf_path']}",
                                )
                    st.divider()
