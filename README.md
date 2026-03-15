---
title:Moodify
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
tags:
  - computer-vision
  - clustering
  - mood-board
  - clip
  - umap
  - hdbscan
  - streamlit
pinned: false
---

<div align="center">

# Moodify

### Keyword-Driven Mood Board Generation via Multi-Source Visual Scraping and Semantic Image Clustering

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![CLIP](https://img.shields.io/badge/Model-CLIP_ViT--B/32-orange?style=flat-square)](https://huggingface.co/openai/clip-vit-base-patch32)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue?style=flat-square&logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Overview](#overview) · [Demo](#demo) · [How It Works](#how-it-works) · [Getting Started](#getting-started) · [Usage](#usage) · [Project Structure](#project-structure) · [Documentation](#documentation)

</div>

---

## Overview

moodify takes a search keyword and automatically generates aesthetic mood boards by scraping visual content from Behance and Pinterest, clustering images by visual and semantic similarity, and composing them into designed, exportable boards.

The system combines web scraping, computer vision, and unsupervised machine learning into a single end-to-end pipeline. Given a keyword like `"dark minimalism"`, it produces clusters labelled `"cinematic deep navy"` or `"stark geometric"` — each one a distinct aesthetic identity found in the data.

This is a learning-oriented project. Every module is heavily commented, every design decision is documented, and the codebase is structured to be readable by someone new to ML engineering.

---

## Demo

| Dashboard | Cluster Gallery |
|---|---|
| Pipeline status per keyword | Mood boards grouped by aesthetic cluster |

| Scatter Plot | Download Centre |
|---|---|
| Interactive 2D UMAP cluster view | PNG and PDF download per board |

---

## How It Works

The pipeline runs in seven phases, each building on the last:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  keyword  ──►  Playwright browser  ──►  Behance + Pinterest URLs    │
│                                                  │                   │
│                                            SQLite DB                │
│                                                  │                   │
│                                         HTTP download               │
│                                         data/images/                │
│                                                  │                   │
│                               ┌──────────────────┤                  │
│                               ▼                  ▼                  │
│                    K-Means color palette    CLIP image encoder      │
│                    (LAB color space)        (512-dim vector)        │
│                               └──────────────────┘                  │
│                                                  │                   │
│                              Unified feature matrix (560-dim)       │
│                                                  │                   │
│                          UMAP dimensionality reduction              │
│                          560d → 10d (cluster) + 2d (visualize)     │
│                                                  │                   │
│                          HDBSCAN clustering                         │
│                          (auto-discovers cluster count)             │
│                                                  │                   │
│                          Auto-labelling                             │
│                          color name + CLIP mood word                │
│                                                  │                   │
│                          Mood board composition (Pillow)            │
│                          Hero + grid layout · 2400×1600px           │
│                          PNG + PDF export (ReportLab)               │
│                                                  │                   │
│                          Streamlit web app                          │
└─────────────────────────────────────────────────────────────────────┘
```

### What makes this interesting

**CLIP embeddings capture aesthetics, not just content.** Two dark editorial photos with different subjects will have similar CLIP embeddings because they share visual style. K-Means on RGB pixel values could not do this.

**HDBSCAN finds the right number of clusters automatically.** For a keyword like "dark minimalism", the number of distinct aesthetic sub-groups is unknown in advance. HDBSCAN discovers it from the data, and marks ambiguous images as noise rather than forcing them into a cluster.

**The feature vector is a deliberate design.** 512 dimensions of CLIP (semantic/aesthetic signal) plus 48 dimensions of LAB color histogram (color distribution signal), weighted 2:1. This produces clusters that are both aesthetically and chromatically coherent.

---

## Getting Started

### Prerequisites

- Python 3.11 or newer
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/DilharaJay/aesthete-ai.git
cd aesthete-ai
```

**2. Create a virtual environment**

```bash
uv venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
uv pip install -r requirements.txt
```

For a smaller PyTorch install (CPU only, recommended for laptops):

```bash
uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

**4. Install Playwright's browser**

```bash
playwright install chromium
```

---

## Usage

### Run the full pipeline

```bash
python main.py --action pipeline --keyword "dark minimalism" --pages 3
```

This runs all phases in sequence: scrape → download → color extraction → CLIP embeddings → feature matrix → clustering → mood board generation.

### Run individual steps

```bash
# Scrape image URLs to the database
python main.py --action scrape --keyword "dark minimalism" --pages 3 --source all

# Download all pending images
python main.py --action download

# Extract color palettes and histograms
python main.py --action extract_colors

# Generate CLIP embeddings
python main.py --action embed --keyword "dark minimalism"

# Build unified feature matrix
python main.py --action build_features --keyword "dark minimalism"

# Cluster images and generate labels
python main.py --action cluster --keyword "dark minimalism"

# Generate mood board PNG and PDF files
python main.py --action boards --keyword "dark minimalism"

# Check pipeline progress
python main.py --action stats
```

### If you already have JSON files from Phase 2 scraping

```bash
python migrate.py       # import JSON records into SQLite
python reconcile.py     # link already-downloaded images to DB records
```

### Launch the UI

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Run with Docker

```bash
docker-compose up --build
# Browse to http://localhost:8501
```

---

## Project Structure

```
aesthete-ai/
│
├── app.py                    Streamlit web application
├── main.py                   CLI entry point and pipeline dispatcher
├── config.py                 All settings in one place
├── migrate.py                One-time JSON → SQLite migration
├── reconcile.py              Links existing image files to DB records
│
├── scraper/
│   ├── behance_scraper_v2.py Playwright scraper for Behance
│   └── pinterest_scraper.py  Playwright scraper for Pinterest
│
├── utils/
│   ├── database.py           SQLite schema, CRUD, and migrations
│   ├── downloader.py         HTTP image downloader with retry/logging
│   ├── color_extractor.py    K-Means palette + LAB histogram
│   ├── embedder.py           CLIP image/text encoding + feature matrix
│   ├── clusterer.py          UMAP + HDBSCAN + auto-labelling + plots
│   └── board_maker.py        Mood board composition + PNG/PDF export
│
├── tests/                    Full pytest test suite (7 test files)
│
├── data/
│   ├── Moodify.db         SQLite database
│   ├── raw/                  Phase 2 JSON scraper output
│   ├── images/               Downloaded image files
│   ├── embeddings/           CLIP .npy files + feature matrices
│   └── clusters/             Per-cluster image preview grids
│
├── output/
│   ├── boards/               Mood board PNG and PDF exports
│   └── plots/                UMAP scatter plots
│
├── docs/
│   ├── ARCHITECTURE.md       System design and module responsibilities
│   └── TECHNOLOGIES.md       Every library explained
│
├── COOKBOOK.md               Complete beginner build guide
├── Dockerfile                Container definition
├── docker-compose.yml        Local development setup
└── .streamlit/config.toml    Dark theme and server config
```

---

## Tech Stack

| Area | Technology | Why |
|---|---|---|
| Environment | Python 3.11, uv | Fast, modern package management |
| Scraping | Playwright + Chromium | Bypasses JS-based bot detection |
| HTML parsing | BeautifulSoup4 + lxml | Fallback img tag extraction |
| Storage | SQLite (built-in) | Zero-config file database |
| Image processing | Pillow | Image loading, manipulation, board composition |
| Color analysis | scikit-learn K-Means, scikit-image LAB | Perceptually meaningful color clustering |
| Numerical computing | NumPy | Arrays, embeddings, matrix ops |
| Neural embeddings | CLIP (openai/clip-vit-base-patch32) via HuggingFace transformers + PyTorch | Semantic image representation |
| Dimensionality reduction | UMAP | Non-linear, preserves neighborhood structure |
| Clustering | HDBSCAN | Auto cluster count, handles noise |
| Visualization | Matplotlib | Scatter plots and preview grids |
| PDF export | ReportLab | Professional-quality PDF generation |
| Web UI | Streamlit | Pure Python web app |
| Interactive charts | Plotly | Zoomable, hoverable scatter plot |
| Deployment | Docker + Hugging Face Spaces | Free, portable, public demo |
| Testing | pytest + unittest.mock | Fast, isolated, no network needed |

---

## Documentation

| Document | Description |
|---|---|
| [`COOKBOOK.md`](COOKBOOK.md) | Complete beginner guide — build the project from scratch with every concept explained |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, data flow, module responsibilities, and design decisions |
| [`docs/TECHNOLOGIES.md`](docs/TECHNOLOGIES.md) | Every library explained: what it is, why it was chosen, and how it is used |

---

## Deployment

### Hugging Face Spaces

This repository is configured for HuggingFace Spaces Docker deployment. The YAML header at the top of this README is read by Hugging Face to configure the Space.

```bash
git remote add space https://huggingface.co/spaces/DilharaJay/aestheteai
git push space main
```

> **Note:** HuggingFace Spaces free tier does not persist data between restarts. Pre-computed results need to be included in the image or uploaded to Spaces persistent storage separately.

---

## Running Tests

```bash
# Run all tests
pytest -v

# Run a specific test file
pytest tests/test_database.py -v

# Run tests matching a keyword
pytest -k "test_insert" -v
```

All tests use in-memory SQLite databases and mocked HTTP/ML calls — no network access or model downloads required to run the test suite.

---

## Academic Context

This project was developed as a university assignment for the Department of Data Science at Sabaragamuwa University of Sri Lanka.

**Research question:** Can unsupervised clustering of CLIP embeddings produce aesthetically coherent groupings of web-scraped images that are interpretable without human labelling?

**Key findings documented in the project:**
- Phase 1 static scrapers produced 403/401 errors — documented as a finding about JavaScript-based bot detection
- CLIP embeddings in a shared image-text space enable mood-word labelling of clusters without any supervised training
- Combining CLIP (semantic) and LAB histogram (color) features produces more aesthetically coherent clusters than either signal alone

---

## Author

**DilharaJay**
Data Science Undergraduate, Sabaragamuwa University of Sri Lanka
GitHub: [@DilharaJay](https://github.com/DilharaJay)

---