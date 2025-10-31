# Low-Rank Implementation — Hyena vs MHA, Iterative Sums Signature & Path Development Network

## Overview

This repository implements three related components for modeling and classifying multi-dimensional time series:

1. **Hyena Hierarchy vs Multi-Head Attention (MHA)** — experiments that train a Hyena model to *mimic* the outputs of a Multi-Head Attention model at different orders, to evaluate where Hyena approximates MHA while being more efficient (Hyena is designed to be less costly than full quadratic MHA).
2. **Iterative Sums Signature (ISS)** — a low-rank implementation of iterative sums signatures for time series.
3. **Path Development Network** — tools to compute path development features and combine them with classifiers.

The repository also includes classifiers that use ISS or Path Development features (optionally combined with Hyena outputs) to perform classification on multidimensional time series.

> **Note:** The primary focus of current experiments is on comparing **Hyena** and **MHA** (in terms of performance, representational similarity, and computational efficiency). The ISS and Path Development modules complement these by providing feature extraction and classification capabilities.

---

## Highlights

- Train **Hyena** using **MHA outputs** as targets to evaluate how well Hyena reproduces attention-based representations at different orders.
- Efficient, low-rank **Iterative Sums Signature (ISS)** implementation for time-series feature extraction.
- **Path Development Network** features for sequence modeling and downstream classification.
- **Flexible experiment orchestration** via `Analysis.py`, with built-in logging to **Weights & Biases (wandb)** for metrics, gradients, and loss visualization.

---

## Quick Start

### A. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# or: venv\Scripts\activate   # Windows
pip install -r requirements.txt
