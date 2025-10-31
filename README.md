# Low-Rank Implementation — Iterative Sums Signature (ISS) & Path Development Network  
*(with Hyena ⇄ MHA experiments used for baseline parameterization)*

## Overview

This repository provides a **low-rank implementation of Iterative Sums Signatures (ISS)** and a **Path Development Network** for multi-dimensional time series, along with classifiers built on top of these representations.

The **primary focus** of this work is on the **ISS and Path Development models** — their mathematical formulation, low-rank implementations, and classification performance.  
The **Hyena vs Multi-Head Attention (MHA)** experiments included here are **supporting components**: they are used to **parameterize and tune the Hyena model** so that it can serve as an efficient and fair baseline for comparison with our low-rank ISS and Path Development models.

A key insight is that the **low-rank structure** of these models allows the underlying formulae to be **separated into individual components**, enabling computation via **convolution operations** on each term. Using **Fast Fourier Transform (FFT)-based convolutions**, this reduces computational complexity from **O(n²)** to **O(n log n)** — allowing these methods to efficiently scale to longer sequences.

---

## Highlights

- Primary modules:
  - **Iterative Sums Signature (ISS)** — low-rank formulation for structured time-series features.
  - **Path Development Network** — complementary framework for capturing sequential dependencies.
- **Low-rank factorization** allows separable convolutional computations using FFTs → **O(n log n)** runtime.
- **Hyena vs MHA** experiments used to parameterize Hyena as a tuned baseline.
- **Classifiers** for ISS, Path Development, and hybrid (ISS + Hyena) architectures.
- **Experiment management & logging** through Weights & Biases (wandb).

---

## Quick Start

### For ISS and Path Development Classifiers

Go to the `classifiers/` directory and run:

1. `ISS_Classifier.py` — for ISS  
2. `ISSH_Classifier.py` — for ISS + Hyena  
3. and so on
4. 
---

### For Hyena vs MHA

1. Run `Imports.py`  
2. Run `MultiHeadAttention.py`  
3. Run `HyenaLightning.py`  
4. Run `Analysis.py` — modify hyperparameters such as **order**, **learning rate**, **optimizer**, etc., as needed.  
   Log in to your **wandb** account to record gradients, losses, and errors.

