# GNN Project — MUTAG Graph Generation

Mini-project 3, Advanced Machine Learning

## Files

| File | What it is |
|---|---|
| `baseline.py` | Erdős-Rényi baseline model + all evaluation metrics |
| `train_vae.py` | Graph VAE with node-pair decoder |

## Setup

```bash
pip install torch torch_geometric networkx matplotlib numpy scipy
```

## How to run

### 1. Baseline only
Trains the Erdős-Rényi baseline, samples 1000 graphs, prints novelty/uniqueness,
and saves `histograms_baseline.png`.

```bash
python baseline.py
```

### 2. Full pipeline (VAE + baseline comparison)
Trains the Graph VAE for 300 epochs, then runs both models and produces all outputs.
Takes ~5–10 minutes on CPU.

```bash
python train_vae.py
```

## Outputs

| File | Description |
|---|---|
| `loss_curve.png` | Train/val loss over 200 epochs |
| `histograms_all.png` | 3×3 grid: node degree, clustering coefficient, eigenvector centrality for empirical / ER / VAE |
| `sample_vae_graphs.png` | 9 randomly generated VAE graphs |
| `sample_real_graphs.png` | 9 real MUTAG training graphs (for comparison) |
| `sample_graphs.png` | 9 ER baseline graphs |

## Model summary

**Encoder** — 6 GNN layers (GCN→GCN→GAT→GAT→GIN→GIN) with residual connections,
followed by global sum pooling → outputs `mu` and `logvar` (32-dim latent space).

**Decoder** — MLP expands `z` into 28 node embeddings (128-dim each).
Edges are predicted as scaled dot products between node embedding pairs (VGAE-style).
Node type and graph size predicted from the same embeddings.

**Loss** — node type (cross-entropy) + edge (BCE with pos_weight=7.19) +
node count (cross-entropy) + KL divergence (beta-annealed 0→0.05 over 150 epochs).

## Results

|  | Novel | Unique | Novel+Unique |
|---|---|---|---|
| Baseline (ER) | 100% | 99.8% | 99.8% |
| GraphVAE | 100% | 48.9% | 48.9% |
