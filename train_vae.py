"""
train_vae.py — Graph VAE for MUTAG + baseline comparison
"""
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GraphNorm, global_add_pool

from baseline import ErdosRenyiBaseline, compute_novelty_uniqueness, collect_stats, plot_histograms

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 128
LATENT_DIM     = 32
EPOCHS         = 300
DROPOUT        = 0.15
LR             = 5e-4
BATCH_SIZE     = 32
EDGE_THRESHOLD = 0.5
SEED           = 42
N_SAMPLES      = 1000
device         = 'cpu'

# ── Data ──────────────────────────────────────────────────────────────────────
root = os.path.expanduser('~/mutag_data')
os.makedirs(root, exist_ok=True)

pyg_dataset      = TUDataset(root=root, name='MUTAG')
node_feature_dim = pyg_dataset.num_node_features
max_num_nodes    = max(data.num_nodes for data in pyg_dataset)
print(f"Dataset: {len(pyg_dataset)} graphs | max_nodes={max_num_nodes} | node_features={node_feature_dim}")

gen = torch.Generator().manual_seed(SEED)
train_pyg, val_pyg, test_pyg = random_split(pyg_dataset, (100, 44, 44), generator=gen)

train_loader = DataLoader(train_pyg, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_pyg,   batch_size=44)

# NetworkX versions of training graphs (structure only, no node attrs)
nx_train = []
for i in train_pyg.indices:
    g = to_networkx(pyg_dataset[i], to_undirected=True)
    nx_train.append(nx.Graph(g.edges(), nodes=g.nodes()))

# Pos-weight for edge BCE: compensate for sparse graphs
total_possible, total_actual = 0, 0
for data in train_pyg:
    n = data.num_nodes
    total_possible += n * (n - 1)
    total_actual   += data.num_edges
pos_weight = torch.tensor([(total_possible - total_actual) / max(total_actual, 1)])
print(f"Edge pos_weight: {pos_weight.item():.2f}")


# ── Model ─────────────────────────────────────────────────────────────────────
class GraphVAE(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, max_num_nodes, dropout=0.15):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim       = hidden_dim
        self.latent_dim       = latent_dim
        self.max_num_nodes    = max_num_nodes
        self.dropout          = dropout

        self.input_proj = torch.nn.Linear(node_feature_dim, hidden_dim)

        self.gcn1  = GCNConv(hidden_dim, hidden_dim);      self.norm1 = GraphNorm(hidden_dim)
        self.gcn2  = GCNConv(hidden_dim, hidden_dim);      self.norm2 = GraphNorm(hidden_dim)
        self.gat1  = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.norm3 = GraphNorm(hidden_dim)
        self.gat2  = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.norm4 = GraphNorm(hidden_dim)
        self.gin1  = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        ));                                                 self.norm5 = GraphNorm(hidden_dim)
        self.gin2  = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        ));                                                 self.norm6 = GraphNorm(hidden_dim)

        self.mu_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.logvar_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )

        embed_dim = hidden_dim  # per-node embedding size in decoder
        self.embed_dim = embed_dim

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim * 2), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 4), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim * 4), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 4), torch.nn.ReLU()
        )
        # Expand global h into one embedding vector per node position
        self.node_embed_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 4), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 4, max_num_nodes * embed_dim)
        )
        # Atom type from each node embedding
        self.node_type_head = torch.nn.Linear(embed_dim, node_feature_dim)

        self.node_count_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, max_num_nodes)
        )

    def _conv_block(self, h, edge_index, batch, conv, norm):
        h_out = torch.relu(norm(conv(h, edge_index), batch))
        h_out = torch.nn.functional.dropout(h_out, p=self.dropout, training=self.training)
        return h_out + h

    def encode(self, x, edge_index, batch):
        h = self.input_proj(x)
        h = self._conv_block(h, edge_index, batch, self.gcn1, self.norm1)
        h = self._conv_block(h, edge_index, batch, self.gcn2, self.norm2)
        h = self._conv_block(h, edge_index, batch, self.gat1, self.norm3)
        h = self._conv_block(h, edge_index, batch, self.gat2, self.norm4)
        h = self._conv_block(h, edge_index, batch, self.gin1, self.norm5)
        h = self._conv_block(h, edge_index, batch, self.gin2, self.norm6)
        graph_h = global_add_pool(h, batch)
        mu     = self.mu_net(graph_h)
        logvar = torch.clamp(self.logvar_net(graph_h), -10, 10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z):
        h = self.decoder(z)
        N = self.max_num_nodes

        # One embedding vector per node position: [B, N, embed_dim]
        node_embeds = self.node_embed_decoder(h).view(-1, N, self.embed_dim)

        # Atom type per node: [B, N, 7]
        node_logits = self.node_type_head(node_embeds)

        # Edge logits as scaled dot products — edge_ij depends on both endpoints
        scale       = self.embed_dim ** 0.5
        edge_logits = torch.bmm(node_embeds, node_embeds.transpose(1, 2)) / scale
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        eye         = torch.eye(N, device=edge_logits.device).unsqueeze(0).bool()
        edge_logits = edge_logits.masked_fill(eye, -10.0)

        node_count_logits = self.node_count_decoder(h)
        return node_logits, edge_logits, node_count_logits

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_logits, node_count_logits = self.decode(z)
        return node_logits, edge_logits, node_count_logits, mu, logvar


# ── Loss ──────────────────────────────────────────────────────────────────────
def loss_fn(node_logits, edge_logits, node_count_logits,
            x, edge_index, batch, mu, logvar, beta, pos_weight):
    X, mask = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)
    A       = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes)

    # Node type loss (only over real nodes)
    node_loss = torch.nn.functional.cross_entropy(
        node_logits.reshape(-1, node_feature_dim),
        X.argmax(dim=2).reshape(-1),
        reduction='none'
    ).view(X.shape[0], max_num_nodes)
    node_loss = (node_loss * mask.float()).sum() / mask.float().sum()

    # Edge loss — diagonal excluded from mask (no self-loops)
    edge_mask = mask.float().unsqueeze(1) * mask.float().unsqueeze(2)
    eye       = torch.eye(max_num_nodes, device=edge_mask.device).unsqueeze(0)
    edge_mask = edge_mask * (1.0 - eye)
    edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        edge_logits, A,
        pos_weight=pos_weight.to(edge_logits.device),
        reduction='none'
    )
    edge_loss = (edge_loss * edge_mask).sum() / edge_mask.sum()

    # Node count loss (0-indexed: target = n_nodes - 1)
    graph_sizes     = mask.sum(dim=1).long() - 1
    node_count_loss = torch.nn.functional.cross_entropy(node_count_logits, graph_sizes)

    # KL divergence
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total = node_loss + edge_loss + 0.25 * node_count_loss + beta * kl_loss
    return total, node_loss, edge_loss, node_count_loss, kl_loss


# ── Training ──────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
model     = GraphVAE(node_feature_dim, HIDDEN_DIM, LATENT_DIM, max_num_nodes, DROPOUT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nTraining GraphVAE for {EPOCHS} epochs...")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    beta = min(0.05, 0.05 * (epoch + 1) / 150)

    for data in train_loader:
        node_logits, edge_logits, node_count_logits, mu, logvar = model(
            data.x, data.edge_index, data.batch
        )
        loss, *_ = loss_fn(
            node_logits, edge_logits, node_count_logits,
            data.x, data.edge_index, data.batch,
            mu, logvar, beta, pos_weight
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item() * data.num_graphs / len(train_pyg)

    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            node_logits, edge_logits, node_count_logits, mu, logvar = model(
                data.x, data.edge_index, data.batch
            )
            loss, *_ = loss_fn(
                node_logits, edge_logits, node_count_logits,
                data.x, data.edge_index, data.batch,
                mu, logvar, beta, pos_weight
            )
            val_loss += loss.item() * data.num_graphs / len(val_pyg)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | lr={scheduler.get_last_lr()[0]:.1e} "
              f"| beta={beta:.4f} | train={train_loss:.3f} | val={val_loss:.3f}")

print("Training complete.\n")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
plt.legend(); plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print("Saved loss_curve.png")


# ── Generate NetworkX graphs from VAE ─────────────────────────────────────────
@torch.no_grad()
def generate_nx_graphs(model, num_graphs, edge_threshold=EDGE_THRESHOLD):
    """Guarantee exactly num_graphs NetworkX graphs (no silent drops)."""
    model.eval()
    graphs = []
    while len(graphs) < num_graphs:
        needed = max((num_graphs - len(graphs)) * 2, 64)
        z = torch.randn(needed, model.latent_dim)
        node_logits, edge_logits, node_count_logits = model.decode(z)
        edge_probs   = torch.sigmoid(edge_logits).numpy()
        node_counts  = node_count_logits.argmax(dim=1).numpy() + 1

        for i in range(needed):
            n   = int(node_counts[i])
            adj = (edge_probs[i, :n, :n] > edge_threshold).astype(float)
            np.fill_diagonal(adj, 0)
            g = nx.from_numpy_array(adj)
            graphs.append(g)
            if len(graphs) >= num_graphs:
                break

    return graphs[:num_graphs]


# ── Sample 1000 from each model ───────────────────────────────────────────────
print(f"Fitting ER baseline on {len(nx_train)} training graphs...")
baseline = ErdosRenyiBaseline()
baseline.fit(nx_train)

print(f"Sampling {N_SAMPLES} graphs from ER baseline...")
er_samples  = baseline.sample_n(N_SAMPLES, seed=0)

torch.seed()  # break determinism for generation
print(f"Sampling {N_SAMPLES} graphs from GraphVAE...")
vae_samples = generate_nx_graphs(model, N_SAMPLES)


# ── Novelty / uniqueness table ────────────────────────────────────────────────
er_m  = compute_novelty_uniqueness(er_samples,  nx_train)
vae_m = compute_novelty_uniqueness(vae_samples, nx_train)

print("\n" + "─" * 55)
print(f"{'':25s} {'Novel':>8} {'Unique':>8} {'Nov+Uniq':>10}")
print("─" * 55)
print(f"{'Baseline (ER)':25s} {er_m['novel']:>8.1%} {er_m['unique']:>8.1%} {er_m['novel_and_unique']:>10.1%}")
print(f"{'GraphVAE':25s} {vae_m['novel']:>8.1%} {vae_m['unique']:>8.1%} {vae_m['novel_and_unique']:>10.1%}")
print("─" * 55)


# ── Histograms (3×3 grid) ─────────────────────────────────────────────────────
print("\nCollecting graph statistics and plotting histograms...")
stats_emp = collect_stats(nx_train)
stats_er  = collect_stats(er_samples)
stats_vae = collect_stats(vae_samples)

plot_histograms(stats_emp, stats_er, stats_vae, model_label="GraphVAE", save_path="histograms_all.png")
print("Saved histograms_all.png")


# ── Sample graph visualisation ────────────────────────────────────────────────
print("Plotting 9 sample VAE graphs...")
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, (g, ax) in enumerate(zip(vae_samples[:9], axes.flat)):
    pos = nx.spring_layout(g, seed=i)
    nx.draw(g, pos, ax=ax, node_size=120, node_color='seagreen',
            edge_color='gray', with_labels=False, width=1.5)
    ax.set_title(f'nodes={g.number_of_nodes()}  edges={g.number_of_edges()}', fontsize=9)
plt.suptitle('GraphVAE — 9 sampled graphs', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('sample_vae_graphs.png', dpi=150, bbox_inches='tight')
print("Saved sample_vae_graphs.png")
print("\nAll done!")
