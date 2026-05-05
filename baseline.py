import numpy as np
import networkx as nx
from collections import defaultdict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mutag():
    dataset = TUDataset(root="/tmp/MUTAG", name="MUTAG")
    graphs = [to_networkx(data, to_undirected=True) for data in dataset]
    return graphs


def train_test_split(graphs, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    split = int(len(graphs) * train_ratio)
    return [graphs[i] for i in idx[:split]], [graphs[i] for i in idx[split:]]


# ── Erdős-Rényi baseline ──────────────────────────────────────────────────────

class ErdosRenyiBaseline:
    def __init__(self):
        # Maps n_nodes -> link probability r
        self._node_counts: list[int] = []
        self._density: dict[int, float] = {}

    def fit(self, graphs: list[nx.Graph]):
        self._node_counts = [g.number_of_nodes() for g in graphs]

        # Group graphs by node count and compute mean density per group
        buckets: dict[int, list[float]] = defaultdict(list)
        for g in graphs:
            n = g.number_of_nodes()
            max_edges = n * (n - 1) / 2
            density = g.number_of_edges() / max_edges if max_edges > 0 else 0.0
            buckets[n].append(density)

        self._density = {n: float(np.mean(d)) for n, d in buckets.items()}

        # For unseen node counts we fall back to the global mean density
        all_densities = [d for ds in buckets.values() for d in ds]
        self._global_density = float(np.mean(all_densities))

    def sample(self, rng: np.random.Generator | None = None) -> nx.Graph:
        if rng is None:
            rng = np.random.default_rng()

        n = int(rng.choice(self._node_counts))
        r = self._density.get(n, self._global_density)
        return nx.erdos_renyi_graph(n, r, seed=int(rng.integers(1 << 31)))

    def sample_n(self, n: int, seed: int = 0) -> list[nx.Graph]:
        rng = np.random.default_rng(seed)
        return [self.sample(rng) for _ in range(n)]


# ── Graph comparison (Weisfeiler-Lehman) ─────────────────────────────────────

def wl_hash(g: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(g)


def are_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    return wl_hash(g1) == wl_hash(g2)


# ── Novelty / uniqueness metrics ─────────────────────────────────────────────

def compute_novelty_uniqueness(
    sampled: list[nx.Graph],
    training: list[nx.Graph],
) -> dict[str, float]:
    train_hashes = {wl_hash(g) for g in training}
    sample_hashes = [wl_hash(g) for g in sampled]

    novel_flags = [h not in train_hashes for h in sample_hashes]
    seen: set[str] = set()
    unique_flags = []
    for h in sample_hashes:
        unique_flags.append(h not in seen)
        seen.add(h)

    novel = np.mean(novel_flags)
    unique = np.mean(unique_flags)
    novel_and_unique = np.mean([n and u for n, u in zip(novel_flags, unique_flags)])

    return {"novel": novel, "unique": unique, "novel_and_unique": novel_and_unique}


# ── Graph statistics ──────────────────────────────────────────────────────────

def _safe_eigenvector_centrality(g: nx.Graph) -> list[float]:
    if g.number_of_nodes() == 0:
        return []
    try:
        ec = nx.eigenvector_centrality(g, max_iter=1000)
        return list(ec.values())
    except nx.PowerIterationFailedConvergence:
        return [0.0] * g.number_of_nodes()


def collect_stats(graphs: list[nx.Graph]) -> dict[str, np.ndarray]:
    degrees, clustering, eigenvec = [], [], []
    for g in graphs:
        degrees.extend(dict(g.degree()).values())
        clustering.extend(nx.clustering(g).values())
        eigenvec.extend(_safe_eigenvector_centrality(g))
    return {
        "degree": np.array(degrees, dtype=float),
        "clustering": np.array(clustering, dtype=float),
        "eigenvector": np.array(eigenvec, dtype=float),
    }


def plot_histograms(
    stats_empirical: dict[str, np.ndarray],
    stats_baseline: dict[str, np.ndarray],
    stats_model: dict[str, np.ndarray] | None = None,
    model_label: str = "Deep model",
    save_path: str | None = None,
):
    metrics = ["degree", "clustering", "eigenvector"]
    titles = ["Node degree", "Clustering coefficient", "Eigenvector centrality"]
    cols = 3 if stats_model is not None else 2
    rows = 3

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    for row, (metric, title) in enumerate(zip(metrics, titles)):
        all_vals = np.concatenate([
            stats_empirical[metric],
            stats_baseline[metric],
            *([] if stats_model is None else [stats_model[metric]]),
        ])
        bins = np.histogram_bin_edges(all_vals, bins=20)

        sources = [
            (stats_empirical[metric], "Empirical", "steelblue"),
            (stats_baseline[metric], "Baseline (ER)", "tomato"),
        ]
        if stats_model is not None:
            sources.append((stats_model[metric], model_label, "seagreen"))

        for col, (vals, label, color) in enumerate(sources):
            ax = axes[row, col]
            ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white", density=True)
            ax.set_title(f"{title}\n{label}", fontsize=10)
            ax.set_xlabel(title)
            ax.set_ylabel("Density")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved histogram to {save_path}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MUTAG dataset...")
    graphs = load_mutag()
    train_graphs, _ = train_test_split(graphs)
    print(f"  Total: {len(graphs)}  |  Train: {len(train_graphs)}")

    print("\nFitting Erdős-Rényi baseline...")
    baseline = ErdosRenyiBaseline()
    baseline.fit(train_graphs)

    print("Sampling 1000 graphs from baseline...")
    sampled = baseline.sample_n(1000, seed=42)

    print("\nComputing novelty/uniqueness metrics...")
    metrics = compute_novelty_uniqueness(sampled, train_graphs)
    print(f"  Novel:           {metrics['novel']:.1%}")
    print(f"  Unique:          {metrics['unique']:.1%}")
    print(f"  Novel & unique:  {metrics['novel_and_unique']:.1%}")

    print("\nCollecting graph statistics...")
    stats_emp = collect_stats(train_graphs)
    stats_base = collect_stats(sampled)

    print("Plotting histograms (baseline only)...")
    plot_histograms(stats_emp, stats_base, save_path="histograms_baseline.png")
    print("Done.")
