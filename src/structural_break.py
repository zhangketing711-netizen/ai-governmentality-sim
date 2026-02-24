import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CATEGORIES = [
    "normative_appeal",
    "moralization",
    "problematization",
    "subject_positions",
    "conduct_of_conduct",
    "regimes_of_truth",
]

def shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent

def build_directed_scale_free_graph(n_users: int, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n_users, 3, seed=seed)
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rng.random() < 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)
    return DG

def pick_category(mode: str, rng: np.random.Generator) -> str:
    # human: more concentrated; ai: more diverse
    if mode == "human":
        probs = np.array([0.35, 0.30, 0.12, 0.10, 0.08, 0.05])
    else:
        probs = np.array([0.18, 0.17, 0.17, 0.16, 0.16, 0.16])
    probs = probs / probs.sum()
    return rng.choice(CATEGORIES, p=probs)

def maybe_mutate(category: str, mode: str, rng: np.random.Generator) -> str:
    p_change = 0.06 if mode == "human" else 0.22
    if rng.random() < p_change:
        others = [c for c in CATEGORIES if c != category]
        return rng.choice(others)
    return category

def run_structural_break(seed: int = 42, n_users: int = 900, n_steps: int = 30, break_t: int = 12) -> pd.DataFrame:
    """
    Structural break design:
    - t < break_t: human regime
    - t >= break_t: AI-assisted regime
    """
    rng = np.random.default_rng(seed)
    DG = build_directed_scale_free_graph(n_users, seed=seed)

    human = {"production_rate": 2, "p_repost": 0.06}
    ai = {"production_rate": 6, "p_repost": 0.08}

    user_label = {}
    infected = set()
    newly = set()

    # initial seeds (start in human regime)
    init = set(rng.choice(list(DG.nodes()), size=8, replace=False))
    infected |= init
    newly |= init
    for u in init:
        user_label[u] = pick_category("human", rng)

    rows = []

    for t in range(n_steps):
        mode = "human" if t < break_t else "ai"
        params = human if mode == "human" else ai

        counts = {c: 0 for c in CATEGORIES}
        for u in infected:
            lab = user_label.get(u)
            if lab is not None:
                counts[lab] += 1

        rows.append({
            "t": t,
            "mode": mode,
            "total_infected": len(infected),
            "entropy": shannon_entropy(counts),
            "break_t": break_t,
            **{f"count_{c}": counts[c] for c in CATEGORIES},
        })

        # production: add new seed users each step
        new_seeds = set(rng.choice(list(DG.nodes()), size=params["production_rate"], replace=False))
        for u in new_seeds:
            if u not in infected:
                infected.add(u)
                newly.add(u)
                user_label[u] = pick_category(mode, rng)

        # diffusion: Independent Cascade
        next_new = set()
        for u in newly:
            if u not in DG:
                continue

            base_cat = user_label.get(u, pick_category(mode, rng))
            spread_cat = maybe_mutate(base_cat, mode, rng)

            for v in DG.successors(u):
                if v in infected:
                    continue
                if rng.random() < params["p_repost"]:
                    next_new.add(v)
                    user_label[v] = spread_cat

        newly = next_new
        infected |= next_new

        if len(newly) == 0:
            break

    return pd.DataFrame(rows)

def main():
    os.makedirs("results", exist_ok=True)

    df = run_structural_break(seed=42, n_users=900, n_steps=30, break_t=12)
    df.to_csv("results/structural_break_timeseries.csv", index=False)

    break_t = int(df["break_t"].iloc[0])

    # plot: diffusion scale
    plt.figure()
    plt.plot(df["t"], df["total_infected"])
    plt.axvline(break_t, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Total Infected Users")
    plt.title("Structural Break: Diffusion Scale (Human -> AI)")
    plt.savefig("results/structural_break_infected.png")

    # plot: entropy
    plt.figure()
    plt.plot(df["t"], df["entropy"])
    plt.axvline(break_t, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Shannon Entropy")
    plt.title("Structural Break: Discursive Diversification (Entropy)")
    plt.savefig("results/structural_break_entropy.png")

    print("Saved: results/structural_break_timeseries.csv")
    print("Saved: results/structural_break_infected.png")
    print("Saved: results/structural_break_entropy.png")

if __name__ == "__main__":
    main()
