import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# =========================
# 1) Utilities
# =========================

CATEGORIES = [
    "normative_appeal",
    "moralization",
    "problematization",
    "subject_positions",
    "conduct_of_conduct",
    "regimes_of_truth",
]

def shannon_entropy(counts: dict) -> float:
    """Compute Shannon entropy (base e) from a dict of counts."""
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

def pick_initial_category(mode: str, rng: np.random.Generator) -> str:
    """
    Human mode: more concentrated distribution (less diverse).
    AI mode: flatter distribution (more diverse).
    """
    if mode == "human":
        # concentrated: first 2 categories dominate
        probs = np.array([0.35, 0.30, 0.12, 0.10, 0.08, 0.05])
    else:
        # more diverse: flatter
        probs = np.array([0.18, 0.17, 0.17, 0.16, 0.16, 0.16])

    probs = probs / probs.sum()
    return rng.choice(CATEGORIES, p=probs)

def maybe_mutate_category(category: str, mode: str, rng: np.random.Generator) -> str:
    """
    Template-based variation:
    - Human: low variation
    - AI: higher variation (rewriting/genre switching)
    """
    if mode == "human":
        p_change = 0.06
    else:
        p_change = 0.22

    if rng.random() < p_change:
        # switch to a different category
        others = [c for c in CATEGORIES if c != category]
        return rng.choice(others)
    return category


# =========================
# 2) Simulation core
# =========================

def run_mode(mode: str, seed: int = 42) -> pd.DataFrame:
    """
    Simulate diffusion with:
    - production_rate: how many new seed posts each step
    - p_repost: diffusion probability
    - category mutation: AI has more variation

    Track:
    - total infected users
    - entropy of category distribution among infected users
    """
    rng = np.random.default_rng(seed)

    # --- platform-like network ---
    n_users = 800
    DG = build_directed_scale_free_graph(n_users, seed=seed)

    # --- mode parameters ---
    if mode == "human":
        production_rate = 2      # new seed posts per step
        p_repost = 0.06
    else:
        production_rate = 6      # AI produces more content per step
        p_repost = 0.08

    n_steps = 25
    initial_seed_posts = 6

    # Each user holds ONE category label = the discourse frame they "adopted"
    user_label = {}

    # Initialize: infect some users as seed posters
    initial_users = set(rng.choice(list(DG.nodes()), size=initial_seed_posts, replace=False))
    infected = set(initial_users)
    newly_infected = set(initial_users)

    for u in initial_users:
        user_label[u] = pick_initial_category(mode, rng)

    rows = []

    for t in range(n_steps):
        # ----- record metrics -----
        counts = {c: 0 for c in CATEGORIES}
        for u in infected:
            lab = user_label.get(u, None)
            if lab is not None:
                counts[lab] += 1

        rows.append({
            "mode": mode,
            "t": t,
            "total_infected": len(infected),
            "entropy": shannon_entropy(counts),
            **{f"count_{c}": counts[c] for c in CATEGORIES}
        })

        # ----- production: introduce new seed posts -----
        # This operationalizes "AI enables rapid scaling" (more posts per step)
        new_seed_users = set(rng.choice(list(DG.nodes()), size=production_rate, replace=False))
        for u in new_seed_users:
            if u not in infected:
                infected.add(u)
                newly_infected.add(u)
                user_label[u] = pick_initial_category(mode, rng)

        # ----- diffusion (Independent Cascade style) -----
        next_new = set()

        for u in newly_infected:
            if u not in DG:
                continue

            u_cat = user_label.get(u, pick_initial_category(mode, rng))
            # AI mode: category may mutate more (templated rewriting/genre switching)
            spread_cat = maybe_mutate_category(u_cat, mode, rng)

            for v in DG.successors(u):
                if v in infected:
                    continue

                if rng.random() < p_repost:
                    next_new.add(v)
                    user_label[v] = spread_cat

        newly_infected = next_new
        infected |= next_new

        if len(newly_infected) == 0:
            # still continue? we break to keep curves readable
            break

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    df_human = run_mode("human", seed=42)
    df_ai = run_mode("ai", seed=42)

    df = pd.concat([df_human, df_ai], ignore_index=True)
    out_csv = "results/ai_vs_human_timeseries.csv"
    df.to_csv(out_csv, index=False)

    # ---- plot: total infected over time ----
    plt.figure()
    for mode in ["human", "ai"]:
        sub = df[df["mode"] == mode]
        plt.plot(sub["t"], sub["total_infected"], label=mode)
    plt.xlabel("Time Step")
    plt.ylabel("Total Infected Users")
    plt.title("AI vs Human: Diffusion Scale")
    plt.legend()
    out1 = "results/ai_vs_human_infected.png"
    plt.savefig(out1)

    # ---- plot: entropy over time ----
    plt.figure()
    for mode in ["human", "ai"]:
        sub = df[df["mode"] == mode]
        plt.plot(sub["t"], sub["entropy"], label=mode)
    plt.xlabel("Time Step")
    plt.ylabel("Shannon Entropy (Discursive Diversity)")
    plt.title("AI vs Human: Discursive Diversification (Entropy)")
    plt.legend()
    out2 = "results/ai_vs_human_entropy.png"
    plt.savefig(out2)

    print("Done.")
    print("Saved:", out_csv)
    print("Saved:", out1)
    print("Saved:", out2)


if __name__ == "__main__":
    main()
