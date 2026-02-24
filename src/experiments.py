import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_simulation(p_repost, seed):

    np.random.seed(seed)

    n_users = 500
    G = nx.barabasi_albert_graph(n_users, 3)

    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if np.random.rand() < 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)

    initial_infected = set(np.random.choice(list(DG.nodes()), 5, replace=False))
    infected = set(initial_infected)
    newly_infected = set(initial_infected)

    for t in range(20):

        next_new = set()

        for u in newly_infected:
            for v in DG.successors(u):
                if v not in infected:
                    if np.random.rand() < p_repost:
                        next_new.add(v)

        infected |= next_new
        newly_infected = next_new

        if len(newly_infected) == 0:
            break

    return len(infected)


# =============================
# Parameter Sweep
# =============================

probabilities = [0.02, 0.05, 0.08, 0.12, 0.15]
n_runs = 20

results = []

for p in probabilities:
    for run in range(n_runs):
        final_size = run_simulation(p, seed=run)
        results.append({
            "p_repost": p,
            "run_id": run,
            "final_infected": final_size
        })

df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
df.to_csv("results/parameter_sweep_results.csv", index=False)

# =============================
# Aggregate results
# =============================

summary = df.groupby("p_repost")["final_infected"].agg(["mean", "std"]).reset_index()

plt.errorbar(
    summary["p_repost"],
    summary["mean"],
    yerr=summary["std"],
    fmt='o-',
    capsize=5
)
plt.xlabel("Repost Probability")
plt.ylabel("Average Final Infected Users")
plt.title("Parameter Sweep with Variability")
plt.savefig("results/parameter_sweep_plot.png")

print("Experiment complete.")
print("Results saved in 'results' folder.")
