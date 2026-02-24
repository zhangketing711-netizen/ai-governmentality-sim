import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# 1. Build a scale-free network
# =============================

np.random.seed(42)

n_users = 500
G = nx.barabasi_albert_graph(n_users, 3)

DG = nx.DiGraph()
DG.add_nodes_from(G.nodes())

for u, v in G.edges():
    if np.random.rand() < 0.5:
        DG.add_edge(u, v)
    else:
        DG.add_edge(v, u)

# =============================
# 2. Diffusion parameters
# =============================

p_repost = 0.08
initial_infected = set(np.random.choice(list(DG.nodes()), 5, replace=False))

infected = set(initial_infected)
newly_infected = set(initial_infected)

results = []

# =============================
# 3. Independent Cascade model
# =============================

for t in range(20):

    results.append({
        "time_step": t,
        "total_infected": len(infected)
    })

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

# =============================
# 4. Save results
# =============================

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("results/diffusion_output.csv", index=False)

# =============================
# 5. Plot results
# =============================

plt.plot(df["time_step"], df["total_infected"])
plt.xlabel("Time Step")
plt.ylabel("Total Infected Users")
plt.title("Information Diffusion Simulation")
plt.savefig("results/diffusion_plot.png")

print("Simulation complete.")
print("Results saved in the 'results' folder.")
