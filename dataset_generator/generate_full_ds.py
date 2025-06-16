import torch
import networkx as nx
import numpy as np
from pathlib import Path
from scipy.linalg import pinv
from tqdm import tqdm
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append("/home/jovyan/Diplomski/my_graphs_dataset")
from my_graphs_dataset import GraphDataset

# -------------------------------
# DEFINICIJE METRIKA
# -------------------------------

def algebraic_connectivity(G):
    # L = nx.laplacian_matrix(G).toarray()
    # lambdas = sorted(np.linalg.eigvalsh(L))
    # return lambdas[1]
    return nx.algebraic_connectivity(G)

def effective_graph_resistance(G):
    return nx.effective_graph_resistance(G)

def node_connectivity(G):
    return nx.node_connectivity(G)

def edge_connectivity(G):
    return nx.edge_connectivity(G)

def network_criticality(G):
    L = nx.laplacian_matrix(G).toarray()
    L_plus = pinv(L)
    trace_L_plus = np.trace(L_plus)
    N = L.shape[0]
    tau_hat = 2 * trace_L_plus / (N - 1)        
    return tau_hat

def calculate_metric(metric, G):
    match metric:
        case "algebraic_connectivity":
            return algebraic_connectivity(G)
        case "effective_graph_resistance":
            return effective_graph_resistance(G)
        case "node_connectivity":
            return node_connectivity(G)
        case "edge_connectivity":
            return edge_connectivity(G)
        case "network_criticality":
            return network_criticality(G)
        case default:
            return algebraic_connectivity(G)

def process_graph(i, g6, metric):
    G = nx.from_graph6_bytes(g6.encode())
    G_orig = G.copy()
    
    metric_orig = calculate_metric(metric, G_orig)

    edge_index = []
    delta_metric = []

    for edge in list(G.edges()):
        G.remove_edge(*edge)
        # if nx.is_connected(G):
        metric_new = calculate_metric(metric, G)
        delta = metric_orig - metric_new
        edge_index.append(edge)
        delta_metric.append(delta)
        G.add_edge(*edge)

    return {
        "graph_id": i,
        "graph6": nx.to_graph6_bytes(G_orig).decode().strip(),
        "edge_index": edge_index,
        metric: delta_metric,
    }

# -------------------------------
# GENERATOR
# -------------------------------

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('argument1', nargs='?', default='algebraic_connectivity', help='The first argument (default: algebraic_connectivity)')
args = parser.parse_args()
metric = args.argument1
print("Metric:", metric)

selection = {
    3: -1,
    4: -1,
    5: -1, 
    6: -1,
    7: -1,
    8: -1
    # 9: -1
    # 10: -1
}
dataset = GraphDataset(selection=selection)
output = []
# dataset = GraphDataset()  # Load all graphs

# import os

# save_dir = Path("criticality_dataset_" + str(metric))
# save_dir.mkdir(exist_ok=True)

# with ProcessPoolExecutor(max_workers=8) as executor:  # LIMITIRAJ radnike
#     futures = {executor.submit(process_graph, i, g6, metric): i for i, g6 in enumerate(tqdm(dataset.graphs(batch_size=1)))}

#     for future in tqdm(as_completed(futures), total=len(futures), desc="Spremanje grafova"):
#         record = future.result()

#         save_path = save_dir / f"graph_{record['graph_id']:06d}.pt"
#         torch.save(record, save_path)


# Use ProcessPoolExecutor to parallelize the metric calculations
with ProcessPoolExecutor() as executor:
    futures = []
    for i, g6 in enumerate(tqdm(dataset.graphs(batch_size=1))):
        futures.append(executor.submit(process_graph, i, g6, metric))

    for future in as_completed(futures):
        output.append(future.result())

path_name = "criticality_dataset_" + str(metric) + "_full.pt"
torch.save(output, Path(path_name))
print("[✓] Dataset spremljen kao " + path_name)