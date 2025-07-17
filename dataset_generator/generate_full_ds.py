import torch
import networkx as nx
import numpy as np
from pathlib import Path
from scipy.linalg import pinv
from tqdm import tqdm
import sys
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
INFINITY = sys.maxsize

sys.path.append("/home/jovyan/Diplomski/my_graphs_dataset")
from my_graphs_dataset import GraphDataset

# -------------------------------
# METRICS
# -------------------------------

def algebraic_connectivity(G):
    L = nx.laplacian_matrix(G).toarray()
    lambdas = sorted(np.linalg.eigvalsh(L))
    return lambdas[1]

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
    node_index = []
    edge_delta_metric = []
    node_delta_metric = []

    for edge in list(G.edges()):
        G.remove_edge(*edge)
        # if nx.is_connected(G):
        metric_new = calculate_metric(metric, G)
        if math.isinf(metric_new):
            metric_new = INFINITY
        delta = metric_new - metric_orig
        edge_index.append(edge)
        edge_delta_metric.append(round(delta, 3))
        G.add_edge(*edge)

    for node in list(G.nodes()):
        G.remove_node(node) #*???
        metric_new = calculate_metric(metric, G)
        if math.isinf(metric_new):
            metric_new = INFINITY
        delta = metric_new - metric_orig
        node_index.append(node)
        node_delta_metric.append(round(delta, 3))
        G = G_orig.copy()
        
    return {
        "graph_id": i,
        "graph6": nx.to_graph6_bytes(G_orig).decode().strip(),
        "edge_index": edge_index,
        "edge_" + metric: edge_delta_metric,
        "node_index": node_index,
        "node_" + metric: node_delta_metric
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
    8: -1,
    9: 11000
    # 10: -1
}
dataset = GraphDataset(selection=selection)
output = []

# for i, g6 in enumerate(tqdm(dataset.graphs(batch_size=1))):
#         counter_started += 1
#         res = process_graph(i, g6, metric)
#         output.append(res)
with ProcessPoolExecutor() as executor:
    futures = []
    for i, g6 in enumerate(tqdm(dataset.graphs(batch_size=1))):
        futures.append(executor.submit(process_graph, i, g6, metric))
    for future in as_completed(futures):
        output.append(future.result())
        
print("HERE")
path_name = "criticality_dataset_" + str(metric) + ".pt"
torch.save(output, Path(path_name))
print("Dataset saved as " + path_name)

# -------------------------------
# FEATURES
# -------------------------------

model = torch.load("./" + path_name, weights_only=False)
if isinstance(model, list) and len(model) > 0:
    print(f"Num of graphs: {len(model)}")
    features_to_calculate = ["degree", "clustering", "pagerank", "betweenness", "eigenvector", "closeness", "core number"]

    # calculate features
    for i, graph_data in enumerate(model):
        if 'graph6' in graph_data:
            G = nx.from_graph6_bytes(graph_data['graph6'].encode())
            
            graph_data['features'] = {}
            if "degree" in features_to_calculate:
                graph_data['features']['degree'] = [val for _, val in G.degree()]
            if "clustering" in features_to_calculate:
                graph_data['features']['clustering'] = list(nx.clustering(G).values())
            if "pagerank" in features_to_calculate:
                graph_data['features']['pagerank'] = [v for v in nx.pagerank(G).values()]
            if "betweenness" in features_to_calculate:
                graph_data['features']['betweenness'] = [v for v in nx.betweenness_centrality(G).values()]
            if "eigenvector" in features_to_calculate:
                try:
                    graph_data['features']['eigenvector'] = [v for v in nx.eigenvector_centrality_numpy(G).values()]
                except nx.PowerIterationFailedConvergence:
                    print(f"Upozorenje: Eigenvector centrality nije konvergirala za graf {graph_data.get('graph_id', i)}. Postavljam na nule.")
                    graph_data['features']['eigenvector'] = [0.0] * G.number_of_nodes()
            if "closeness" in features_to_calculate:
                graph_data['features']['closeness'] = [v for v in nx.closeness_centrality(G).values()]
            if "core number" in features_to_calculate:
                graph_data['features']['core number'] = [v for v in nx.core_number(G).values()]
            
            print(f"Calculated graphs {i+1}/{len(model)}")
        else:
            print(f"Skip graf {i}.")

    torch.save(model, path_name)
    print(f"Modified dataset saved.")
else:
    print("Error, ds empty.")

print(model[:1])