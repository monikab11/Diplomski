# Load the .pt file
# import torch
# model = torch.load("/home/jovyan/Diplomski/dataset_novi/criticality_dataset_effective_graph_resistance.pt", weights_only=False)
# # model = torch.load('your_model.pt')

# # If the model is a dictionary, you can access its keys
# if isinstance(model, dict):
#     for key in list(model.keys())[1]:  # Get the first 10 keys
#         print(key, model[key])
# else:
#     print(model)

# import multiprocessing

# num_processors = multiprocessing.cpu_count()
# print(f"Number of processors: {num_processors}")

import torch
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('argument1', nargs='?', default=0, help='Graph id (default: 0)')
args = parser.parse_args()
graph_id = args.argument1

# Load the dataset
# model = torch.load("/home/jovyan/Diplomski/dataset_novi/criticality_dataset_algebraic_connectivity_1_8.pt", weights_only=False)
model = torch.load("/home/jovyan/Diplomski/Diplomski/dataset_generator/precalculated_features_network_criticality.pt", weights_only=False)

# print(graph_idd)
# Check if the model is a list and print the first graph
if isinstance(model, list) and len(model) > 0:
    for graph in model:
        if graph.get('graph_id') == int(graph_id):
            print(graph)
else:
    print("The loaded model is not a list or is empty.")