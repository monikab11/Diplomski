import torch
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('argument1', nargs='?', default="network_criticality", help='Metric (default: network_criticality)')
parser.add_argument('argument2', nargs='?', default=0, help='Graph id (default: 0)')
args = parser.parse_args()
metric = args.argument1
graph_id = args.argument2

# Load the dataset
# model = torch.load("/home/jovyan/Diplomski/dataset_novi/criticality_dataset_algebraic_connectivity_1_8.pt", weights_only=False)
model = torch.load("./criticality_dataset_"+metric+".pt", weights_only=False)

# print(graph_idd)
# Check if the model is a list and print the first graph
# print(model[8])
for i in range(29):
    ix = 0
    for graph in model:
        if graph.get('graph_id') == int(i):
            # print(graph)
            print(f"{i} --> {ix}  {graph.get('graph6')}")
        ix += 1


if isinstance(model, list) and len(model) > 0:
    ix = 0
    for graph in model:
        if graph.get('graph_id') == int(graph_id):
            print(graph)
            print(ix)
        ix += 1
else:
    print("The loaded model is not a list or is empty.")