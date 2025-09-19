import argparse
import json

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('argument1', nargs='?', default=-1, help='Graph id (default: 0)')
args = parser.parse_args()
graph_id = args.argument1

# Load JSON data
path = "./all_rankings/ranking_net_crit_final.jsonl"

data = []
with open(path, 'r') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print("Greška u dekodiranju JSON-a:", line)

# print(graph_id)
if graph_id == -1:
    for i in range(30):
        result = next((item for item in data if item["graph_id"] == int(i)), None)
        if result:
            print(result)
        else:
            print(f"Element with graph_id {graph_id} not found.")
else:
    result = next((item for item in data if item["graph_id"] == int(graph_id)), None)

    if result:
        print(result)
    else:
        print(f"No element with graph_id {graph_id}.")
