from torch_geometric.data import Data, Dataset
import networkx as nx
import torch


class CriticalityDataset(Dataset):
    def __init__(self, path, metric_name="network_criticality", transform=None, pre_transform=None, features=None, is_edge=True):
        super().__init__(path, transform, pre_transform)
        self.data_list = torch.load(path, weights_only=False)
        self.metric_name = metric_name
        self.features = features
        self.is_edge = is_edge

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        item = self.data_list[idx]
        G = nx.from_graph6_bytes(item['graph6'].encode())
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long).t().contiguous()

        selected_features = []
        if 'features' in item and self.features:
            for feature_name in self.features:
                if feature_name in item['features']:
                    feature_values = item['features'][feature_name]
                    selected_features.append(torch.tensor(feature_values, dtype=torch.float).unsqueeze(1))
                else:
                    raise ValueError(f"Feature{feat} is missing for graph {idx}")
        else:
            raise ValueError("self.features is not defined.")
        
        if not selected_features:
            raise ValueError("No features for this graph.")       
        
        x = torch.cat(selected_features, dim=1)
        if self.is_edge:
            if self.metric_name in ["algebraic_connectivity", "network_criticality", "node_connectivity", "edge_connectivity", "nc1", "nc2", "nc3"]:
                y = -1.0 * torch.tensor(item["edge_"+self.metric_name], dtype=torch.float)
            else:
                y = torch.tensor(item["edge_"+self.metric_name], dtype=torch.float)
        else:
            if self.metric_name in ["algebraic_connectivity", "network_criticality", "node_connectivity", "edge_connectivity", "nc1", "nc2", "nc3"]:
                y = -1.0 * torch.tensor(item["node_"+self.metric_name], dtype=torch.float)
            else:
                y = torch.tensor(item["node_"+self.metric_name], dtype=torch.float)
            
        graph_id = torch.tensor(item['graph_id'], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y, graph_id=graph_id)
