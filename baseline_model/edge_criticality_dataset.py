import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np

class EdgeCriticalityDataset(Dataset):
    def __init__(self, path, metric_name="algebraic_connectivity", transform=None, pre_transform=None, features=None):
        super().__init__(path, transform, pre_transform)
        self.data_list = torch.load(path, weights_only=False)
        self.metric_name = metric_name
        self.features = features

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        item = self.data_list[idx]
        G = nx.from_graph6_bytes(item['graph6'].encode())
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long).t().contiguous()
        # x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)

        selected_features = []
        if 'features' in item and self.features:
            for feature_name in self.features:
                if feature_name in item['features']:
                    # Važno: Ensure that if a feature is missing or None, it's handled,
                    # e.g., by providing a default like zeros or skipping it.
                    # Assuming pre-calculation ensures all selected features exist.
                    feature_values = item['features'][feature_name]
                    selected_features.append(torch.tensor(feature_values, dtype=torch.float).unsqueeze(1))
                else:
                    print(f"Upozorenje: Značajka '{feature_name}' nije pronađena za graf {item.get('graph_id', idx)}.")
                    # Ovisno o tome kako želite rukovati, možete dodati nule ili preskočiti
                    # Na primjer, dodati vektor nula ako značajka nije pronađena:
                    # selected_features.append(torch.zeros((item['num_nodes'], 1), dtype=torch.float)) # Pretpostavlja da znate broj čvorova
        else:
            raise ValueError("Pre-izračunate značajke ('features' ključ) nisu pronađene u datasetu ili self.features nije definiran.")
        
        if not selected_features:
            raise ValueError("Nijedna od traženih značajki nije pronađena ili je prazna za ovaj graf.")       
        
        x = torch.cat(selected_features, dim=1)
        y = torch.tensor(item[self.metric_name], dtype=torch.float)
        graph_id = torch.tensor(item['graph_id'], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y, graph_id=graph_id)
