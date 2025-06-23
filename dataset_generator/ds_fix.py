import torch
import networkx as nx # Potrebno za izračun značajki
import numpy as np   # Potrebno za eventualne numpy operacije

# Učitaj dataset
model = torch.load("/home/jovyan/Diplomski/Diplomski/dataset_generator/criticality_dataset_network_criticality_full_and_9.pt", weights_only=False)

# Provjeri da je lista i nije prazna
if isinstance(model, list) and len(model) > 0:
    print(f"Ukupan broj grafova: {len(model)}")

    # Lista značajki koje želite izračunati i spremiti
    # Ovo mora odgovarati listi 'features' koju prosljeđujete EdgeCriticalityDatasetu
    features_to_calculate = ["degree", "clustering", "pagerank", "betweenness", "eigenvector", "closeness", "core number"]

    # Prođi kroz svaki graf i izračunaj značajke
    for i, graph_data in enumerate(model):
        if 'graph6' in graph_data:
            # Rekonstruiraj networkx graf iz graph6 stringa
            G = nx.from_graph6_bytes(graph_data['graph6'].encode())

            # Inicijaliziraj rječnik za spremanje značajki tog grafa
            graph_data['features'] = {}

            # Izračunaj odabrane značajke
            if "degree" in features_to_calculate:
                graph_data['features']['degree'] = [val for _, val in G.degree()]
            if "clustering" in features_to_calculate:
                graph_data['features']['clustering'] = list(nx.clustering(G).values())
            if "pagerank" in features_to_calculate:
                graph_data['features']['pagerank'] = [v for v in nx.pagerank(G).values()]
            if "betweenness" in features_to_calculate:
                graph_data['features']['betweenness'] = [v for v in nx.betweenness_centrality(G).values()]
            if "eigenvector" in features_to_calculate:
                # nx.eigenvector_centrality_numpy može baciti grešku za određene grafove (npr. ako je graf nepovezan)
                try:
                    graph_data['features']['eigenvector'] = [v for v in nx.eigenvector_centrality_numpy(G).values()]
                except nx.PowerIterationFailedConvergence:
                    print(f"Upozorenje: Eigenvector centrality nije konvergirala za graf {graph_data.get('graph_id', i)}. Postavljam na nule.")
                    graph_data['features']['eigenvector'] = [0.0] * G.number_of_nodes()
            if "closeness" in features_to_calculate:
                graph_data['features']['closeness'] = [v for v in nx.closeness_centrality(G).values()]
            if "core number" in features_to_calculate:
                graph_data['features']['core number'] = [v for v in nx.core_number(G).values()]
            
            # Zaokruži algebraic_connectivity vrijednosti
            if 'network_criticality' in graph_data:
                graph_data['network_criticality'] = [round(val, 3) for val in graph_data['network_criticality']]
            
            print(f"Izračunato i spremljeno značajke za graf {i+1}/{len(model)}")
        else:
            print(f"Upozorenje: Graf {i} ne sadrži 'graph6' podatak. Preskačem izračun značajki.")

    # Spremi modificirani dataset u novi .pt fajl
    torch.save(model, "precalculated_features_network_criticality_and_9.pt")
    print("Modificirani dataset je spremljen kao 'precalculated_features_network_criticality_and_9.pt'")
else:
    print("Greška: Dataset nije lista ili je prazan.")

print(model[:1]) # Ispis prvog grafa radi provjere (smanjite na 1 jer je izlaz velik)