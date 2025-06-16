import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from gnn_model import GraphSAGEEncoder, EdgeScorer
from edge_criticality_dataset import EdgeCriticalityDataset
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import json
from scipy.stats import rankdata
from torch.utils.data import random_split
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures
import argparse
import wandb
from gnn_utils.utils import (
    count_parameters,
    create_combined_histogram,
    create_graph_wandb,
    extract_graphs_from_batch,
    graphs_to_tuple,
)
import time

def train_epoch_fastest(encoder, scorer, loader, optimizer, loss_fn, device, config):
    encoder.train()
    scorer.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        node_emb = encoder(batch.x, batch.edge_index)
        preds = scorer(node_emb, batch.edge_index)
        
        # Simplified loss - just use global median instead of per-graph
        if config.loss_fn == 'combined':
            y_threshold = (batch.y >= torch.median(batch.y)).float()
            loss = loss_fn(preds, batch.y, y_threshold)
        else:
            loss = loss_fn(preds, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


# def train_epoch(encoder, scorer, loader, optimizer, loss_fn, device, config):
#     # todo: model.train()
#     encoder.train()
#     scorer.train()
#     total_loss = 0
#     start_epoch_time = time.time()

#     for batch in tqdm(loader, desc="Training"):
#         batch = batch.to(device)
#         optimizer.zero_grad()

#         node_emb = encoder(batch.x, batch.edge_index) # batch = encoder.batch
#         preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)

#         edge_batch = batch.batch[batch.edge_index[0]]
#         loss = 0
#         graphs = batch.to_data_list()
#         for g in range(batch.num_graphs):
#             mask = (edge_batch == g)
#             if mask.sum() == 0:
#                 continue
#             preds_g = preds[mask]
#             y_g = batch.y[mask].unsqueeze(-1)
#             y_threshold_g = (y_g >= torch.median(y_g)).float()
#             if (config.loss_fn == 'mse' or config.loss_fn == 'ranking'):
#                 loss += loss_fn(preds_g, y_g)
#             elif (config.loss_fn == 'combined'):
#                 loss += loss_fn(preds_g, y_g, y_threshold_g)
   
#         loss = loss / batch.num_graphs  # average over graphs in batch
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         print(f"Passed batch time: {time.time()-start_epoch_time}")
#         start_epoch_time = time.time()

#     return total_loss / len(loader)

def evaluate(encoder, scorer, loader, device, save_path="./all_rankings/ranking_net_crit_dbg.jsonl"):
    encoder.eval()
    scorer.eval()

    all_hausdorff = []
    all_rankings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index)
            preds = scorer(node_emb, batch.edge_index)

            # Split batch into individual graphs
            graphs = batch.to_data_list()
            # preds is for all edges in batch; split preds per graph
            edge_ptr = batch.ptr  # batch.ptr gives cumulative node counts for graphs

            # To split preds and true values per graph, use batch information:
            edge_batch = batch.batch[batch.edge_index[0]]  # edge to graph mapping

            for g, graph in enumerate(graphs):
                mask = (edge_batch == g)
                preds_g = preds[mask].cpu().numpy()
                true_g = graph.y.cpu().numpy()

                # create_graph_wandb(graph)
                

                graph_id = int(graph.graph_id) if hasattr(graph, "graph_id") else g
                if graph_id == 5 or graph_id == 10:
                    print(f"\n[Eval Graph {graph_id}] preds_g: {preds_g[:5].flatten()}")
                    print(f"[Eval Graph {graph_id}] true_g: {true_g[:5].flatten()}")

                preds_g = np.array([round(x, 1) for x in preds_g])
                pred_ranks = rankdata(-preds_g, method='dense')
                true_ranks = rankdata(-true_g, method='dense')

                if graph_id == 5 or graph_id == 10:
                    print(f"[Eval Graph {graph_id}] pred_ranks: {pred_ranks[:5]}")
                    print(f"[Eval Graph {graph_id}] true_ranks: {true_ranks[:5]}")

                all_rankings.append({
                    "graph_id": int(graph.graph_id) if hasattr(graph, "graph_id") else g,
                    "edge_index": graph.edge_index.cpu().t().tolist(),
                    "pred_values": preds_g.tolist(),
                    "true_values": true_g.tolist(),
                    "pred_ranks": pred_ranks.tolist(),
                    "true_ranks": true_ranks.tolist()
                })

                pred_coords = np.expand_dims(pred_ranks, axis=1)
                true_coords = np.expand_dims(true_ranks, axis=1)
                hausdorff_dist = max(
                    directed_hausdorff(pred_coords, true_coords)[0],
                    directed_hausdorff(true_coords, pred_coords)[0]
                )
                all_hausdorff.append(hausdorff_dist)

    avg_hausdorff = np.mean(all_hausdorff)

    # OTKOMENTIRAJ ZA SPREMANJE!!!!!!!!!!!!!!!!!!!!!!!!!
    with open(save_path, "w") as f:
        for result, dist in zip(all_rankings, all_hausdorff):
            compact_result = {
                "graph_id": int(result["graph_id"]),
                "pred_ranks": [int(x) for x in result["pred_ranks"]],
                "true_ranks": [int(x) for x in result["true_ranks"]],
                "hausdorff": float(dist)
            }
            f.write(json.dumps(compact_result) + "\n")

    # print(f"[Evaluation] Average Hausdorff distance: {avg_hausdorff:.4f}")
    return all_rankings, avg_hausdorff


def validate_epoch(encoder, scorer, loader, loss_fn, device, config):
    encoder.eval()
    scorer.eval()
    total_loss = 0

    metrics = {
        'hausdorff': [],
        'all_match': 0,
        'all_pred_in_true': 0,
        'all_true_in_pred': 0,
        'all_true_in_pred_1_2': 0,
        'total_graphs': 0
    }
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index)
            preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)
            
            edge_batch = batch.batch[batch.edge_index[0]]
            loss = 0
            for g in range(batch.num_graphs):
                mask = (edge_batch == g)
                if mask.sum() == 0:
                    continue
                preds_g = preds[mask]
                y_g = batch.y[mask].unsqueeze(-1)
                
                # Use same loss calculation as training                
                if (config.loss_fn == 'mse' or config.loss_fn == 'ranking'):
                    loss += loss_fn(preds_g, y_g)
                elif (config.loss_fn == 'combined'):
                    y_threshold_g = (y_g >= torch.median(y_g)).float() #.unsqueeze(-1)
                    loss += loss_fn(preds_g, y_g, y_threshold_g)

                pred_ranks = rankdata(-preds_g.cpu().numpy(), method='dense')
                true_ranks = rankdata(-y_g.cpu().numpy(), method='dense')

                # Izračun metrika za svaki graf
                # pred_ranks = rankdata(-preds_g, method='dense')
                # true_ranks = rankdata(-true_g, method='dense')

                # graph_id = int(g.graph_id) if hasattr(g, "graph_id") else g
                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} pred: {pred_ranks}")
                # print(f"{graph_id} true: {true_ranks}")
                
                # Metrike
                pred_top = np.where(pred_ranks == 1)[0]
                true_top = np.where(true_ranks == 1)[0]
                pred_top_1_2 = np.where((pred_ranks == 1) | (pred_ranks == 2))[0]

                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} pred_top: {pred_top}")
                # print(f"true_top: {true_top}")
                # print(f"{graph_id} pred_top_1_2: {pred_top_1_2}")

                metrics['all_match'] += int(np.array_equal(pred_top, true_top))
                metrics['all_pred_in_true'] += int(all(elem in true_top for elem in pred_top))
                metrics['all_true_in_pred'] += int(all(elem in pred_top for elem in true_top))
                metrics['all_true_in_pred_1_2'] += int(all(elem in pred_top_1_2 for elem in true_top))
                metrics['total_graphs'] += 1

                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} all_match: {int(np.array_equal(pred_top, true_top))}")
                # print(f"true_top all_pred_in_true: {int(all(elem in true_top for elem in pred_top))}")
                # print(f"true_top all_true_in_pred: {int(all(elem in pred_top for elem in true_top))}")
                # print(f"{graph_id} all_true_in_pred_1_2: {int(all(elem in pred_top_1_2 for elem in true_top))}")
                    
            total_loss += loss / batch.num_graphs

    metrics = {
        'avg_hausdorff': np.mean(metrics['hausdorff']),
        'match_pct': metrics['all_match'] / metrics['total_graphs'],
        'pred_in_true_pct': metrics['all_pred_in_true'] / metrics['total_graphs'],
        'true_in_pred_pct': metrics['all_true_in_pred'] / metrics['total_graphs'],
        'true_in_pred_1_2_pct': metrics['all_true_in_pred_1_2'] / metrics['total_graphs'],
        'test_loss': total_loss / len(loader)
    }
            
    return metrics

def validate_epoch_fast(encoder, scorer, loader, device):
    encoder.eval()
    scorer.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass za cijeli batch
            node_emb = encoder(batch.x, batch.edge_index)
            preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)
            
            # Izračun MSE loss za cijeli batch
            loss = F.mse_loss(preds, batch.y.unsqueeze(-1))
            
            total_loss += loss.item()

    print(len(loader))
    return total_loss / len(loader)

def validate_epoch_new(encoder, scorer, loader, loss_fn, device, config):
    encoder.eval()
    scorer.eval()
    total_loss = 0

    metrics = {
        'hausdorff': [],
        'all_match': 0,
        'all_pred_in_true': 0,
        'all_true_in_pred': 0,
        'all_true_in_pred_1_2': 0,
        'total_graphs': 0
    }
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index)
            preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)
            
            edge_batch = batch.batch[batch.edge_index[0]]
            loss = 0
            for g in range(batch.num_graphs):
                mask = (edge_batch == g)
                if mask.sum() == 0:
                    continue
                preds_g = preds[mask]
                y_g = batch.y[mask].unsqueeze(-1)
                
                # Use same loss calculation as training                
                if (config.loss_fn == 'mse' or config.loss_fn == 'ranking'):
                    loss += loss_fn(preds_g, y_g)
                elif (config.loss_fn == 'combined'):
                    y_threshold_g = (y_g >= torch.median(y_g)).float() #.unsqueeze(-1)
                    loss += loss_fn(preds_g, y_g, y_threshold_g)

                pred_ranks = rankdata(-preds_g.cpu().numpy(), method='dense')
                true_ranks = rankdata(-y_g.cpu().numpy(), method='dense')

                # Izračun metrika za svaki graf
                # pred_ranks = rankdata(-preds_g, method='dense')
                # true_ranks = rankdata(-true_g, method='dense')

                # graph_id = int(g.graph_id) if hasattr(g, "graph_id") else g
                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} pred: {pred_ranks}")
                # print(f"{graph_id} true: {true_ranks}")
                
                # Metrike
                pred_top = np.where(pred_ranks == 1)[0]
                true_top = np.where(true_ranks == 1)[0]
                pred_top_1_2 = np.where((pred_ranks == 1) | (pred_ranks == 2))[0]

                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} pred_top: {pred_top}")
                # print(f"true_top: {true_top}")
                # print(f"{graph_id} pred_top_1_2: {pred_top_1_2}")

                metrics['all_match'] += int(np.array_equal(pred_top, true_top))
                metrics['all_pred_in_true'] += int(all(elem in true_top for elem in pred_top))
                metrics['all_true_in_pred'] += int(all(elem in pred_top for elem in true_top))
                metrics['all_true_in_pred_1_2'] += int(all(elem in pred_top_1_2 for elem in true_top))
                metrics['total_graphs'] += 1

                # if graph_id == 5 or graph_id == 10:
                # print(f"{graph_id} all_match: {int(np.array_equal(pred_top, true_top))}")
                # print(f"true_top all_pred_in_true: {int(all(elem in true_top for elem in pred_top))}")
                # print(f"true_top all_true_in_pred: {int(all(elem in pred_top for elem in true_top))}")
                # print(f"{graph_id} all_true_in_pred_1_2: {int(all(elem in pred_top_1_2 for elem in true_top))}")
                    
            total_loss += loss / batch.num_graphs

    metrics = {
        'avg_hausdorff': np.mean(metrics['hausdorff']),
        'match_pct': metrics['all_match'] / metrics['total_graphs'],
        'pred_in_true_pct': metrics['all_pred_in_true'] / metrics['total_graphs'],
        'true_in_pred_pct': metrics['all_true_in_pred'] / metrics['total_graphs'],
        'true_in_pred_1_2_pct': metrics['all_true_in_pred_1_2'] / metrics['total_graphs'],
        'test_loss': total_loss / len(loader)
    }
            
    return metrics



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--margin', type=float, default=0.1)  # used if loss is "ranking"
    parser.add_argument("--no-wandb", action="store_true", help="Do not use W&B for logging.")
    parser.add_argument("--features", type=list, default=["degree"])
    parser.add_argument("--edge_feature_mode", type=str, default='dot')
    parser.add_argument("--loss_fn", type=str, default='mse')
    parser.add_argument("--feature_normalization", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    config = args
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and normalize features
    if (config.feature_normalization):
        transform = NormalizeFeatures()
        dataset = EdgeCriticalityDataset(path="/home/jovyan/Diplomski/Diplomski/dataset_generator/precalculated_features_enetwork_criticality.pt",  metric_name="network_criticality", transform=transform, features=config.features)
    else:
        dataset = EdgeCriticalityDataset(path="/home/jovyan/Diplomski/Diplomski/dataset_generator/precalculated_features_network_criticality.pt", metric_name="network_criticality", features=config.features)
        
    # Split ds for train and test
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=16) # mozda ovdje ipak 1????
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    # GNN model, optimizer
    encoder = GraphSAGEEncoder(in_channels=len(config.features), hidden_channels=config.hidden_channels, num_layers=config.gnn_layers).to(device)
    scorer = EdgeScorer(node_dim=config.hidden_channels, hidden_dim=config.hidden_channels, edge_feat_mode='dot').to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()), 
        lr=config.learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)  # beta1 is momentum term here
    )
    # criterion = torch.nn.L1Loss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # loss function
    if (config.loss_fn == 'mse'):
        loss_fn = F.mse_loss

    # training...
    try:
        for epoch in range(1, 100):
            train_loss = train_epoch_fastest(encoder, scorer, train_loader, optimizer, loss_fn, device, config=config)
            print(f"Epoch {epoch}, Loss: {train_loss:.4f}")
            eval_metrics = validate_epoch(encoder, scorer, val_loader, loss_fn, device, config)
            print(f"traintest_loss: {eval_metrics['test_loss']}, match_pct: {eval_metrics['match_pct']}")
            res = validate_epoch_fast(encoder, scorer, val_loader, device)
            print("DONE")
            print(res)
            scheduler.step(train_loss)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Batch size 128 is too large, try 64 or 32")

    print("Evaluating model...")
    all_rankings, avg_hausdorff = evaluate(encoder, scorer, val_loader, device)

    # Primjer ispisa prvih 3 rezultata
    for i, result in enumerate(all_rankings[:3]):
        print(f"\nGraph ID: {result['graph_id']}")
        print("Edge Index:", result["edge_index"])
        print("True Ranks:", result["true_ranks"])
        print("Predicted Ranks:", result["pred_ranks"])

if __name__ == "__main__":
    run = main()
    # run.finish() # ako nije sweep
