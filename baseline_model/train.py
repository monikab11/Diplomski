from gnn_model import GraphSAGEEncoder, GCNEncoder, EdgeScorer, NodeScorer
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss, BCEWithLogitsLoss
from torch_geometric.utils import to_undirected, to_dense_batch, scatter
from torch_geometric.transforms import NormalizeFeatures
from scipy.stats import rankdata, spearmanr, kendalltau
from criticality_dataset import CriticalityDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import itertools
import argparse
import torch
import wandb
import json
import os

min_scorer_value = 100
max_scorer_value = -100


class PureRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, max_pairs=5000):
        super().__init__()
        self.margin_ranking_loss = MarginRankingLoss(margin=margin)
        self.max_pairs = max_pairs

    def forward(self, outputs, y):
        outputs = outputs.view(-1)
        y = y.view(-1)
        n = outputs.size(0)

        if n < 2:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        max_possible = n * (n - 1) // 2

        if max_possible <= self.max_pairs:
            # all possible (i, j), where i < j
            pairs = torch.tensor(list(itertools.combinations(range(n), 2)), device=outputs.device)
        else:
            # random max_pairs pairs
            idx_i, idx_j = torch.randint(0, n, (2, self.max_pairs * 2), device=outputs.device)

            mask = (idx_i != idx_j) & (idx_i < idx_j)
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]

            if idx_i.numel() == 0:
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)

            pairs = torch.stack([idx_i, idx_j], dim=1)
            pairs = torch.unique(pairs, dim=0)

            if pairs.size(0) > self.max_pairs:
                rand_idx = torch.randperm(pairs.size(0), device=outputs.device)[:self.max_pairs]
                pairs = pairs[rand_idx]

        idx_i, idx_j = pairs[:, 0], pairs[:, 1]

        if idx_i.numel() == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        y_diff = y[idx_i] - y[idx_j]
        target = torch.sign(y_diff)
        valid = target != 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        return self.margin_ranking_loss(
            outputs[idx_i][valid],
            outputs[idx_j][valid],
            target[valid]
        )


class CombinedRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, max_pairs=5000):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.margin_ranking_loss = MarginRankingLoss(margin=margin)
        self.max_pairs = max_pairs

    def forward(self, outputs, y, y_threshold):
        bce = self.bce_loss(outputs, y_threshold)

        outputs = outputs.view(-1)
        y = y.view(-1)
        n = outputs.size(0)

        # Sample pairs (i, j)
        idx_i, idx_j = torch.randint(0, n, (2, self.max_pairs), device=outputs.device)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        if idx_i.numel() == 0:
            return bce

        y_diff = y[idx_i] - y[idx_j]
        target = torch.sign(y_diff)
        valid = target != 0

        if valid.sum() == 0:
            return bce

        mr_loss = self.margin_ranking_loss(outputs[idx_i][valid], outputs[idx_j][valid], target[valid])
        return bce + mr_loss


class BatchMarginRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.loss_fn = MarginRankingLoss(margin=margin, reduction="none")

    def forward(self, outputs, y, edges_batch):
        """
        Computes the Margin Ranking Loss for a batch of graphs.

        This function adapts the standard MarginRankingLoss to work on batches of
        graphs as handled by PyTorch Geometric. It compares all pairs of edges
        within each graph in the batch.

        Args:
            outputs (torch.Tensor): A 1D tensor of model outputs for each edge.
            y (torch.Tensor): A 1D tensor of ground truth labels for each edge.
            edge_index (torch.Tensor): A [2, num_edges] tensor representing the
                                    graph edges.
            batch_nodes (torch.Tensor): A 1D tensor mapping each node to its
                                        corresponding graph in the batch.

        Returns:
            torch.Tensor: The computed margin ranking loss as a single scalar tensor.
        """
        # 2. Generate edge combinations within each graph
        # Get the global indices of edges
        edge_indices_global = torch.arange(len(outputs), device=outputs.device)

        # Use to_dense_batch to group edges by their graph index.
        # It returns a dense tensor where rows are graphs and columns are edges,
        # padded with -1. It also provides a mask.
        dense_edge_indices, mask = to_dense_batch(edge_indices_global, edges_batch, fill_value=-1)

        i, j = self.batched_combinations(dense_edge_indices, mask)

        # 3. Create the target tensor
        # Compare the ground truth labels for the paired edges.
        # target = 1 if y[i] > y[j], else -1
        target = torch.sign(y[i] - y[j])

        # 4. Compute the loss
        # Select the model outputs corresponding to the paired edges
        outputs_i = outputs[i]
        outputs_j = outputs[j]

        # Compute and return the final loss
        loss = self.loss_fn(outputs_i, outputs_j, target)
        # Create a batch tensor indicating which graph each loss element belongs to
        # For each pair (i, j), both i and j belong to the same graph, so we can use edges_batch[i]
        loss_batch = edges_batch[i]

        # Compute mean loss for each graph using scatter
        mean_loss_per_graph = scatter(loss, loss_batch, reduce="mean")

        # The final loss is the mean of the per-graph mean losses
        final_loss = mean_loss_per_graph.mean()

        return final_loss

    @staticmethod
    def batched_combinations(dense_global_indices, mask):
        # Get the number of edges in each graph
        num_edges_per_graph = mask.sum(dim=1)

        # Find the maximum number of edges in any single graph
        max_edges_in_graph = num_edges_per_graph.max().int()

        if max_edges_in_graph < 2:
            return torch.tensor(0.0, device=dense_global_indices.device, requires_grad=True)

        # Create a template of combinations for a graph of size `max_edges_in_graph`
        template_combos = torch.combinations(torch.arange(max_edges_in_graph, device=dense_global_indices.device))
        i_local = template_combos[:, 0]
        j_local = template_combos[:, 1]

        # Use the template to gather the global indices for all possible pairs
        # Shape: [num_graphs, num_combinations_in_template]
        i_global_candidates = dense_global_indices[:, i_local]
        j_global_candidates = dense_global_indices[:, j_local]

        # Create a mask to filter out invalid combinations. A combination is valid
        # only if both of its local indices are less than the number of edges in
        # that specific graph. We only need to check the larger index, j_local.
        combination_mask = j_local < num_edges_per_graph.unsqueeze(1)

        # Apply the mask to get the final, valid pairs of global edge indices
        i = i_global_candidates[combination_mask]
        j = j_global_candidates[combination_mask]

        return i, j


def train_epoch(encoder, scorer, loader, optimizer, loss_fn, device, config):
    global min_scorer_value
    global max_scorer_value
    encoder.train()
    scorer.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        node_emb = encoder(batch.x, batch.edge_index)
        if config.prediction_target == "edge":
            preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)
            batch_index = batch.batch[batch.edge_index[0]]
        else:
            preds = scorer(node_emb).unsqueeze(-1)
            batch_index = batch.batch

        if preds.min() < min_scorer_value:
            min_scorer_value = preds.min()
        if preds.max() > max_scorer_value:
            max_scorer_value = preds.max()
        
        loss = 0

        if config.loss_fn == 'batch_ranking':
            epoch_loss = loss_fn(preds.squeeze(-1), batch.y.squeeze(-1), batch_index)
            total_loss += epoch_loss
            epoch_loss.backward()
            optimizer.step()
        else:
            for g in range(batch.num_graphs):
                mask = (batch_index == g)
                if mask.sum() == 0:
                    continue
    
                preds_g = preds[mask]
                y_g = batch.y[mask].unsqueeze(-1)
    
                if config.loss_fn == 'combined':
                    y_threshold_g = (y_g >= torch.median(y_g)).float()
                    loss += loss_fn(preds_g, y_g, y_threshold_g)
                elif config.loss_fn == 'mse':
                    loss += torch.nn.MSELoss()(preds_g, y_g)   
                else:
                    loss += loss_fn(preds_g, y_g) 
                    
            loss = loss / batch.num_graphs  # average over graphs in batch
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(loader)


def train_epoch_fastest(encoder, scorer, loader, optimizer, loss_fn, device, config):
    encoder.train()
    scorer.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        node_emb = encoder(batch.x, batch.edge_index)
        if config.prediction_target == "edge":
            preds = scorer(node_emb, batch.edge_index)
        else:
            preds = scorer(node_emb)
        
        # Simplified loss - use global median instead of per-graph
        if config.loss_fn == 'combined':
            y_threshold = (batch.y >= torch.median(batch.y)).float()
            loss = loss_fn(preds, batch.y, y_threshold)
        else:
            loss = loss_fn(preds, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def top_n_percent_accuracy(true, pred, n_percent=0.1):
    k = max(1, int(len(true) * n_percent))
    top_true = set(np.argsort(-true)[:k])
    top_pred = set(np.argsort(-pred)[:k])
    intersection = top_true & top_pred
    return len(intersection) / len(top_true)


def compute_graph_ranking_metrics(true, pred, top_k_ratio=0.1):
    true = np.asarray(true)
    pred = np.asarray(pred)

    if len(true) < 2:
        return None

    true_ndcg = true.copy()
    if true_ndcg.min() < 0:
        true_ndcg = true_ndcg - true_ndcg.min()

    if np.all(true == true[0]) or np.all(pred == pred[0]):
        spearman_corr = np.nan
        kendall_corr = np.nan
    else:
        spearman_corr, _ = spearmanr(true, pred)
        kendall_corr, _ = kendalltau(true, pred)

    true_2d = true_ndcg.reshape(1, -1)
    pred_2d = pred.reshape(1, -1)
    ndcg = ndcg_score(true_2d, pred_2d, k=min(len(true), 10))

    top_n_acc = top_n_percent_accuracy(true, pred, n_percent=top_k_ratio)

    return {
        "spearman": spearman_corr,
        "kendall": kendall_corr,
        "ndcg": ndcg,
        "top_n_percent": top_n_acc
    }


def evaluate(encoder, scorer, loader, device, config, save_path="./all_rankings/ranking_net_crit_final.jsonl"):
    encoder.eval()
    scorer.eval()

    all_rankings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index)
            if config.prediction_target == "edge":
                preds = scorer(node_emb, batch.edge_index)
                batch_index = batch.batch[batch.edge_index[0]]
            else:
                preds = scorer(node_emb)
                batch_index = batch.batch

            graphs = batch.to_data_list()

            for g, graph in enumerate(graphs):
                mask = (batch_index == g)
                preds_g = preds[mask].cpu().numpy()
                true_g = graph.y.cpu().numpy()

                preds_g = np.array([round(x, 1) for x in preds_g])
                pred_ranks = rankdata(-preds_g, method='dense')
                true_ranks = rankdata(-true_g, method='dense')

                all_rankings.append({
                    "is_edge": config.prediction_target == "edge",
                    "graph_id": int(graph.graph_id) if hasattr(graph, "graph_id") else g,
                    "edge_node_index": graph.edge_index.cpu().t().tolist() if config.prediction_target == "edge" else list(range(graph.x.size(0))),
                    "pred_values": preds_g.tolist(),
                    "true_values": true_g.tolist(),
                    "pred_ranks": pred_ranks.tolist(),
                    "true_ranks": true_ranks.tolist()
                })

    with open(save_path, "w") as f:
        for result in all_rankings:
            compact_result = {
                "graph_id": int(result["graph_id"]),
                "pred_ranks": [int(x) for x in result["pred_ranks"]],
                "true_ranks": [int(x) for x in result["true_ranks"]]
            }
            f.write(json.dumps(compact_result) + "\n")

    return all_rankings


def validate_epoch(encoder, scorer, loader, loss_fn, device, config):
    encoder.eval()
    scorer.eval()
    total_loss = 0

    metrics = {
        'all_match': 0,
        'all_pred_in_true': 0,
        'all_true_in_pred': 0,
        'all_true_in_pred_1_2': 0,
        'total_graphs': 0,
        'spearman_list': [],
        'kendall_list': [],
        'ndcg_list': [],
        'topn_list': []
    }
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index)
            if config.prediction_target == "edge":
                preds = scorer(node_emb, batch.edge_index).unsqueeze(-1)
                batch_index = batch.batch[batch.edge_index[0]]
            else:
                preds = scorer(node_emb).unsqueeze(-1)
                batch_index = batch.batch
            
            loss = 0
            if config.loss_fn == 'batch_ranking':
                loss = loss_fn(preds.squeeze(-1), batch.y.squeeze(-1), batch_index)
            for g in range(batch.num_graphs):
                mask = (batch_index == g)
                if mask.sum() == 0:
                    continue
                preds_g = preds[mask]
                y_g = batch.y[mask].unsqueeze(-1)
                
                # Use same loss calculation as training                
                if config.loss_fn == 'mse' or config.loss_fn == 'ranking':
                    loss += loss_fn(preds_g, y_g)
                elif config.loss_fn == 'combined':
                    y_threshold_g = (y_g >= torch.median(y_g)).float() #.unsqueeze(-1)
                    loss += loss_fn(preds_g, y_g, y_threshold_g)

                pred_ranks = rankdata(-preds_g.cpu().numpy(), method='dense')
                true_ranks = rankdata(-y_g.cpu().numpy(), method='dense')

                # Izračun metrika za svaki graf
                pred_top = np.where(pred_ranks == 1)[0]
                true_top = np.where(true_ranks == 1)[0]
                pred_top_1_2 = np.where((pred_ranks == 1) | (pred_ranks == 2))[0]

                metrics['all_match'] += int(np.array_equal(pred_top, true_top))
                metrics['all_pred_in_true'] += int(all(elem in true_top for elem in pred_top))
                metrics['all_true_in_pred'] += int(all(elem in pred_top for elem in true_top))
                metrics['all_true_in_pred_1_2'] += int(all(elem in pred_top_1_2 for elem in true_top))
                metrics['total_graphs'] += 1
                true_np = y_g.squeeze().cpu().numpy()
                pred_np = preds_g.squeeze().cpu().numpy()
                graph_metrics = compute_graph_ranking_metrics(true_np, pred_np, top_k_ratio=0.1)
                metrics['spearman_list'].append(graph_metrics['spearman'])
                metrics['kendall_list'].append(graph_metrics['kendall'])
                metrics['ndcg_list'].append(graph_metrics['ndcg'])
                metrics['topn_list'].append(graph_metrics['top_n_percent'])

            if config.loss_fn == 'batch_ranking':
                total_loss += loss    
            else:
                total_loss += loss / batch.num_graphs

    valid_spearman = [v for v in metrics['spearman_list'] if not np.isnan(v)]
    valid_kendall = [v for v in metrics['kendall_list'] if not np.isnan(v)]

    metrics = {
        'match_pct': metrics['all_match'] / metrics['total_graphs'],
        'pred_in_true_pct': metrics['all_pred_in_true'] / metrics['total_graphs'],
        'true_in_pred_pct': metrics['all_true_in_pred'] / metrics['total_graphs'],
        'true_in_pred_1_2_pct': metrics['all_true_in_pred_1_2'] / metrics['total_graphs'],
        'test_loss': total_loss / len(loader),
        'spearman_avg': sum(valid_spearman) / len(valid_spearman) if valid_spearman else np.nan,
        'kendall_avg': sum(valid_kendall) / len(valid_kendall) if valid_kendall else np.nan,
        'ndcg_avg': sum(metrics['ndcg_list']) / metrics['total_graphs'],
        'topn_avg': sum(metrics['topn_list']) / metrics['total_graphs']
    }
            
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--train_batch', type=int, default=2048)
    parser.add_argument("--test_batch", type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=100) # 100
    parser.add_argument('--margin', type=float, default=0.1)  # used if loss is "ranking"
    parser.add_argument("--no-wandb", action="store_true", help="Do not use W&B for logging.")
    parser.add_argument("--features", type=list, default=["degree","betweenness","eigenvector","closeness"])
    parser.add_argument("--edge_feature_mode", type=str, default='concat')
    parser.add_argument("--loss_fn", type=str, default='batch_ranking')
    parser.add_argument("--feature_normalization", type=bool, default=False)
    parser.add_argument("--architecture", type=str, default='GraphSAGE')   
    parser.add_argument("--prediction_target", type=str, default="node", choices=["edge", "node"]) # edge
    return parser.parse_args()

def main():
    best_model_state = None
    best_metrics = None
    best_match_pct = -1
    BEST_MODEL_PATH = "./best_models/best_model.pt"
    global min_scorer_value
    global max_scorer_value

    WANDB_ENABLED = False

    args = parse_args()
    config = args
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if WANDB_ENABLED:
        wandb.init(project="GNN_diplomski_7_mj", config=args)
        config = wandb.config    

    # Load the dataset and normalize features
    if config.feature_normalization:
        transform = NormalizeFeatures()
        dataset = CriticalityDataset(path="../dataset_generator/criticality_dataset_network_criticality.pt",  metric_name="network_criticality", transform=transform, features=config.features, is_edge=config.prediction_target=="edge")
    else:
        dataset = CriticalityDataset(path="../dataset_generator/criticality_dataset_network_criticality.pt", metric_name="network_criticality", features=config.features, is_edge=config.prediction_target=="edge")
        
    # Split ds for train and test
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.test_batch, shuffle=False)
    val_loader2 = DataLoader(dataset, batch_size=config.test_batch, shuffle=False)

    # GNN model, optimizer
    if config.architecture == 'GCN':
        encoder = GCNEncoder(in_channels=len(config.features), hidden_channels=config.hidden_channels, num_layers=config.gnn_layers).to(device)
    elif config.architecture == 'GraphSAGE':
        encoder = GraphSAGEEncoder(in_channels=len(config.features), hidden_channels=config.hidden_channels, num_layers=config.gnn_layers).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
        
    if config.prediction_target == "edge":
        scorer = EdgeScorer(node_dim=config.hidden_channels, hidden_dim=config.hidden_channels, edge_feat_mode=config.edge_feature_mode).to(device)
    else:
        scorer = NodeScorer(node_dim=config.hidden_channels, hidden_dim=config.hidden_channels).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(scorer.parameters()), 
        lr=config.learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)  # beta1 is momentum term here
    )
    # criterion = torch.nn.L1Loss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # loss function
    match config.loss_fn:
        case 'mse':
            loss_fn = F.mse_loss
        case 'ranking':
            loss_fn = PureRankingLoss(margin=config.margin)
        case 'combined':
            loss_fn = CombinedRankingLoss(margin=0.1)
        case 'batch_ranking':
             loss_fn = BatchMarginRankingLoss(margin=config.margin)
        case _:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

    # training...
    early_stopping_patience = 15
    epochs_without_improvement = 0
    try:
        for epoch in range(1, config.epochs+1):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # train_loss = train_epoch_fastest(encoder, scorer, train_loader, optimizer, loss_fn, device, config=config)
            train_loss = train_epoch(encoder, scorer, train_loader, optimizer, loss_fn, device, config=config)
            print(f"Epoch {epoch}, Loss: {train_loss:.4f}")
            eval_metrics = validate_epoch(encoder, scorer, val_loader, loss_fn, device, config)
            print(f"train test_loss: {eval_metrics['test_loss']}, match_pct: {eval_metrics['match_pct']}")
            if eval_metrics['match_pct'] > best_match_pct:
                best_match_pct = eval_metrics['match_pct']
                best_model_state = {
                    "architecture": config.architecture,
                    "in_channels": len(config.features),
                    "hidden_channels": config.hidden_channels,
                    "num_layers": config.gnn_layers,
                    "out_channels": config.hidden_channels,
                    "state_dict": encoder.state_dict(),
                    "encoder": encoder.state_dict(),
                    "scorer": scorer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "prediction_target": config.prediction_target,
                    "edge_feat_mode": config.edge_feature_mode,
                    "features": config.features,
                    "min_value": min_scorer_value.item(),
                    "max_value": max_scorer_value.item()
                }
                
                os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
                best_metrics = eval_metrics.copy()
                torch.save(best_model_state, BEST_MODEL_PATH)
                print(f"Saving model (epoch {epoch}) u '{BEST_MODEL_PATH}'")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

            if WANDB_ENABLED:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": eval_metrics['test_loss'], 
                    "match_pct": eval_metrics['match_pct'],
                    "pred_in_true_pct": eval_metrics['pred_in_true_pct'],
                    "true_in_pred_pct": eval_metrics['true_in_pred_pct'],
                    "true_in_pred_1_2_pct": eval_metrics['true_in_pred_1_2_pct'],
                    "spearman": eval_metrics["spearman_avg"],
                    "kendall": eval_metrics["kendall_avg"],
                    "ndcg": eval_metrics["ndcg_avg"],
                    "topn@10%": eval_metrics["topn_avg"]
                })
            scheduler.step(train_loss)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Batch size {config.train_batch} is too large, try 64 or 32")

    if best_model_state:
        encoder.load_state_dict(best_model_state["encoder"])
        scorer.load_state_dict(best_model_state["scorer"])

        if WANDB_ENABLED:
            wandb.log({
                "best_epoch": best_model_state["epoch"],
                "best_match_pct": best_metrics['match_pct'],
                "best_test_loss": best_metrics['test_loss'],
                "best_pred_in_true_pct": best_metrics['pred_in_true_pct'],
                "best_true_in_pred_pct": best_metrics['true_in_pred_pct'],
                "best_true_in_pred_1_2_pct": best_metrics['true_in_pred_1_2_pct'],
                "spearman": best_metrics["spearman_avg"],
                "kendall": best_metrics["kendall_avg"],
                "ndcg": best_metrics["ndcg_avg"],
                "topn@10%": best_metrics["topn_avg"]
            })
    
        print(f"Best model (epoch {best_model_state['epoch']}):")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("Evaluating model...") # TODO: on the best one
    all_rankings = evaluate(encoder, scorer, val_loader, device, config)
    # all_rankings = evaluate(encoder, scorer, val_loader, device, config)

    # output examples
    for i, result in enumerate(all_rankings[:3]):
        print(f"\nGraph ID: {result['graph_id']}")
        print("Edge Index:", result["edge_node_index"])
        print("True Ranks:", result["true_ranks"])
        print("Predicted Ranks:", result["pred_ranks"])

if __name__ == "__main__":
    run = main()
    # run.finish() # ako nije sweep
