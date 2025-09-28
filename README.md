# Diplomski
Identifying critical nodes and links in multi-agent systems with graph neural networks

## Project structure

```
.
├── baseline_model
│   ├── best_models
│   │   ├── best_model_1.pt
│   │   ├── best_model_82.pt
│   │   └── best_model.pt
│   ├── check_results.py
│   ├── config
│   │   └── experimenting_sweep.yaml
│   ├── criticality_dataset.py
│   ├── gnn_model.py
│   ├── requierments.txt
│   ├── run_sweep.sh
│   └── train.py
├── dataset_generator
│   ├── analyze_features.ipynb
│   ├── criticality_dataset_network_criticality.pt
│   ├── distributions.ipynb
│   ├── generate_full_ds.py
│   ├── graph6
│   │   ├── graphs_03.txt
│   │   ├── graphs_04.txt
│   │   ├── graphs_05.txt
│   │   ├── graphs_06.txt
│   │   ├── graphs_07.txt
│   │   ├── graphs_08.txt
│   │   └── graphs_09.txt
│   ├── graphs_visualization.ipynb
│   └── print_table.py
└── README.md
```
*baseline model*
- `best_models` - Folder where the trained models are saved.
- `check_results.py` - Prints true and predicted values for a selected graph index, or the first 30 graphs (by index) from the file in all_rankings folder generated after running `train.py`.
- `config/experimenting_sweep.yaml` - file specifying hyperparameters for the W&B sweep
- `criticality_dataset.py` - custom PyTorch Geometric dataset loader that converts stored graph data (with graph6 encoding, features, and criticality scores) into torch_geometric.data.Data objects for training GNN models
- `gnn_model.py` - implementation of GNN architectures and scoring modules for node and link criticality prediction
- `requierments.txt` - Python dependencies required for training and running the project
- `run_sweep.sh` - bash script for launching a Weights & Biases (W&B) hyperparameter sweep and running the corresponding agent for automated experiments
- `train.py` - script for training, validating, and evaluating the GNN model with configurable hyperparameters

*dataset_generator*
- `analyze_features.ipynb` - displays correlations between features.
- `criticality_dataset_network_criticality.pt` - dataset example generated using `generate_full_ds.py`.
- `distributions.ipynb` - visualises the distribution of the desired dataset.
- `generate_full_ds.py` - generates a dataset for a specified metric.
- `graph6` - folder containing files with graphs in the Graph6 format.
- `graphs_visualization.ipynb` - visualizes graphs with 3 to 6 nodes.
- `print_table.py` - prints the mapping between sequential indices and dataset indices, along with a single dataset entry for a given graph.

## Dataset generation
To generate the dataset, first clone the repository:
```
git clone https://github.com/mkrizmancic/my_graphs_dataset.git
```
Next, in `my_graphs_dataset/dataset_loader.py`, add the following lines at the top:
```
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(current_dir)
```
Then, inside the `my_graphs_dataset` folder, create a folder named `data` and copy `graph6` folder from the `dataset_generator` into it.

Finally, run the dataset generation script:
```
python3 generate_full_ds.py $METRIC$
```
Replace `$METRIC$` with one of the following values:
- `algebraic_connectivity` 
- `effective_graph_resistance`
- `node_connectivity`
- `edge_connectivity`
- `network_criticality`
- `nc3` - modified network criticality.

## Usage
To run the code, install required packages by running 
```
pip install -r requierments.txt
```
and simply run
```
python3 train.py
```

To enable Weights & Biases, set WANDB_ENABLED flag to True.
