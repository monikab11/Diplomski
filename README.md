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
