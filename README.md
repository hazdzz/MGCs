# Magnetic Graph Convolutional Networks
[![issues](https://img.shields.io/github/issues/hazdzz/MGCs)](https://github.com/hazdzz/MGCs/issues)
[![forks](https://img.shields.io/github/forks/hazdzz/MGCs)](https://github.com/hazdzz/MGCs/network/members)
[![stars](https://img.shields.io/github/stars/hazdzz/MGCs)](https://github.com/hazdzz/MGCs/stargazers)
[![License](https://img.shields.io/github/license/hazdzz/MGCs)](./LICENSE)

## About
The official PyTorch implementation for the paper *sMGC: Complex-Valued Graph Convolutional Networks via the Magnetic Laplacian for Directed Graphs*.

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt
```

## Results
### Node classification accuracy in Citation networks (%) (random seed = 10, 100, or 1000)
| Model | CoRA | CiteSeer | PubMed |
| :----: | :----: | :----: | :----: |
| GAT | 82.35 ± 0.35 | 70.65 ± 0.75 | 77.45 ± 0.45 |
| sMGC | 82.70 ± 0.00 | **73.30 ± 0.00** | 79.90 ± 0.10 |
| MGC | **82.50 ± 1.00** | 71.25 ± 0.95 | **79.70 ± 0.40** |

### Reproduce experiment results
#### sMGC
CoRA:
```console
python3 main_smgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/cora.ini' --alpha=0.03 --t=8.05 --K=38
```

CiteSeer:
```console
python3 main_smgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/citeseer.ini' --alpha=0.01 --t=5.16 --K=40
```

PubMed:
```console
python3 main_smgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/pubmed.ini' --alpha=0.01 --t=5.95 --K=25
```

#### MGC
CoRA:
```console
python3 main_mgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/cora.ini' --alpha=0.08 --t=5.85 --K=10 --droprate=0.4
```

CiteSeer:
```console
python3 main_mgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/citeseer.ini' --alpha=0.01 --t=25.95 --K=35 --droprate=0.3
```

PubMed:
```console
python3 main_mgc.py --mode='test' --seed=100 --dataset_config_path='./config/data/pubmed.ini' --alpha=0.03 --t=15.95 --K=20 --droprate=0.5
```