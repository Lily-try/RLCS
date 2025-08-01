# RLCS

This repository is for the following article:

Robust Learning-Based Community Search on Noisy Graphs
---

# Datasets

6 datasets are used in this paper:

- Cora, Citeseer: https://github.com/kimiyoung/planetoid/tree/master/data

- Facebook: https://github.com/yzhang1918/kdd2020seal
- Cocs: https://github.com/guaiyoui/TransZero/
- Amazon, DBLP: http://snap.stanford.edu/data/index.html

The preprocessed data used in our experiments can be found in the "data" folder.

---

# Rreproduce major results

## Requirements

- Python version: 3.9.20
- Pytorch version: 2.5.1+cu124
- torch-geometric version: 2.6.1
- deeprobust version: 0.2.11

## How to run

for cora:

```bash
python main.py --dataset cora --attack none
python main.py --dataset cora --attack random_add
```

