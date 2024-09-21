#Imports
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embedding = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Datasets and Dataloaders
# TODO load and generate paper dataset


def main():

    sys.exit(0)

if __name__ == "__main__":
    main()