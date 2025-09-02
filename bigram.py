import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"  # it's not available on my machine
eval_iters = 200

torch.manual_seed(1337)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


