# gptv1.py, but using `names.txt` for the training data (needs some tweaking)
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

batch_size = 32
block_size = 4
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # it's not available on my machine
eval_iters = 200
n_embd = 27
n_head = 6
n_layer = 6
dropout = 0.2  # every forward/backward pass 20% of calculations are dropped to 0


with open("../makemore/names.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65 characters
print("chars", chars)
print("vocab_size", vocab_size)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    out = []
    for c in s:
        out.append(stoi[c])
    return out


def decode(character_codes):
    character_chars = []
    for c in character_codes:
        character_chars.append(itos[c])
    return "".join(character_chars)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# 1:19:26
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril: torch.Tensor  # for pyright
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, C = x.shape  # (B, T, C)
        k = self.key(
            x
        )  # (B, T, C); note that C is technically head_size; head_size == n_embd
        q = self.query(x)  # (B, T, C); note that C is tecnhically head_size
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # makes it a decoder block
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


# -----------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # vocab_size and n_embd are global variables
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)  # self-attention head
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model head

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        # note that C is considered to be n_embd.size in this code
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # positional encodings add the concept of space (a token's position) to x
        # attention on its own doesn't deal with position (?)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # apply one head of self attention
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Note: reshaping logits and targets so they're in the shape expected by PyTorch
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # pluck out last element in the T dimension
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(
    device
)  # move the model to device so all calculations happen on GPU if it's available


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # set model's mode to "eval" (not implemented in Bigram model)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set model's mode back to "train" (not implemented in Bigram model)
    return out


# train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(50000):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # break  # for debugging


# sample from the model
# context shape: (B, T) 0 is the newline character; reasonable starting point
context = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # create context on device
sampled = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(sampled)
