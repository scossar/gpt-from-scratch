# video ~58:00 -->
# start by making some changes to the bigram.py code
# continued in gptv1.py

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
n_embd = 32

torch.manual_seed(1337)

with open("./tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)  # used in the model (no need to pass it to the model constructor, at least in the machine learning world)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output list of integers
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # instead of directly going to the embedding for the logits (nn.Embedding(vocab_size, vocab_size))
        # we're going to add a level of "indirection" (allow for more dims for each logit?)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each position (from 0 to block_size-1 gets its own embedding vector)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # a linear layer to go from token embeddings to logits
        self.lm_head = nn.Linear(n_embd, vocab_size)  # lm_head means language_model_head

    # trying to understand/clarify: "So far we've taken these indices (I'm understanding 'indices' to be referring to idx) and encoded them based on the identity of the tokens inside idx.
    # The next thing that people often do is not just encode the identity of the tokens, but also their position, so we're going to have use a position embedding table"
    # ** explanation: "encode" in this context describes the process of looking up the embedding vectors from the raw indices.
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # (B,T,C) (Batch, Time, Channel)
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) (B,T,n_embd)
        # look up indices (0 -> T-1) from position_embedding_table:
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        # the summation (x = tok_emb + pos_emb) works because of broadcasting
        # all the embeddings are trainable - the get updated during backprop
        x = tok_emb + pos_emb  # (B,T,C), so x holds token identities and token positions
        # at this point, lm_head is a "spurious" layer of indirection - adding it didn't accomplish anything
        logits = self.lm_head(x)  # (B,T,vocab_size)
        # we have the identity of the next character, so can as how we'll we're predicting
        # the next character; the correct dimension (index) of logits should have a high number,
        # all other dimensions should have a low number
        
        if targets is None:
            loss = None
        else:
            # we can't call it like I'm doing below, because PyTorch expects multidimensional logits
            # to have the shape (B,C,T)
            # loss = F.cross_entropy(logits, targets)
            # so reduce the dimensions
            B, T, C = logits.shape  # (B,T,n_embd)
            logits = logits.view(B*T, C)  # 2D
            targets = targets.view(B*T)  # 1D
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution (see notes)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
