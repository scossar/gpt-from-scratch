# making gptv1.py again
# implementing a single attention head (video: 1:02-->)
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # it's not available on my machine
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2  # every forward/backward pass 20% of calculations are dropped to 0


with open("./tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65 characters

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
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # vocab_size and n_embd are global variables
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        # note that C is considered to be n_embd.size in this code
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # pluck out last element in the T dimension
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


batch_size = 32  # how many independent sequences are processed in parallel
block_size = 8  # the *maximum* context length for predictions

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

# single attention head (ignore above code)
# #########################################

# the basic math concept
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # (batch, time, channels)
x = torch.randn(B, T, C)
tril = torch.tril(torch.ones(T, T))
# print(tril.shape)
# print(tril)
# torch.Size([8, 8])
# tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 1.]])

# initialize the affinity between tokens to be 0
wei = torch.zeros((T, T))  # (8, 8)
wei = wei.masked_fill(tril == 0, float("-inf"))
# tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])
wei = F.softmax(wei, dim=-1)  # wei is the affinitiy between tokens
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
#         [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
#         [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
#         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

# a simple average of all the past tokens and the current token
out = wei @ x  # (8, 8) @ (4, 8, 32) --> (4, 8, 32)
# print(out[0][0][0] == x[0][0][0])  # True

# actual implementation
# #####################

# a (4, 8) arrangement of tokens; the information for each token has 32 dimensions
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# wei should not be all uniform, different tokens will find other tokens more or less interesting
# the level of interest, or affinitiy that one token has for another is data dependent (depends on the value of x)
# this is the problem that self attention solves (data dependent affinity/interest)
# in the math example above, the affinity between tokesn was initialized to 0

# self attention:
# every single node/token emits two vectors: a query and a key
# affinities are generated/calculated by doing a dot product between the keys and the queries
# the query dot products with all the keys of the other tokens; the dot product becomes the affinitiy, wei

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
# forward the Linear modules on x
k = key(x)  # (4, 8, 16) (B, T, head_size)
# key(x):
# x @ key.weights; (4, 8, 32) @ (32, 16) --> (4, 8, 16)
# remember that batch matrix multiplication can be thought of as (8, 32) @ (32, 16) --> (8, 16) happening batch number (4) times
# for each batch B, for each token T, calculate the dot product of C
q = query(x)  # (4, 8, 16) (B, T, head_size)


# wei: the affinities between queries and keys
# for every row of B, wei will be a (T, T) matrix giving us the affinities between tokens:
# every T of wei will have a different value in each batch, because the tokens for x are different for each batch
# so **wei is data dependent**
wei = q @ k.transpose(
    -2, -1
)  # (4, 8, 16) @ (4, 16, 8); (B, T, head_size) @ (B, head_size, T) --> (B, T, T)

# now convert wei so that it's a decoder
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))

v = value(x)  # the elements that are aggregated with wei

out = wei @ v  # (4, 8, 8) @ (4, 8, 16) --> (4, 8, 16)
