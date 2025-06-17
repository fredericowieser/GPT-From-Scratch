import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2500
eval_interval = 500
learning_rate = 3e-4
device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")
eval_iters = 200
n_embd = 384
n_head = 6
dropout = 0.2
lambda_comp: float = 1e-3

# Total layers will be the sum of the three stages
n_entry_layers = 1 # Number of always-on entry blocks
n_dynamic_layers = 4 # Number of dynamically gated blocks
n_exit_layers = 1 # Number of always-on exit blocks
n_layer = n_entry_layers + n_dynamic_layers + n_exit_layers
# Sparsity regularization strength
lambda_sparsity = 1e-4

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
] # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        gate_activations = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Unpack the new return values from the model
            logits, loss, sparsity_loss = model(X, Y)
            losses[k] = loss.item()
            # Track average gate activation for monitoring
            if sparsity_loss is not None:
                gate_activations[k] = sparsity_loss.item() / lambda_sparsity

        out[split + "_loss"] = losses.mean()
        if gate_activations.mean() > 0:
            out[split + "_gate_avg"] = gate_activations.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GatedBlock(nn.Module):
    """
    A dynamically gated Transformer block based on Bayesian surprise.
    This block wraps a classic Transformer block and decides whether to
    execute it based on a learned, surprise-based gating mechanism.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # The classic block that performs the expensive computation
        self.classic_block = Block(n_embd, n_head)

        # Learnable parameters for the gating mechanism
        # w > 0 is enforced by exp() to ensure surprise increases gate probability
        self.gate_w = nn.Parameter(torch.randn(1))
        self.gate_b = nn.Parameter(torch.randn(1))

        # Learnable log variance for the posterior Q(h'|h)
        self.log_var_q = nn.Parameter(torch.zeros(1))

    def forward(self, x, prev_gate=None):
        """
        Forward pass with conditional execution.
        x: input hidden state (B, T, C)
        prev_gate: gate values from the previous GatedBlock for hard hierarchy
        """
        # Get the potential next state by running the expensive block:
        # This is mu_q in the surprise calculation
        mu_q = self.classic_block(x)

        # Calculate Bayesian Surprise (KL divergence):
        # We detach mu_q for the surprise calculation so that gradients don't
        # flow back to the classic_block through the gating mechanism itself.
        # The block's parameters should only be updated when the gate is open.
        mu_q_detached = mu_q.detach()
        # KL( Q || P_static ) = ||mu_q - h||^2 / (2*sigma^2) - 0.5*log(sigma^2)
        # We drop the constant -0.5 as it doesn't affect the gate's behavior
        l2_sq = torch.sum((mu_q_detached - x.detach()) ** 2, dim=-1)
        sigma_sq = torch.exp(self.log_var_q)
        surprise = (l2_sq / (2 * sigma_sq)) - 0.5 * self.log_var_q

        # Compute the gate probability and sample a binary gate
        pi = torch.sigmoid(torch.exp(self.gate_w) * surprise - self.gate_b)

        # Use Gumbel-Softmax trick for differentiable sampling (straight-through)
        g = -torch.log(-torch.log(torch.rand_like(pi)))
        gate = (torch.log(pi) + g > torch.log(1 - pi)).float()

        # Enforce Hard Hierarchy
        if prev_gate is not None:
            # Gate can only be 1 if the previous gate was also 1
            gate = gate * prev_gate

        # Conditional update: mix between original x and transformed mu_q
        # The gate is (B, T), needs to be (B, T, 1) for broadcasting
        gate_unsqueezed = gate.unsqueeze(-1)
        h_out = gate_unsqueezed * mu_q + (1 - gate_unsqueezed) * x

        return h_out, gate


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # We use a ModuleList to hold the different types of blocks
        self.blocks = nn.ModuleList()
        # Fixed Entry Blocks
        for _ in range(n_entry_layers):
            self.blocks.append(Block(n_embd, n_head=n_head))
        # Dynamic Core Blocks
        for _ in range(n_dynamic_layers):
            self.blocks.append(GatedBlock(n_embd, n_head=n_head))
        # Fixed Exit Blocks
        for _ in range(n_exit_layers):
            self.blocks.append(Block(n_embd, n_head=n_head))

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = tok_emb + pos_emb

        all_gates = []
        prev_gate = None
        for block in self.blocks:
            if isinstance(block, GatedBlock):
                x, gate = block(x, prev_gate)
                all_gates.append(gate)
                prev_gate = gate # For hard hierarchy
            else: # It's a standard, always-on block
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
            sparsity_loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            task_loss = F.cross_entropy(logits_flat, targets_flat)

            # SPARSITY REGULARIZATION LOSS
            sparsity_loss = 0.0
            if all_gates:
                # We want to encourage gates to be zero
                # The loss is the mean activation of all gates in the batch
                mean_gate_activation = torch.stack(all_gates).mean()
                sparsity_loss = lambda_sparsity * mean_gate_activation

            loss = task_loss + sparsity_loss

        # Return sparsity_loss for monitoring purposes
        return logits, loss, sparsity_loss if "sparsity_loss" in locals() else None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Note: we ignore the loss and sparsity terms during generation
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
print(f"Model has {sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters)):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        gate_info = (
            f", val gate avg {losses['val_gate_avg']:.4f}"
            if "val_gate_avg" in losses
            else ""
        )
        print(
            f"step {iter}: train loss {losses['train_loss']:.4f}, val loss {losses['val_loss']:.4f}{gate_info}"
        )

    xb, yb = get_batch("train")
    logits, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))