import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 15000
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
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

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
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
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
    
class DynamicBlock(nn.Module):
    """
    A Dynamic Transformer block that can conditionally skip its transformation,
    acting as an identity function based on a surprise metric.
    """

    def __init__(self, n_embd, n_head, ffn_prior, k_surprise_threshold=2.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # New component: The prior model for predicting state change
        self.ffn_prior = ffn_prior

        # Hyperparameters for the surprise-based gating
        self.k_surprise_threshold = k_surprise_threshold
        # We use buffers for state that should be part of the model's state_dict
        # but not considered a model parameter.
        self.register_buffer('moving_avg_surprise', torch.tensor(0.0))
        self.surprise_momentum = 0.99 # Momentum for the moving average
        self.current_step = 0

    def forward(self, x, past_states=None):
        """
        x: input for the current timestep (B, T, C)
        past_states: a tuple (h_full_past, h_mha_past) from the previous timestep.
                     Used only during generation. For training, we approximate this
                     by looking at the t-1 token in the sequence.
        """
        B, T, C = x.shape

        # --- Standard Transformer Block Computation (to get the posterior) ---
        # Note: We must always compute this to make the decision.
        h_mha = x + self.sa(self.ln1(x))
        q_posterior = h_mha + self.ffwd(self.ln2(h_mha))

        # --- Surprise Calculation and Gating Decision ---
        # 1. Define the Static Prior: The hypothesis that the state is just the input
        p_static = x

        # 2. Define the Change Prior: The prediction of the state based on the past
        if self.training:
            # During training, approximate past state with the t-1 token in the sequence
            # Prepend zeros for the first token which has no past
            h_mha_past = torch.cat((torch.zeros(B, 1, C, device=x.device), h_mha[:, :-1, :]), dim=1)
        else: # During generation (T=1)
            if past_states is None or past_states[1] is None:
                # No past state available for the very first token generated
                h_mha_past = torch.zeros_like(h_mha)
            else:
                h_mha_past = past_states[1] # Use the cached MHA state
        
        p_change = self.ffn_prior(h_mha_past)

        # 3. Compute Surprise Metrics (using MSE as a proxy for KL-divergence)
        # We compute loss per token in the sequence, so we don't reduce the batch/time dims yet
        # d_st = F.mse_loss(q_posterior.detach(), p_static.detach(), reduction='none').mean(dim=-1) # (B, T)
        # d_ch = F.mse_loss(q_posterior.detach(), p_change.detach(), reduction='none').mean(dim=-1) # (B, T)
        # In DynamicBlock.forward, modify the surprise calculation:
        # We still detach the posterior and static prior to prevent the main block
        # from "gaming" the system. But we DO NOT detach the change prior.
        d_st = F.mse_loss(q_posterior.detach(), p_static.detach(), reduction='none').mean(dim=-1)
        # Let gradients flow into p_change to train the ffn_prior!
        d_ch = F.mse_loss(q_posterior.detach(), p_change, reduction='none').mean(dim=-1)

        # 4. Update moving average for the "unexpected event" criterion (Criterion U)
        # This should only happen during training.
        if self.training:
            with torch.no_grad():
                self.moving_avg_surprise = self.surprise_momentum * self.moving_avg_surprise + \
                                           (1 - self.surprise_momentum) * d_st.mean()

        # 5. Make the decision: Has an event occurred?
        # Criterion E (Expected): posterior is closer to change prior than static prior
        # expected_event = d_st > d_ch
        # # Criterion U (Unexpected): static surprise is abnormally high
        # unexpected_event = d_st > self.k_surprise_threshold * self.moving_avg_surprise
        
        # event_mask = (expected_event | unexpected_event).float().unsqueeze(-1) # (B, T, 1)

        gating_warmup_steps = 1000 # Hyperparameter: for how many steps to force updates
    
        if self.training and self.current_step < gating_warmup_steps:
            # During warmup, always update.
            event_mask = torch.ones(B, T, 1, device=x.device, dtype=torch.float)
        else:
            # ... (the original surprise calculation and event_mask creation) ...
            # d_st = F.mse_loss(q_posterior.detach(), p_static.detach(), reduction='none').mean(dim=-1)
            # d_ch = F.mse_loss(q_posterior.detach(), p_change, reduction='none').mean(dim=-1)
            expected_event = d_st > d_ch
            unexpected_event = d_st > self.k_surprise_threshold * self.moving_avg_surprise
            event_mask = (expected_event | unexpected_event).float().unsqueeze(-1)

        # --- Conditional Update ---
        # If event_mask is 1, use the new state (q_posterior).
        # If event_mask is 0, use the input (x), making the block an identity function.
        h_final = event_mask * q_posterior + (1 - event_mask) * x

        # We need to return h_mha for the next step's prior calculation during generation
        return h_final, h_mha, event_mask.squeeze(-1)

class DynamicGPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # # Use nn.ModuleList instead of nn.Sequential to allow for manual iteration
        # self.blocks = nn.ModuleList([DynamicBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Create one prior network to be shared
        shared_ffn_prior = FeedFoward(n_embd)
        
        # Pass the shared network to each block
        self.blocks = nn.ModuleList([
            DynamicBlock(n_embd, n_head=n_head, ffn_prior=shared_ffn_prior) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # x = tok_emb + pos_emb
        
        # # Manually iterate through the blocks
        # # For the training forward pass, we don't need to manage past states across calls,
        # # as the block's internal logic approximates it by looking at t-1.
        # for block in self.blocks:
        #     x, _ = block(x) # We can ignore the returned h_mha during training
        x = tok_emb + pos_emb
    
        activity_loss = 0.0
        activity_reg_strength = 0.01 # Hyperparameter
        target_activity = 0.5 # Hyperparameter

        for block in self.blocks:
            x, _, event_mask = block(x) # event_mask is (B, T)
            # Calculate the mean activity for this block in this batch
            mean_activity = event_mask.mean()
            # Add a loss that penalizes deviation from the target
            activity_loss += (mean_activity - target_activity).pow(2)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            main_loss = F.cross_entropy(logits, targets)
            # Add the auxiliary loss to the main loss
            loss = main_loss + activity_reg_strength * activity_loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Initialize the cache for past states for each layer.
        # Each element is a tuple (h_full, h_mha)
        past_states = [(None, None)] * n_layer

        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # --- Single Token Forward Pass ---
            B, T = idx_cond.shape
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
            
            # Get the embedding for just the newest token
            tok_emb = self.token_embedding_table(idx_cond[:, -1:]) # (B, 1, C)
            x = tok_emb + pos_emb[-1:, :] # Add the positional embedding for the current step

            # New cache for the states we are about to compute
            new_past_states = []
            
            # Manually iterate through blocks, passing the cached state
            for i, block in enumerate(self.blocks):
                x, h_mha, _ = block(x, past_states[i])
                # Detach states from the graph to prevent memory leaks during generation
                new_past_states.append((x.detach(), h_mha.detach()))
            
            # Update the cache for the next iteration
            past_states = new_past_states

            x = self.ln_f(x)
            logits = self.lm_head(x) # (B, 1, vocab_size)
            # --- End Single Token Forward Pass ---

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = DynamicGPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    for block in m.blocks:
        block.current_step = iter

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # <-- Add this line
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))