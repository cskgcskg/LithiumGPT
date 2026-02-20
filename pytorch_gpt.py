import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import urllib.request
import ssl

# Ignore Mac SSL certificate checks
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. HYPERPARAMETERS (The M4 "Creative" Settings) ---
batch_size = 64
gradient_accumulation_steps = 1
block_size = 256
max_iters = 6000
learning_rate = 5e-4
n_embd = 384
n_head = 6
n_layer = 8
dropout = 0.2  # <--- NEW: 20% Dropout to prevent exact memorization

# Automatically use Mac GPU (MPS)
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware Acceleration: Running on {device.upper()}")

# --- 2. DATA LOADING & TOKENIZER ---
# <--- THE FIX: Hardcoded path to the exact file on your Desktop --->
file_path = "/Users/michalkowalski/Desktop/Agents/karpathy : microgpt.py/input.txt"

if not os.path.exists(file_path):
    print(f"\nERROR: Could not find {file_path}")
    print("Please make sure your Wall Street book is saved exactly there!")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load directly to GPU
data = torch.tensor(encode(text), dtype=torch.long, device=device)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# --- 3. THE GPT MODEL DEFINITION ---

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout) # <--- DROPOUT ADDED

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # <--- DROPOUT ADDED TO FLASH ATTENTION --->
        drop_rate = dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_rate)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout) # <--- DROPOUT ADDED
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 4. THE TRAINING LOOP ---
print(f"Training started with {sum(p.numel() for p in model.parameters())} parameters...")
optimizer.zero_grad(set_to_none=True)
ptdtype = torch.bfloat16 if device in ['mps', 'cuda'] else torch.float32

for iter in range(max_iters):
    xb, yb = get_batch()

    with torch.autocast(device_type='mps', dtype=ptdtype):
        logits, loss = model(xb, yb)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if iter % 100 == 0 or iter == max_iters - 1:
        print(f"step {iter:4d} / {max_iters:4d} | loss {loss.item():.4f}")

# --- 5. CHAT / INFERENCE MODE ---
print("\n--- Model Trained! ---")

# <--- NEW: Tells the model to stop dropping out neurons for the Exam! --->
model.eval()

print("I am ready to speak like a Wall Street Trader. (Type 'quit' to exit)")

while True:
    user_prompt = input("\nYou: ")
    if user_prompt.lower() == 'quit':
        break
        
    if len(user_prompt.strip()) == 0:
        print("Trader AI: I cannot read an empty page. Please type some words!")
        continue
    
    try:
        context = torch.tensor((encode(user_prompt)), dtype=torch.long, device=device).unsqueeze(0)
    except KeyError:
        print("Error: You used a character that isn't in the training data!")
        continue

    print("Trader AI: ", end="")
    output_tensor = model.generate(context, max_new_tokens=400)
    response = decode(output_tensor[0].tolist())
    print(response[len(user_prompt):])