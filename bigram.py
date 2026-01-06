import torch
import torch.nn as nn
from torch.nn import functional as F

# .Hyperparameters
BATCH_SIZE = 15     # Number of independent sequences to process simultaneously
BLOCK_SIZE = 60  # Maximum context length for predictions
MAX_ITERATIONS = 9999
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERATIONS = 200
N_EMBED = 384       
N_HEAD = 6       
N_LAYER = 6        
DROPOUT = 0.2       

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)


stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# Split data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loding 
def get_batch(split):
    """Generate a small batch of data of inputs x and targets y."""
    data_source = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in random_indices])
    y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in random_indices])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Averages the loss over multiple batches for both train and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERATIONS)
        for k in range(EVAL_ITERATIONS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class AttentionHead(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        attention_weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        attention_weights = attention_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        v = self.value(x) # (B, T, head_size)
        out = attention_weights @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForwardNetwork(nn.Module):
    """A simple linear layer followed by a non-linearity."""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """A single Transformer block: communication followed by computation."""
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForwardNetwork(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
       
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class SimpleGPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(N_EMBED, num_heads=N_HEAD) for _ in range(N_LAYER)])
        self.final_layer_norm = nn.LayerNorm(N_EMBED)
        self.language_model_head = nn.Linear(N_EMBED, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context, targets=None):
        B, T = context.shape
        
       
        token_embeddings = self.token_embedding_table(context) 
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE)) 
        x = token_embeddings + positional_embeddings 
        x = self.transformer_blocks(x) 
        x = self.final_layer_norm(x) 
        logits = self.language_model_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        for _ in range(max_new_tokens):
            context_cropped = context[:, -BLOCK_SIZE:]
            logits, loss = self(context_cropped)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1) 
        return context

if __name__ == "__main__":
    model = SimpleGPTModel()
    m = model.to(DEVICE)
    
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")
    for iter_num in range(MAX_ITERATIONS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERATIONS - 1:
            losses = estimate_loss()
            print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete.")

    print("\n--- Generated Text ---")
    start_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_output = decode(m.generate(start_context, max_new_tokens=500)[0].tolist())
    print(generated_output)
