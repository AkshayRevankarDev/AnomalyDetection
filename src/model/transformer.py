import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LatentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.tok_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.pos_emb = nn.Parameter(torch.zeros(1, config['block_size'], config['n_embd']))
        self.drop = nn.Dropout(config['embd_pdrop'])
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        self.block_size = config['block_size']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."
        
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.forward(idx_cond)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
