"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py
"""
import math
import sys
sys.path.append("./")
from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from decifer.scripts.tokenizer import (
    Tokenizer,
)

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.Tensor(input_dim, rank))
        self.B = nn.Parameter(torch.Tensor(rank, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        return x @ self.A @ self.B

@dataclass
class DeciferConfig:
    block_size: int = 1024
    vocab_size: int = 372
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    lora_rank: int = 4
    use_lora: bool = False
    lora_mlp: bool = False
    lora_proj: bool = False
    condition_with_emb: bool = False

class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        """
        Initialize the LayerNorm module.

        :param ndim: dimensionality of the input tensor
        :param bias: whether to add a learnable bias to the output
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        # LoRA
        if config.use_lora:
            self.lora_attn = LoRALayer(config.n_embd, 2 * config.n_embd, rank=config.lora_rank)
            if config.lora_proj:
                self.lora_proj = LoRALayer(config.n_embd, config.n_embd, rank=config.lora_rank)

        # Masked attention
        self.masked_bias = torch.tensor(-1e4)


    def forward(self, x: Tensor, attention_bias: Tensor = None) -> Tensor:
        """
        Applies causal self-attention to the given tensor,
        with a mask to prevent attention to future positions.

        :param x: tensor of shape (batch size, sequence length, embedding dimension)
        :returns: result of applying the causal self-attention operation
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.config.use_lora:
            lora_k, lora_v = self.lora_attn(x).split(self.n_embd, dim=2)

            lora_k = lora_k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            lora_v = lora_v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            k = k + lora_k
            v = v + lora_v

        #causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            if attention_bias is not None:
                # Expand attention_bias to match the number of heads
                attention_bias = attention_bias.unsqueeze(1) # Shape (B, 1, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_bias, dropout_p=self.dropout
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_bias is not None:
                attention_bias = attention_bias.unsqueeze(1) # Shap (B, 1, T, T)
                att = att + attention_bias
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        if self.config.use_lora and self.config.lora_proj:
            y = self.resid_dropout(self.c_proj(y) + self.lora_proj(y))
        else:
            y = self.resid_dropout(self.c_proj(y))
        return y

def gelu(x: Tensor) -> Tensor:
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function, as used in the Google BERT and
    OpenAI GPT models. See: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415

    :param x: the tensor to which the GELU activation function will be applied
    :returns: the result tensor after applying the GELU activation function,
              possessing the same shape as the input tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_lora:
            if config.lora_mlp:
                self.lora_fc = LoRALayer(config.n_embd, 4 * config.n_embd, rank=config.lora_rank)
            if config.lora_proj:
                self.lora_proj = LoRALayer(4 * config.n_embd, config.n_embd, rank=config.lora_rank)

    def forward(self, x: Tensor) -> Tensor:
        if self.config.use_lora and self.config.lora_mlp:
            x = self.c_fc(x) + self.lora_fc(x)
        else:
            x = self.c_fc(x)
        x = gelu(x)
        if self.config.use_lora and self.config.lora_proj:
            x = self.c_proj(x) + self.lora_proj(x)
        else:
            x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Transformer Block module. A Block module includes causal self-attention,
        layer normalization, and MLP, and residual connections.

        :param x: input to the transformer block
        :returns: output of the transformer block, with the same shape as in the input
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decifer(nn.Module):

    def __init__(self, config: DeciferConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            cond_embedding=nn.Linear(config.block_size, config.n_embd) if config.condition_with_emb else None,
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of total parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True, trainable: bool = False) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        if not trainable:
            n_params = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n_params -= self.transformer.wpe.weight.numel()
        else:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, cond_vec=None, targets=None, start_indices_batch=None,):
        device = idx.device
        b, t = idx.size()
        ptdtype = self.transformer.wte.weight.dtype

        #if cond_vec is not None: t=t+1

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        positions = torch.arange(t, device=device).unsqueeze(0).expand(b, t) # Shape (b, t)

        if start_indices_batch is not None:
            # Init start_mask
            start_mask = torch.zeros((b, t), dtype=torch.long, device=device)

            # Collect valid start indeices and batch indices
            start_indices_list = []
            batch_indices_list = []
            for i, start_indices in enumerate(start_indices_batch):
                valid_indices = start_indices[(start_indices >= 0) & (start_indices < t)]
                start_indices_list.append(valid_indices)
                batch_indices_list.append(torch.full_like(valid_indices, i))

            # Concatenate all valid start indices and batch indices
            if start_indices_list:
                start_indices_flat = torch.cat(start_indices_list)
                batch_indices_flat = torch.cat(batch_indices_list)

                # Set start mask
                start_mask[batch_indices_flat, start_indices_flat] = 1
            else:
                # If no valid start indices, default to zeros
                pass

            #start_indices_flat = start_indices_batch.view(-1)  # Shape: (b * max_num_starts)
            #batch_indices = torch.arange(b, device=device).unsqueeze(1).expand(b, start_indices_batch.size(1)).reshape(-1)  # Shape: (b * max_num_starts)

            # Mask out invalid indices (assuming padding with -1)
            #valid_mask = (start_indices_flat >= 0) & (start_indices_flat < t)
            #valid_start_indices = start_indices_flat[valid_mask]
            #valid_batch_indices = batch_indices[valid_mask]

            # Set start_mask
            #start_mask[valid_batch_indices, valid_start_indices] = 1

            # Compute group IDs
            group_ids = torch.cumsum(start_mask, dim=1)  # Shape: (b, t)

            # Compute last start positions
            start_positions = torch.where(start_mask == 1, positions, torch.tensor(0, device=device))
            last_start_positions, _ = torch.cummax(start_positions, dim=1)

            # Compute positions within each group
            positions_in_group = positions - last_start_positions

            # Create attention mask
            attention_mask = (group_ids.unsqueeze(2) == group_ids.unsqueeze(1)) & (positions.unsqueeze(2) >= positions.unsqueeze(1))
            # Convert to additive attention bias
            attention_bias = torch.zeros_like(attention_mask, dtype=ptdtype)
            attention_bias.masked_fill_(~attention_mask, float('-inf'))
            attention_bias = attention_bias.unsqueeze(1)  # Shape: (b, 1, t, t)
        else:
            positions_in_group = positions
            attention_bias = None

        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t+1), pos_idx -1 for conditioning input

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(positions_in_group)  # position embeddings of shape (1, t, n_embd)

        # Conditioning
        if cond_vec is not None:
            cond_emb = self.transformer.cond_embedding(cond_vec).unsqueeze(1) # shape (b, 1, n_embd)
            tok_emb = torch.cat([cond_emd, tok_emb], dim=1)
            pos_emb = torch.cat([torch.zeros_like(cond_emb), pos_emb], dim=1)
            t = t + 1 # Update sequence length

            # Update attention mask
            if attention_bias is not None:
                attention_bias = F.pad(attention_bias, (1,0,1,0), value=0)
                attention_bias[:,:,0,:] = float('-inf') # Prevent cond token from attending to future tokens
                attention_bias[:,:,:,0] = 0 # Allows tokens to attend to cond token

            positions_in_group = F.pad(positions_in_group, (1,0), value=0)
            t = positions_in_group.size(1)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if cond_vec is not None:
            # Remove conditioning again
            x = x[:, 1:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

                if 'lora' in fpn:
                    decay.add(fpn)

        # subtle: "transformer.wte.weight" and "lm_head.weight" are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, keyed by "transformer.wte.weight", below.
        # so let's manually remove "lm_head.weight" from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, disable_pbar=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        tokenizer = Tokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        pad_id = tokenizer.padding_id
        prev_id = None
        generation_pbar = tqdm(total=max_new_tokens, desc='Generating sequence', leave=False, disable=disable_pbar)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
                break
            # as soon as <pad> is hit -> end of cif
            if idx_next.item() == pad_id:
                idx = idx[:,:-1]
                break
            prev_id = idx_next.item()
            generation_pbar.update(1)
        generation_pbar.close()
        
        return idx

    @torch.no_grad()
    def generate_batched_reps(self, idx, max_new_tokens, temperature=1.0, top_k=None, disable_pbar=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (batch_size, seq_len)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        tokenizer = Tokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        pad_id = tokenizer.padding_id
        batch_size = idx.size(0)
        device = idx.device

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_id = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)
        seq_lens = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

        generation_pbar = tqdm(total=max_new_tokens, desc='Generating sequence', leave=False, disable=disable_pbar)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.full((batch_size, 1), fill_value=pad_id, dtype=torch.long, device=device)
            active_mask = ~finished
            if active_mask.any():
                idx_next[active_mask] = torch.multinomial(probs[active_mask], num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            # Update prev_id and check stopping conditions
            idx_next_squeezed = idx_next.squeeze(-1)
            end_condition = ((prev_id == newline_id) & (idx_next_squeezed == newline_id)) | (idx_next_squeezed == pad_id)
            # Record sequence lengths when sequences finish
            just_finished = end_condition & (~finished)
            # Exclude the last token that triggered the end condition
            seq_lens[just_finished] = idx.size(1) - 1
            finished = finished | end_condition
            prev_id = idx_next_squeezed
            generation_pbar.update(1)
            if finished.all():
                break
        generation_pbar.close()
        # For sequences that didn't finish, set seq_lens to idx.size(1)
        seq_lens[seq_lens == -1] = idx.size(1)
        # Truncate sequences to their actual length
        max_seq_len = seq_lens.max().item()
        idx_truncated = torch.full((batch_size, max_seq_len), fill_value=Tokenizer().padding_id, dtype=idx.dtype, device=idx.device)
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            idx_truncated[i, :seq_len] = idx[i, :seq_len]
        return idx_truncated
    
    @torch.no_grad()
    def generate_and_print(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        tokenizer = Tokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        pad_id = tokenizer.padding_id
        prev_id = None
            
        for id in idx[0]:
            token = tokenizer.id_to_token[id.item()]
            for char in token:
                sys.stdout.write(char)
                sys.stdout.flush()

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # as soon as <pad> is hit -> end of cif
            if idx_next.item() == pad_id:
                idx = idx[:,:-1]
                break

            # Decode and print
            token_next = tokenizer.id_to_token[idx_next[0].item()]
            for char in token_next:
                sys.stdout.write(char)
                sys.stdout.flush()

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
                break
                
            prev_id = idx_next.item()

        return idx

    @torch.no_grad()
    def generate_sequences(self, sequences, prompt_lengths, max_new_tokens, temperature=1.0, top_k=None, disable_pbar=False):
        """
        Perform inference one by one for each input sequence, but as fast as possible with pre-allocation and minimization of tensor concatenation.
        """
        tokenizer = Tokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        pad_id = tokenizer.padding_id

        generation_pbar = tqdm(total=len(sequences) * max_new_tokens, desc='Generating sequences', leave=False, disable=disable_pbar)

        # Store the generated outputs for all sequences
        all_generated = []

        # Ensure the model is in evaluation mode for faster inference
        self.eval()

        # Pre-allocate the generated sequence tensor to avoid repeated memory allocations
        for i, prompt in enumerate(sequences):
            # Pre-allocate the max sequence length
            max_length = prompt_lengths[i] + max_new_tokens
            generated_sequence = torch.zeros((max_length,), dtype=torch.long, device=prompt.device)
            generated_sequence[:prompt_lengths[i]] = prompt[:prompt_lengths[i]]
            seq_len = prompt_lengths[i]
            finished = False

            for step in range(max_new_tokens):
                if finished:
                    break

                # Forward pass: only take the valid portion of the sequence (last block_size tokens)
                idx_cond = generated_sequence[max(0, seq_len - self.config.block_size):seq_len].unsqueeze(0)  # (1, seq_len)
                logits, _ = self(idx_cond)

                # Take logits for the last position and scale by temperature
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Compute probabilities and sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(0)

                # Append the new token to the generated sequence
                generated_sequence[seq_len] = next_token
                seq_len += 1

                # Check for stopping conditions: <pad> or double newline
                if next_token.item() == pad_id or (generated_sequence[seq_len - 2].item() == newline_id and next_token.item() == newline_id):
                    finished = True

                generation_pbar.update(1)

            # Add the final generated sequence without padding
            all_generated.append(generated_sequence[:seq_len])

        generation_pbar.close()

        return all_generated




