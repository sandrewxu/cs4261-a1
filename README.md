# A1: Understanding, Programming and Analysis of GPT Workload
## Andrew Xu

## Part 1: KV Cache Analysis
I examined `starter/gpt_with_kv_mha.py`.

I notice that the following pieces of code are relevant to the KV-cache.

1. Initializing a KV-Cache buffer in MHA initialization
```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
self.ptr_current_pos = 0
```
We store the KV cache as a buffer to keep it with the model.

2. Using the cache in the forward pass
```python
if use_cache:
    if self.cache_k is None:
        self.cache_k, self.cache_v = keys_new, values_new
    else:
        self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
        self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
    keys, values = self.cache_k, self.cache_v
else:
    keys, values = keys_new, values_new
```
I notice that this means only new tokens should be put into the transformer.

3. The forward pass of the model
```python
if use_cache:
    pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
    self.current_pos += seq_len
```
To account for the previous tokens cached, we have to start at the current position instead of 0.

4. Generating text cached
```python
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx

```
Finally, in this function, we see that we only pass the prefix through the Transformer once. Afterwards, each new token is individually passed into the transformer. This means we only calculate Q and K for each token once after the initial pass. Alternatively, in the `else` statement, we have that each token is processed through each generation.

### Example of a KV-Cache
Suppose we are predicting the following sentence: `the lazy fox jumps over the brown dog`, with the following prefix: `the lazy fox jumps`.

Then, the standard LLM will process
`the lazy fox jumps` to decode `over`,
`the lazy fox jumps over` to decode `the`,
`the lazy fox jumps over the` to decode `brown`,
`the lazy fox jumps over the brown` to decode `dog`.

The KV-cached LLM will process (and cache)
`the lazy fox jumps` to decode `over`,
`over` to decode `the`,
`the` to decode `brown`,
`brown` to decode `dog`.

Each time, it just concatenates the cached queries and keys with the new one.

## Part 2: GQA Implementation and Analysis
### 2.1: Implementation
The GQA code based on `starter/gpt_with_kv_gqa.py` is implemented in `gqa.py`. The changes I made were as follows.

1. Initialization
```python
self.heads_per_group = num_heads // num_kv_groups
self.kv_dim = num_kv_groups * self.head_dim

self.W_key = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
self.W_value = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
```
We initialize `self.W_key` and `self.W_value` with output dimension `num_kv_groups * self.head_dim` instead of the traditional `num_heads * self.head_dim` that `self.W_queries` uses, which is the main benefit of GQA.

2. Projection Repeat
```python
# Apply projections        
# 4261/5261: investigate repeat vs repeat_interleave
queries = self.W_query(x)
keys = self.W_key(x)    # shape: b num_tokens num_kv_groups * head_dim
values = self.W_value(x)

queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
keys = keys.view(b, num_tokens, self.num_groups, self.head_dim)
values = values.view(b, num_tokens, self.num_groups, self.head_dim)

# repeat_interleave the head dim heads_per_group times
keys = keys.repeat_interleave(self.heads_per_group, dim=2, output_size=self.num_heads)
values = values.repeat_interleave(self.heads_per_group, dim=2, output_size=self.num_heads)

# transpose head dim to make (b, num_heads, num_tokens, head_dim)
queries = queries.transpose(1, 2)
keys = keys.transpose(1, 2)
values = values.transpose(1, 2)
```
To complete the forward pass, we now need to up-project the final dimension of the keys and values back to `num_heads * self.head_dim`. This can occur after we isolate the number of heads from the head dimension. Then, we transpose to bring the number of heads to dimension 1. In GQA, each embedding in the head/group dimension is repeated by `self.heads_per_group`. Thus, we use `repeat_interleave` on the final dimension.

### 2.2: Memory estimation
The GQA memory estimation code based on `starter/memory_estimator.py` is implemented in `memory_estimator.py`. 

```bash
> python memory_estimator.py --emb_dim 768 --n_heads 48 --n_layers 40 --n_kv_groups 4

==== Config ====
context_length   : 1024
emb_dim          : 768
n_heads          : 48
n_layers         : 40
n_kv_groups      : 4
batch_size       : 1
dtype            : fp16 (2 Bytes/elem)
head_dim         : 16
GQA n_kv_heads   : 12

==== KV-cache totals across all layers ====
MHA total KV cache  : 0.13 GB
GQA total KV cache  : 0.03 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```
We see that in KV memory, MHA takes `n_kv_groups` times more memory than GQA.

### 2.3: FLOPs estimation
# TO-DO


### 2.4: Additional implementations (conceptual)
Using **MultiHeadAttentionCombinedQKV**, our implementation does not change very much. In our combined matrix multiply, we would initialize 
```python
self.qkv = nn.Linear(d_in, d_out + 2 * self.kv_dim, bias=qkv_bias)
qkv = self.qkv(x)
```
instead of 
```python
self.W_key = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
self.W_value = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
queries = self.W_query(x)
keys = self.W_key(x)
values = self.W_value(x)
```
It is worthy to note that because the head dimension is no longer symmetric, we will have to use `torch.split` with the desired dimensions instead of `torch.unbind` after transposing the view of the tensor.

Using **MHAEinsum**, our changes can make the code simpler. Instead of having to reshape our KV groups to the Q heads dimension, we can batch the query heads according to the number of groups, then use `torch.einsum` to implicitly broadcast across the group dimension and avoid `repeat_interleave` and `transpose`. Specifically, we would have something like:
```python
self.W_key = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
self.W_value = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
queries = self.W_query(x)   # batch_size    num_tokens  num_heads * head_dim
keys = self.W_key(x)        # batch_size    num_tokens  num_kv_groups * head_dim
values = self.W_value(x)    # batch_size    num_tokens  num_kv_groups * head_dim

queries.view(b, num_tokens, self.num_groups, self.heads_per_group, self.head_dim)
attn_scores = torch.einsum("b n g h d, b m g d -> b g h n m", queries, keys)
```

## Part 3: MLA Implementation and Analysis
### 3.1: Implementation
The MLA code based on `starter/gpt_with_kv_mla.py` is implemented in `mla.py`. The changes I made were as follows.
1. Initialization
```python
self.W_dkv = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)
self.W_uk = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
self.W_uv = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
```
The difference here is that instead of `W_keys` and `W_values`, we down-project into `W_dkv` to the latent dimension to store the kv-cache before up-projecting again.

2. Caching
```python
queries = self.W_query(x)
c_kv_new = self.W_dkv(x)    # dim: batch_size seq_len latent_dim

if use_cache:
    if self.cache_c_kv is None:
        self.cache_c_kv = c_kv_new
    else:
        self.cache_c_kv = torch.cat([self.cache_c_kv, c_kv_new], dim=1)
    c_kv = self.cache_c_kv
else:
    c_kv = c_kv_new
```
We calculate the queries as normal, but we downproject and cache in the latent dimension to save memory before up-projecting again.

3. Up-projecting and attention score calculation
```python
keys = self.W_uk(c_kv)
values = self.W_uv(c_kv)

queries = self._reshape_to_heads(queries, self.num_heads, self.head_dim)
keys = self._reshape_to_heads(keys, self.num_heads, self.head_dim)
values = self._reshape_to_heads(values, self.num_heads, self.head_dim)

attn_scores = queries @ keys.transpose(2, 3)
```
After projecting the cached KV back to the intended dimension, the calculation is standard attention again.

4. Mask
```python
num_tokens_Q = queries.shape[-2]
num_tokens_K = keys.shape[-2]
device = queries.device
if use_cache:
    q_positions = torch.arange(
        self.ptr_current_pos,
        self.ptr_current_pos + num_tokens_Q,
        device=device,
        dtype=torch.long,
    )
    self.ptr_current_pos += num_tokens_Q
else:
    q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
    self.ptr_current_pos = 0
k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
```
The masking code we copy from `starter/gpt_with_kv_mha.py`. This is to start the query sequence at a different position if the keys were previously cached.

### 3.2: Memory estimation
The MLA memory estimation code based on `starter/memory_estimator.py` is implemented in `memory_estimator.py`. 
```bash
❯ uv run memory_estimator.py --emb_dim 768 --n_heads 48 --n_layers 40 --n_kv_groups 4 --latent_dim 96
==== Config ====
context_length   : 1024
emb_dim          : 768
n_heads          : 48
n_layers         : 40
n_kv_groups      : 4
latent_dim       : 96
batch_size       : 1
dtype            : fp16 (2 Bytes/elem)
head_dim         : 16
GQA n_kv_heads   : 12
MLA latent dim   : 96

==== KV-cache totals across all layers ====
MHA total KV cache  : 0.13 GB
GQA total KV cache  : 0.03 GB
MLA total KV cache  : 0.01 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 16.00x
Savings (MLA vs MHA): 93.75%
```

We see here that the savings of MLA is `emb_dim * 2 / latent_dim`, which is 16x. This comes from the smaller, combined KV cache storage in the latent dimension.

### 3.3: FLOPs estimation
# TO-DO

## Part 4: MoE Implementation and Analysis
### 4.1: Implementation
The MoE code based on `starter/gpt_with_kv_moe.py` is implemented in `moe.py`. The changes I made were as follows.
1. The `forward` function of the `MoEFeedForward` class
```python
topk_probs_flat = topk_probs.reshape(batch * seq_len, -1)   # (b * seq_len, num_experts_per_tok)
topk_indices_flat = topk_indices.reshape(batch * seq_len, -1)   # (b * seq_len, num_experts_per_tok)

# iterate by expert
for i in range(self.num_experts):
    token_idx, slot_idx = torch.where(topk_indices_flat == i)   # token, position
    # expert takes in emb_dim, spits out emb_dim -- use topk_indices to index x_flat
    expert_x = x_flat[token_idx]    # (num_routed, emb_dim)
    expert_out = self.fc3[i](torch.nn.functional.silu(self.fc1[i](expert_x)) * self.fc2[i](expert_x))
    weights = topk_probs_flat[token_idx, slot_idx].unsqueeze(-1)  # (num_routed, 1)
    out_flat[token_idx] += weights * expert_out    # (num_routed, emb_dim)
```
We first flatten the topk_probs and topk_indices similar to x and out, so all tokens can get processed regardless of batch. Then, for each expert, we discover the tokens that call it as well as what position. We run the appropriate FFN on all those tokens, use the slot_idx to determine the weight of that expert on each token, and multiply them and add to the output, as desired.

### 4.2: Memory estimation
The MoE memory estimation code based on `starter/memory_estimator.py` is implemented in `memory_estimator.py`. As the MoE modification happens after the attention layer, in the feed-forward layer, there is no memory savings with regard to the KV cache. It is the same as MHA, or whichever attention implemenetation it uses.

### 4.3: FLOPs estimation
# TO-DO

## Part 5: Backward Computing Analysis
# TO-DO