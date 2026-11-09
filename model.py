from flax import nnx
import jax
import jax.numpy as jnp

# ==========================================
# 1. HARDWARE-OPTIMIZED ATTENTION BLOCK
# ==========================================
class Attention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        initializer = jax.nn.initializers.glorot_uniform()
        key = rngs.params()
        
        # CRITICAL FIX: Keeping weight structures strictly 2D (dim, dim)
        # This maps perfectly onto native hardware Tensor Cores / Matrix Units (GEMM).
        self.w_q = nnx.Param(initializer(key, (dim, dim), jnp.bfloat16))
        self.w_k = nnx.Param(initializer(key, (dim, dim), jnp.bfloat16))
        self.w_v = nnx.Param(initializer(key, (dim, dim), jnp.bfloat16))
        self.w_o = nnx.Param(initializer(key, (dim, dim), jnp.bfloat16))
        self.norm = nnx.LayerNorm(num_features=self.dim, rngs=rngs)

    def __call__(self, x, mask):
        B, S, D = x.shape
        
        # 1. Flatten 3D input to 2D for high-throughput execution
        x_flat = x.reshape(-1, D)  # Shape: (B * S, D)
        
        # 2. Linear projections using standard high-speed 2D dot products
        q = jnp.dot(x_flat, self.w_q.value)  # (B * S, D)
        k = jnp.dot(x_flat, self.w_k.value)  # (B * S, D)
        v = jnp.dot(x_flat, self.w_v.value)  # (B * S, D)
        
        # 3. Reshape and Transpose to hardware-optimized layout: (Batch, Heads, Seq, Head_Dim)
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 4. Standard Batched MatMul for attention scores
        # (B, N, S, H) @ (B, N, H, T) -> (B, N, S, T)
        a = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        a = a / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            a = jnp.where(mask, a, -1e9)
        a = jax.nn.softmax(a, axis=-1)
        
        # 5. Compute contextualized values
        # (B, N, S, T) @ (B, N, T, H) -> (B, N, S, H)
        s = jnp.matmul(a, v)
        
        # 6. Prepare for output projection: Transpose and flatten back to 2D
        s = s.transpose(0, 2, 1, 3).reshape(-1, D)  # (B * S, D)
        
        # 7. Final 2D Output GEMM
        out_flat = jnp.dot(s, self.w_o.value)  # (B * S, D)
        out = out_flat.reshape(B, S, D)        # Restored to 3D sequence
        
        return x + self.norm(out)

# ==========================================
# 2. RAGGED DOT MIXTURE OF EXPERTS FFN
# ==========================================
class MoE_FFN(nnx.Module):
    def __init__(self, dim: int, ffn_dim: int, num_experts: int, k: int, rngs: nnx.Rngs):
        """
        Mixture of Experts Feed-Forward Network using jax.lax.ragged_dot.
        
        Args:
            dim: Hidden dimension size of the model (d_model).
            ffn_dim: Intermediate hidden size of the expert networks.
            num_experts: Total pool size of available experts.
            k: Number of experts activated per individual token.
            rngs: Flax NNX random number generator stream.
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        initializer = jax.nn.initializers.glorot_uniform()
        
        # Router parameter matrix (Strictly 2D for high GEMM alignment)
        self.w_gate = nnx.Param(initializer(rngs.params(), (dim, num_experts), jnp.bfloat16))
        
        # Batched expert weights for ragged execution: (E, D, F) and (E, F, D)
        self.w_in = nnx.Param(initializer(rngs.params(), (num_experts, dim, ffn_dim), jnp.bfloat16))
        self.w_out = nnx.Param(initializer(rngs.params(), (num_experts, ffn_dim, dim), jnp.bfloat16))
        self.norm = nnx.LayerNorm(num_features=dim, rngs=rngs)

    def __call__(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # Flatten batch and sequence: (M, D) where M = B * S
        
        # 1. Routing Tensors Setup
        gate_logits = jnp.dot(x_flat, self.w_gate.value)  # Shape: (M, num_experts)
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)
        
        # Extract top-k values and indices per token
        topk_probs, topk_indices = jax.lax.top_k(gate_probs, self.k)  # Shapes: (M, k)
        
        # Normalize top-k weights dynamically so they sum perfectly back to 1.0
        topk_probs = topk_probs / jnp.sum(topk_probs, axis=-1, keepdims=True)
        
        # 2. Token Duplication for Ragged Layout
        # Duplicate each token vector k times because it must be routed to k distinct experts
        x_flat_expanded = jnp.repeat(x_flat, self.k, axis=0)  # Shape: (M * k, D)
        expert_indices_flat = topk_indices.reshape(-1)         # Shape: (M * k,)
        chosen_probs_flat = topk_probs.reshape(-1)             # Shape: (M * k,)
        
        # 3. Token Clustering / Sorting
        # Group tokens bound for the same expert continuously in memory
        sort_permutation = jnp.argsort(expert_indices_flat)
        x_flat_sorted = x_flat_expanded[sort_permutation]
        
        # 4. Tally Expert Loading Capacity
        # Count exactly how many total tokens hit each expert group (Required by XLA)
        group_sizes = jnp.bincount(expert_indices_flat, length=self.num_experts)
        
        # 5. First Ragged Dot Parallel Matrix Multiply (w_in)
        # Input: (M * k, dim), Weights: (num_experts, dim, ffn_dim) -> Output: (M * k, ffn_dim)
        h_flat_sorted = jax.lax.ragged_dot(x_flat_sorted, self.w_in.value, group_sizes)
        h_flat_sorted = jax.nn.gelu(h_flat_sorted)
        
        # 6. Second Ragged Dot Parallel Matrix Multiply (w_out)
        # Input: (M * k, ffn_dim), Weights: (num_experts, ffn_dim, dim) -> Output: (M * k, dim)
        out_flat_sorted = jax.lax.ragged_dot(h_flat_sorted, self.w_out.value, group_sizes)
        
        # 7. Un-sort & Re-index Operations
        # Return the evaluated tokens back to their original sequential position indexes
        inv_permutation = jnp.argsort(sort_permutation)
        out_flat_expanded = out_flat_sorted[inv_permutation]
        
        # Apply the normalized routing weight coefficients to their respective tokens
        out_flat_scaled = out_flat_expanded * chosen_probs_flat[:, None]
        
        # 8. Accumulate and Merge Multiple Expert Outputs 
        # Reconstruct back to (M, k, D) and sum across the k dimension
        out_flat_combined = out_flat_scaled.reshape(-1, self.k, D).sum(axis=1)  # Shape: (M, D)
        
        # Reconstruct full 3D sequence shape and perform layer normalization + residual connection
        moe_out = out_flat_combined.reshape(B, S, D)
        return x + self.norm(moe_out)

# ==========================================
# 3. TRANSFORMER STRUCTURE
# ==========================================
class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, num_experts: int, k: int, rngs: nnx.Rngs):
        super().__init__()
        self.attn = Attention(dim, num_heads, rngs)
        self.ffn = MoE_FFN(dim, ffn_dim, num_experts,k, rngs)

    def __call__(self, x, mask):
        x = self.attn(x, mask)
        x = self.ffn(x)
        return x

class Transformer(nnx.Module):
    def __init__(
        self, num_layers: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        vocab_size: int,
        seq_len: int,
        k: int,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nnx.Embed(num_embeddings=vocab_size, features=d_model, param_dtype=jnp.bfloat16, rngs=rngs)
        self.positional_embedding = nnx.Embed(num_embeddings=seq_len, features=d_model, param_dtype=jnp.bfloat16, rngs=rngs)
        self.blocks = nnx.List([TransformerBlock(d_model, num_heads, ffn_dim, num_experts, k, rngs) for _ in range(num_layers)])

    def __call__(self, token_ids):
        # Generate masking matrix
        causal_mask = jnp.tril(jnp.ones((self.seq_len, self.seq_len), dtype=jnp.bool_))
        mask = jnp.expand_dims(causal_mask, axis=(0, 1))
        
        embed_matrix = self.embedding.embeddings.value
        
        pos = self.positional_embedding(jnp.arange(self.seq_len, dtype=jnp.int32))
        x = self.embedding(token_ids) + pos
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        # CRITICAL FIX: Replaced 3D LM head einsum with high-throughput 2D jnp.dot
        B, S, D = x.shape
        logits_flat = jnp.dot(x.reshape(-1, D), embed_matrix.T)  # (B * S, V)
        return logits_flat.reshape(B, S, -1)                      # (B, S, V)