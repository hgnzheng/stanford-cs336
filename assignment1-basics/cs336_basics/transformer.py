"""
transformer_util.py
"""
import torch
import torch.nn as nn
import math
import einops
import numpy as np
from typing import List, Optional
from collections.abc import Callable, Iterable
from typing import IO, Union
from pathlib import Path

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct a linear transformation module.

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (str | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features),
        ).to(self.device).to(self.dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        # Numpy and PyTorch both store matrices in row-major order
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        weight_tensor = torch.empty(self.num_embeddings, self.embedding_dim)
        
        self.weight = nn.Parameter(
            weight_tensor,
        ).to(self.device).to(self.dtype)
        
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup the embedding vectors for the given token ids."""
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability
            device (torch.device | None, optional): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.empty(self.d_model)).to(self.device).to(self.dtype)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, seq_len, d_model) and
        return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        result = x / rms * self.weight
        
        return result.to(in_dtype)



class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
    ):
        """Construct the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int, optional): Hidden dimension. Calculated as 8 * d_model / 3 by default
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = self.approximate_d_ff(d_model)
        self.d_ff = d_ff
        
        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        
    def approximate_d_ff(self, d_model: int) -> int:
        """
        Approximate the hidden dimension based on the model dimension.
        Set to approximately 8 * d_model / 3 while ensuring it is a multiple of 64.
        """
        ideal_d_ff = (8 * d_model) / 3
        d_ff = ((ideal_d_ff + 31) // 64) * 64
        return d_ff

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, seq_len, d_model) and
        return a tensor of the same shape.
        """
        gate = x @ self.w1.T
        gate = gate * torch.sigmoid(gate)
        
        value = x @ self.w3.T
        hidden = gate * value
        output = hidden @ self.w2.T
        return output

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Constructor for RoPE module.

        Args:
            theta (float): Theta value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None, optional): Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        assert d_k % 2 == 0, "d_k must be even"
        self.max_seq_len = max_seq_len
        
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0) # (max_seq_len, d_k // 2)
        
        # Precompute cos and sin values
        cos_values = torch.cos(angles) # (max_seq_len, d_k // 2)
        sin_values = torch.sin(angles) # (max_seq_len, d_k // 2)
        
        rot_matrix = torch.zeros(self.max_seq_len, self.d_k // 2, 2, 2)
        rot_matrix[..., 0, 0] = cos_values
        rot_matrix[..., 0, 1] = -sin_values
        rot_matrix[..., 1, 0] = sin_values
        rot_matrix[..., 1, 1] = cos_values
        
        if device is not None:
            rot_matrix = rot_matrix.to(device)
        
        self.register_buffer("rot_matrix", rot_matrix, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to the input tensor."""
        device = x.device
        if self.rot_matrix.device != device:
            self.rot_matrix = self.rot_matrix.to(device)
        
        x_pairs = einops.rearrange(
            x,
            "... seq_len (d_half pair) -> ... seq_len d_half pair",
            pair=2,
        )
        
        original_shape = token_positions.shape
        token_positions_flat = token_positions.reshape(-1)
        
        rot_matrices_flat = self.rot_matrix[token_positions_flat]
        
        # Handle broadcasting
        target_shape = list(original_shape) + [self.d_k // 2, 2, 2]
        rot_matrices = rot_matrices_flat.reshape(target_shape)

        # Apply rotation using einsum
        x_rotated = torch.einsum(
            "...shq,...shpq->...shp", 
            x_pairs, 
            rot_matrices
        )
        # Reshape back to original shape
        x_rotated = einops.rearrange(
            x_rotated,
            "... seq_len d_half pair -> ... seq_len (d_half pair)",
            pair=2,
        )
        return x_rotated

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Apply the softmax operation on a tensor."""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    exp_sum = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / exp_sum

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Implements scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k)
        key (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len, d_k)
        value (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len, d_k)
        mask (torch.Tensor | None, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, ..., seq_len, d_k)
    """
    d_k = query.shape[-1]
    
    scores = torch.einsum('...qd,...kd ->...qk', query, key) / math.sqrt(d_k)
    
    if mask is not None:
        mask_value = float('-inf')
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, mask_value)
        
        attention_weights = softmax(scores, dim=-1)
        output = torch.einsum('...qk,...kd ->...qd', attention_weights, value)
        return output

class MultiHeadSelfAttention(nn.Module):
    """
    Implements causal multi-head self-attention with RoPE.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        use_rope: bool = True,
        theta: float = 10000,
    ):
        """Constructor for the multi-head self-attention module."""
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_rope = use_rope
        
        self.W_q = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_k = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_o = nn.Parameter(torch.empty(self.d_model, self.d_model))
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
            )
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create a causal mask for where position i can only attend to positions j<=i."""
        upper_triangle = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = upper_triangle == 0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention with RoPE."""
        batch_size, seq_len, _ = x.shape
        
        Q = torch.matmul(x, self.W_q) # (batch_size, seq_len, d_model)
        K = torch.matmul(x, self.W_k) # (batch_size, seq_len, d_model)
        V = torch.matmul(x, self.W_v) # (batch_size, seq_len, d_model)
        
        Q = einops.rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = einops.rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = einops.rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)
        
        if self.use_rope:
            if token_positions is None:
                positions = torch.arange(seq_len, device=x.device)
                positions = einops.repeat(positions, 's -> b s', b=batch_size)
            else:
                positions = token_positions
        
            # Treat heads as batch dimension
            Q_flat = einops.rearrange(Q, 'b h s d -> (b h) s d')
            K_flat = einops.rearrange(K, 'b h s d -> (b h) s d')
            positions_flat = einops.repeat(positions, 'b s -> (b h) s', h=self.num_heads)
            
            Q_flat = self.rope(Q_flat, positions_flat)
            K_flat = self.rope(K_flat, positions_flat)
            
            Q = einops.rearrange(Q_flat, '(b h) s d -> b h s d', b=batch_size, h=self.num_heads)
            K = einops.rearrange(K_flat, '(b h) s d -> b h s d', b=batch_size, h=self.num_heads)
            
        causal_mask = self._create_causal_mask(seq_len, device=x.device)
        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        attention_output = einops.rearrange(attention_output, 'b h s d -> b s (h d)')
        
        output = torch.matmul(attention_output, self.W_o)
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        use_rope: bool = True,
        theta: float = 10000,
        eps: float = 1e-5,
    ):
        """Constructor for the transformer block module."""
        super().__init__()        
        self.norm_1 = RMSNorm(d_model, eps=eps)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, use_rope, theta)
        self.norm_2 = RMSNorm(d_model, eps=eps)
        self.ff = SwiGLU(d_model, d_ff)
    
    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Apply the transformer block.
        
        This follows the pre-norm architecture:
        1. First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        2. Second sublayer: y = y + FeedForward(RMSNorm(y))
        """
        y = x + self.mha(self.norm_1(x))
        output = y + self.ff(self.norm_2(y))
        return output

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        rope_theta: float = 10000,
        eps: float = 1e-5,
    ):
        """Constructor for the Transformer Language Model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            context_length (int): Maximum context length, necessary for determining the dimensionality of the positional embedding matrix
            d_model (int): Hidden dimension of the model
            num_layers (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of the feedforward network
            use_rope (bool, optional): Whether to use RoPE
            rope_theta (float, optional): Theta value for the RoPE
            eps (float, optional): Epsilon value for numerical stability
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                use_rope=use_rope,
                theta=rope_theta,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, eps=eps)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Transformer Language Model."""
        x = self.token_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy loss."""
    num_classes = logits.shape[-1]
    logits_flat = logits.view(-1, num_classes)
    targets_flat = targets.view(-1)
    
    max_logits = torch.max(logits_flat, dim=1, keepdim=True).values
    stable_logits = logits_flat - max_logits
    
    log_sum_exp = max_logits.squeeze(1) + torch.log(
        torch.sum(torch.exp(stable_logits), dim=1)
    )
    
    target_logits = logits_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    losses = -target_logits + log_sum_exp
    return losses.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """Constructor for the AdamW optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float, optional): Learning rate
            betas (tuple[float, float], optional): Coefficients for running averages of gradient and its square
            eps (float, optional): Term added to the denominator to improve numerical stability
            weight_decay (float, optional): Strength of the weight decay regularization
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params,defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """Take a single optimization step."""
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                denom = exp_avg_sq.sqrt().add_(eps)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss

def learning_rate_scheduler(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
):
    """
    Implements a learning rate scheduler that uses a cosine annealing schedule with warmup.
    
    Args:
        t (int): Current training step
        alpha_max (float): Maximum learning rate
        warmup_steps (int): Number of warmup steps
        cosine_anneal_steps (int): Number of cosine annealing steps
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(
    params: Iterable[torch.Tensor],
    max_l2_norm: float,
    eps: float = 1e-6,
):
    """
    Implements gradient clipping.
    
    Args:
        params (Iterable[torch.Tensor]): Iterable of parameters to clip
        max_norm (float): Maximum norm of the gradient
    """
    total_norm_sq = 0.0
    for param in params:
        if param.grad is not None:
            total_norm_sq += torch.sum(param.grad.data ** 2).item()
    total_norm = math.sqrt(total_norm_sq)
    
    if total_norm > max_l2_norm:
        clip_factor = max_l2_norm / (total_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)

def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
):
    """
    Loads data into a tensor and splits it into batches.

    Args:
        x (np.ndarray): Input data
        batch_size (int): Batch size
        context_length (int): Context length
        device (str): Device to load the data onto
    """
    ix = torch.randint(len(x) - context_length, (batch_size,))
    
    x_batch = torch.stack([torch.from_numpy((x[i:i+context_length]).astype(np.int64)) for i in ix])
    y_batch = torch.stack([torch.from_numpy((x[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
    
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    return x_batch, y_batch

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, Path, IO[bytes]],
):
    """Save a checkpoint of the model and optimizer."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    
    torch.save(checkpoint, out)

def load_checkpoint(
    src: Union[str, Path, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Load a checkpoint of the model and optimizer."""
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration