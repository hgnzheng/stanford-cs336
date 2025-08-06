"""
transformer_util.py
"""
import torch
import torch.nn as nn
import math
import einops

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
        
        rms = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / self.d_model) + self.eps
        
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
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)