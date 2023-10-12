import einops
import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
	def __init__(self, embed_dim):
		super(SelfAttention, self).__init__()
		self.qkv_vector = nn.Linear(embed_dim, embed_dim * 3, bias=False)
		self.scale = embed_dim ** -0.5  # 1/sqrt(embed_dim)

	def forward(self, x, mask=None):
		qkv = self.qkv_vector(x)  # Giving dim [batch, embed_dim, embed_dim * 3]

		# Converting the qkv to separate tensors of size (batch, tokens, embed_dim)
		q, k, v = tuple(einops.rearrange(qkv, "b t (k d) -> k b t d", k=3))

		# Now scaled product has the size (batch tokens, tokens)
		# This is a dot product for calculating the similarity
		scaled_product = torch.einsum('b i d, b j d -> b i j', q,
									  k) * self.scale

		# This is for self attention, the items that are in the future are masked
		# to -inf
		if mask is not None:
			assert mask.shape == scaled_product.shape[1:]
			scaled_product = scaled_product.masked_fill(mask, -np.inf)

		# Softmax over the last dimension
		attention = torch.softmax(scaled_product, dim=-1)

		# Assume this operation to be a weighted sum of the attention on the value
		return torch.einsum("b i j, b j d -> b i d", attention, v)
