"""
Reduces complexity of self attention from O(n^2) to O(n)

This occurs as the E matrix can be shared among multiple heads, can also be learnt
"""

import torch
import torch.nn as nn
from einops import rearrange
from transformers.vanilla.components.multi_head_attention import \
	compute_multinum_HeadSelfAttention


class LinformerAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=None,
				 proj_shape=None, shared_projection=False,
				 trainable_projection=False):
		super(LinformerAttention, self).__init__()
		self.dim_head = (int(dim / num_heads)) if dim_head is None else dim_head
		self.num_heads = num_heads
		self.to_kqv = nn.Linear(dim, dim * 3, bias=False)
		self.W0 = nn.Linear(dim, dim, bias=False)
		self.scale_factor = self.dim
		self.shared_projection = shared_projection

		if not shared_projection:
			self.E = torch.nn.Parameter(torch.randn(proj_shape),
										requires_grad=trainable_projection)

	def forward(self, x, proj_matrix=None):
		E = proj_matrix if (
				self.shared_projection and proj_matrix is not None) else self.E

		qkv = self.to_kqv(x)

		q, k, v = tuple(
			rearrange(qkv, "b t (k d h) -> k b h t d", k=3, h=self.num_heads))

		v, k = self.project_to_e_matrix(v, k, E)

		out = compute_multinum_HeadSelfAttention(q, k, v, self.scale_factor)

		out = rearrange(out, "b h t d -> b t (h d)")

		return self.W0(out)

	def project_to_e_matrix(self, v, k, E):
		v = torch.einsum("b h j d, j k -> b h k d", v, E)
		k = torch.einsum("b h j d, j k -> b h k d", k, E)
		return v, k
