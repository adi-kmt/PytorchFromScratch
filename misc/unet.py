import torch
import torchvision.transforms.functional
import torch.nn as nn


class Conv3x3(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=3,
							   padding=1)
		self.conv2 = nn.Conv2d(in_channels=out_channels,
							   out_channels=out_channels,
							   kernel_size=3,
							   padding=1)

	def forward(self, x: torch.Tensor):
		x = nn.ReLU(self.conv1(x))
		return nn.ReLU(self.conv2(x))


class Downsample(nn.Module):
	def __init__(self, pool_size: int = 2):
		self.pool = nn.MaxPool2d(pool_size)

	def forward(self, x: torch.Tensor):
		return self.pool(x)


class Upsample(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		self.convT2d = nn.ConvTranspose2d(in_channels=in_channels,
										  out_channels=out_channels,
										  kernel_size=2, stride=2)

	def forward(self, x: torch.Tensor):
		return self.convT2d(x)


class CropAndConcat(nn.Module):
	def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
		contracting_x = torchvision.transforms.functional.center_crop(
			contracting_x, [x.shape[2], x.shape[3]])
		return torch.cat([x, contracting_x], dim=1)


class UNet(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()

		self.down_conv = nn.ModuleList([Conv3x3(i, o) for i, o in
										[(in_channels, 64), (64, 128), (
											128, 256), (256, 512)]])

		self.down_sample = nn.ModuleList([Downsample() for _ in range(4)])

		self.middle_conv = Conv3x3(512, 1024)

		self.up_sample = nn.ModuleList([Upsample(i, o) for i, o in
										[(1024, 512), (512, 256), (
											256, 128), (128, 64)]])

		self.up_conv = nn.ModuleList([Conv3x3(i, o) for i, o in
									  [(1024, 512), (512, 256), (256, 128), (
										  128, 64)]])

		self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

		self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor):
		pass_through = []
		for i in range(len(self.down_conv)):
			x = self.down_conv[i](x)
			pass_through.append(x)
			x = self.down_sample[i](x)
		x = self.middle_conv(x)
		for i in range(len(self.up_conv)):
			x = self.up_sample[i](x)
			x = self.concat[i](x, pass_through.pop())
			x = self.up_conv[i](x)
		return self.final_conv(x)
