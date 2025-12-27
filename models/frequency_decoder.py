import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyDecoderBlock(nn.Module):
    def __init__(self, channels, width, height):
        super().__init__()
        #         args = get_args()
        self.ln = nn.LayerNorm(channels)
        self.fsp = nn.Parameter(torch.randn(1, width, height % 2 + 1, channels))
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.hidden_size = channels

        # self.num_blocks = args.fft_blocks
        self.num_blocks = 1
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02  # default = 0 ,但会报错
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU(False)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        residual = x
        B, L, D = x.shape
        a = b = int(math.sqrt(L))

        x = self.ln(x) 
        x = x.reshape(B, a, b, D).float()
        x_fft = torch.fft.rfft2(x, dim=(1, 2))
        x_fft = x_fft.reshape(B, x_fft.shape[1], x_fft.shape[2], self.num_blocks, self.block_size)

        x_real_1 = F.relu(self.multiply(x_fft.real, self.w1[0]) - self.multiply(x_fft.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x_fft.real, self.w1[1]) + self.multiply(x_fft.imag, self.w1[0]) + self.b1[1])
        x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x_fft = torch.stack([x_real_2, x_imag_2], dim=-1).float()

        x_fft = torch.view_as_complex(x_fft)
        x_fft = x_fft.reshape(B, x_fft.shape[1], x_fft.shape[2], self.hidden_size)

        f_tilde = x_fft * self.fsp

        #         x = torch.fft.irfft2(f_tilde, s=(a, b), dim=(1, 2)).reshape(B,L,D)
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2)).reshape(B, L, D)

        output1 = self.ln(residual + x)

        output2 = self.ffn(output1)

        return output2


class FrequencyDecoder(nn.Module):
    def __init__(self, block, num_blocks, channels, width, height):
        super().__init__()
        self.blocks = nn.ModuleList([block(channels, width, height) for _ in range(num_blocks)])
        self.decoder_norm = nn.LayerNorm(channels)
        self.decoder_pred = nn.Linear(channels, patch_size ** 2 * in_chans, bias=True)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)
        print(x.shape)
        x = self.decoder_pred(x)
        return x

if __name__ == '__main__':
    # Example usage
    B, L, D = 1, 64, 256
    width, height = 8, 8
    num_blocks = 1

    model = FrequencyDecoder(FrequencyDecoderBlock, num_blocks, D, width, height)

    x = torch.randn(B, L, D)

    output = model(x)
    print(output.shape)

    predicted_spectrum = torch.fft.fft2(output.view(B, width, height, D))

    print(predicted_spectrum.shape)