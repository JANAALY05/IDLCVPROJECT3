"""Segmentation models.

EncDec: the simple encoder-decoder from the exercise (lib/model/EncDecModel.py).
  Only change: the exercise used fixed upsample sizes (nn.Upsample(16), (32), ...) that
  only work for 128x128 inputs. Here every upsample doubles the size (scale_factor=2),
  so the same network works for PH2 (192x256) and DRIVE (512x512).

UNet: U-Net (Ronneberger et al. 2015) with 4 down and 4 up levels.
  Differences from EncDec:
   - two 3x3 convs per level (+ BatchNorm) instead of one
   - channels double at every level going down (32 -> 64 -> 128 -> 256 -> 512)
   - skip connections: each decoder level gets the encoder features of the same size
     concatenated to it, so fine details (thin vessels, lesion edges) lost by pooling
     come back
   - learned upsampling (transposed convolution) instead of fixed nearest-neighbour
  Padding=1 keeps the size in every conv (the original paper used no padding and
  cropped the skip connections; padding is simpler and gives output size = input size).

Both: input H and W must be divisible by 16 (4 poolings); output = logits [B,1,H,W].

Shape check / U-Net sketch numbers:  python3 models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        # encoder (downsampling): each pool halves H and W
        self.enc_conv0 = nn.Conv2d(3, ch, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # bottleneck (1/16 of the input size)
        self.bottleneck_conv = nn.Conv2d(ch, ch, 3, padding=1)

        # decoder (upsampling): each upsample doubles H and W
        self.upsample0 = nn.Upsample(scale_factor=2)
        self.dec_conv0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dec_conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.dec_conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.dec_conv3 = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))
        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation -> logits
        return d3


class DoubleConv(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) twice. Keeps H and W, changes channels."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]  # 32, 64, 128, 256, 512

        # encoder
        self.enc0 = DoubleConv(3, c[0])
        self.enc1 = DoubleConv(c[0], c[1])
        self.enc2 = DoubleConv(c[1], c[2])
        self.enc3 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2, 2)

        # bottleneck
        self.bottleneck = DoubleConv(c[3], c[4])

        # decoder: upsample (x2, halve channels), concatenate skip, DoubleConv
        self.up3 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec3 = DoubleConv(c[3] * 2, c[3])
        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec2 = DoubleConv(c[2] * 2, c[2])
        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec1 = DoubleConv(c[1] * 2, c[1])
        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec0 = DoubleConv(c[0] * 2, c[0])

        # 1x1 conv: one score per pixel
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x, verbose=False):
        shapes = []

        def log(name, t):
            if verbose:
                shapes.append((name, tuple(t.shape[1:])))
            return t

        log("input", x)
        # encoder (s = skip connection, saved before pooling)
        s0 = log("enc0", self.enc0(x))
        s1 = log("enc1", self.enc1(log("pool0", self.pool(s0))))
        s2 = log("enc2", self.enc2(log("pool1", self.pool(s1))))
        s3 = log("enc3", self.enc3(log("pool2", self.pool(s2))))
        b = log("bottleneck", self.bottleneck(log("pool3", self.pool(s3))))
        # decoder
        d3 = log("dec3", self.dec3(torch.cat([log("up3", self.up3(b)), s3], dim=1)))
        d2 = log("dec2", self.dec2(torch.cat([log("up2", self.up2(d3)), s2], dim=1)))
        d1 = log("dec1", self.dec1(torch.cat([log("up1", self.up1(d2)), s1], dim=1)))
        d0 = log("dec0", self.dec0(torch.cat([log("up0", self.up0(d1)), s0], dim=1)))
        y = log("output", self.out(d0))

        if verbose:
            for name, (ch, h, w) in shapes:
                print(f"  {name:11s} {ch:4d} x {h:4d} x {w:4d}")
        return y


def get_model(name):
    if name == "encdec":
        return EncDec()
    if name == "unet":
        return UNet()
    raise ValueError(f"unknown model {name}")


if __name__ == "__main__":
    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    print("EncDec:", n_params(EncDec()), "parameters")
    for h, w in [(192, 256), (512, 512)]:
        print(f"  {(h, w)} -> {tuple(EncDec()(torch.zeros(1, 3, h, w)).shape[2:])}")

    unet = UNet().eval()
    print("\nUNet:", n_params(unet), "parameters")
    for h, w in [(192, 256), (512, 512)]:
        print(f"\nUNet feature maps for input {h} x {w}  (channels x height x width):")
        with torch.no_grad():
            out = unet(torch.zeros(1, 3, h, w), verbose=True)
        assert out.shape[2:] == (h, w), "output size must equal input size"
