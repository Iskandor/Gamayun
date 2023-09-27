import numpy
import torch


def aug_random_apply(x, p, aug_func):
    mask = (torch.rand(x.shape[0]) < p)
    mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    mask = mask.float().to(x.device)
    y = (1.0 - mask) * x + mask * aug_func(x)

    return y


def aug_pixelate(x, p=0.5):
    # downsample 2x or 4x
    scale = int(2 ** numpy.random.randint(1, 3))
    ds = torch.nn.AvgPool2d(scale, scale).to(x.device)
    us = torch.nn.Upsample(scale_factor=scale).to(x.device)

    # tiles mask
    mask = 1.0 - torch.rand((x.shape[0], 1, x.shape[2] // scale, x.shape[3] // scale))
    mask = (mask < p).float().to(x.device)
    mask = us(mask)

    scaled = us(ds(x))
    return mask * scaled + (1.0 - mask) * x


def aug_mask_tiles(x, p=0.1):
    if x.shape[2] == 96:
        tile_sizes = [1, 2, 4, 8, 12, 16]
    else:
        tile_sizes = [1, 2, 4, 8, 16]

    tile_size = tile_sizes[numpy.random.randint(len(tile_sizes))]

    size_h = x.shape[2] // tile_size
    size_w = x.shape[3] // tile_size

    mask = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

    mask = torch.kron(mask, torch.ones(tile_size, tile_size))

    return x * mask.float().to(x.device)


def aug_noise(x, k=0.2):
    pointwise_noise = k * (2.0 * torch.rand(x.shape, device=x.device) - 1.0)
    return x + pointwise_noise
