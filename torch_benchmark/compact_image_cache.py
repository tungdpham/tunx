"""Lossless cache of normalized ImageNet inputs as cropped uint8 RGB plus a lookup table."""
from pathlib import Path
import numpy as np
import torch


class CompactImageCache:
    def __init__(self, dataset, directory):
        self.dataset = dataset
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        values = torch.arange(256, dtype=torch.float32).div(255)
        self.lut = torch.stack([(values - mean) / std for mean, std in
                               zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]).numpy()
        self.lut.tofile(self.directory / 'normalization.bin')

    def __getitem__(self, index):
        path = self.directory / f'{index}.bin'
        if not path.exists():
            image, label = self.dataset[index]
            a = image.numpy()
            mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
            pixels = np.rint((a * std + mean) * 255).clip(0, 255).astype(np.uint8)
            restored = self.lut[np.arange(3)[:, None, None], pixels]
            if not np.array_equal(a, restored):
                raise ValueError('Image transform is not losslessly representable by the uint8 normalization cache')
            with path.open('wb') as f:
                np.array([label], np.int32).tofile(f)
                pixels.transpose(1, 2, 0).copy().tofile(f)
        with path.open('rb') as f:
            label = int(np.fromfile(f, np.int32, 1)[0])
            pixels = np.fromfile(f, np.uint8).reshape(224, 224, 3).transpose(2, 0, 1)
        return torch.from_numpy(self.lut[np.arange(3)[:, None, None], pixels]), label
