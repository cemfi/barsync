import json
import math
import os
import random

from pathlib import Path
import torch

from torch.utils.data import Dataset


class StretcherDataset(Dataset):
    def __init__(self, root):
        self.files = list(Path(root).rglob('*.pt'))
        shuffle = random.Random(42).shuffle  # Make sure to have reproducible shuffling
        shuffle(self.files)

    def __getitem__(self, index):
        pt_filepath = self.files[index]
        data = torch.load(pt_filepath)
        return data

    def __len__(self):
        return len(self.files)
