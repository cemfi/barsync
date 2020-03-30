import torch
import statistics
import numpy as np
from pathlib import Path

root = '/mnt/data/datasets/cross-mapping-4s/val'

files = list(Path(root).rglob('*.pt'))

total_loss = []
for idx, pt_filepath in enumerate(files):
    x = torch.load(pt_filepath)

    image_width = x['image'].shape[1]
    map = x['map'][39]
    if image_width < 50:
        continue
    map = map / image_width - 0.5
    loss = map * map
    total_loss.append(loss)

    if idx % 10000 == 0 and idx != 0:
        print(statistics.mean(total_loss))

print('Baseline MSE (final) = ', statistics.mean(total_loss))
print('95% quantile', np.quantile(np.array(total_loss), 0.95))
print('99% quantile', np.quantile(np.array(total_loss), 0.99))