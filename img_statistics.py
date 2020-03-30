import torch
import statistics

from pathlib import Path

root = '/mnt/data/datasets/cross-mapping-4s/val'

files = list(Path(root).rglob('*.pt'))

all_image_width = []
for idx, pt_filepath in enumerate(files):
    x = torch.load(pt_filepath)

    image_width = x['image'].shape[1]
    if image_width < 50:
        continue

    all_image_width.append(image_width)

    if idx % 10000 == 0 and idx != 0:
        print('Mean', statistics.mean(all_image_width))
        print('Median', statistics.median(all_image_width))
        print('*' * 12)

print('FINAL')
print('Mean', statistics.mean(all_image_width))
print('Median', statistics.median(all_image_width))