import torch
from pathlib import Path
from shutil import copyfile
import os

root='/mnt/data/datasets/cross-mapping'

def check_pt(root, train):
    if train:
        root = root + '/train'
        dest_dir = './data/train'
    else:
        dest_dir = './data/val'
        root = root + '/val'
    os.makedirs(dest_dir, exist_ok=True)

    n_bad_files = 0
    all_files = list(Path(root).rglob('*.pt'))
    for pt_filepath in all_files:
        data = torch.load(pt_filepath)
        x = data['image'].shape[0]
        y = data['image'].shape[1]
        assert(x == 200)
        if (y < 50):
            print(pt_filepath, data['image'].shape)
            n_bad_files += 1
        else:
            name = os.path.split(pt_filepath)[-1]
            target = os.path.join(dest_dir, name)
            copyfile(pt_filepath, target)

    return (n_bad_files, len(all_files))


bad_train, all_train = check_pt(root, train=True)
bad_val, all_val = check_pt(root, train=False)

print('train', bad_train, all_train)
print('val', bad_val, all_val)
