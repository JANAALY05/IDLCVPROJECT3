"""Data loaders for PH2 and DRIVE.

Built from the exercise loader (lib/dataset/PhCDataset.py, class PhC) as a starting point.
Changes compared to PhC:
  1. `split` ("train"/"val"/"test") instead of `train=True/False`, read from splits.json,
     because PH2 and DRIVE have no usable pre-made splits.
  2. File paths built from the case IDs (PH2 and DRIVE have their own folder layouts).
  3. Separate transforms for image and label: the label is resized with NEAREST so it
     stays 0/1 (PhC uses the same transform for both, which blurs the mask).
  4. DRIVE also returns the field-of-view (FOV) mask, so metrics can ignore the black
     border. PH2 returns an all-ones FOV so both datasets look the same to the training code.
  5. Optional random flips for training (same flip for image, label and FOV).
  (Also fixes PhC's missing imports: glob and PIL.Image.)

Each sample: X [3,H,W] in 0..1, Y [1,H,W] in {0,1}, FOV [1,H,W] in {0,1}

Check the loaders:  python3 datasets.py
"""
import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

DATA_PATH = '/dtu/datasets1/02516'
SPLITS_PATH = 'splits.json'

# (height, width), both divisible by 16 for a 4-level U-Net
PH2_SIZE = (192, 256)    # keeps PH2's ~4:3 shape
DRIVE_SIZE = (512, 512)  # large, because vessels are only a few pixels wide


def make_transforms(size):
    """Image: bilinear resize. Label/FOV: nearest resize (stays binary)."""
    image_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()])
    label_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])
    return image_transform, label_transform


def load_ids(dataset, split):
    with open(SPLITS_PATH) as f:
        return json.load(f)[dataset][split]


class PH2(Dataset):
    def __init__(self, split, size=PH2_SIZE, augment=False):
        'Initialization'
        self.image_transform, self.label_transform = make_transforms(size)
        self.augment = augment
        root = os.path.join(DATA_PATH, 'PH2_Dataset_images')
        ids = load_ids('PH2', split)
        self.image_paths = [os.path.join(root, c, f'{c}_Dermoscopic_Image', f'{c}.bmp') for c in ids]
        self.label_paths = [os.path.join(root, c, f'{c}_lesion', f'{c}_lesion.bmp') for c in ids]

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')
        fov = Image.new('L', image.size, 255)  # PH2 has no FOV: every pixel counts
        return finish(self, image, label, fov)


class DRIVE(Dataset):
    def __init__(self, split, size=DRIVE_SIZE, augment=False):
        'Initialization'
        self.image_transform, self.label_transform = make_transforms(size)
        self.augment = augment
        root = os.path.join(DATA_PATH, 'DRIVE', 'training')  # test/ has no vessel labels
        ids = load_ids('DRIVE', split)
        self.image_paths = [os.path.join(root, 'images', f'{n}_training.tif') for n in ids]
        self.label_paths = [os.path.join(root, '1st_manual', f'{n}_manual1.gif') for n in ids]  # vessel mask
        self.fov_paths = [os.path.join(root, 'mask', f'{n}_training_mask.gif') for n in ids]    # FOV mask

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')
        fov = Image.open(self.fov_paths[idx]).convert('L')
        return finish(self, image, label, fov)


def finish(ds, image, label, fov):
    """Shared last steps: optional flips, transforms, make masks exactly 0/1."""
    if ds.augment:
        if random.random() < 0.5:
            image, label, fov = TF.hflip(image), TF.hflip(label), TF.hflip(fov)
        if random.random() < 0.5:
            image, label, fov = TF.vflip(image), TF.vflip(label), TF.vflip(fov)
    X = ds.image_transform(image)
    Y = (ds.label_transform(label) > 0.5).float()
    FOV = (ds.label_transform(fov) > 0.5).float()
    return X, Y, FOV


def get_loaders(name, batch_size, num_workers=3):
    """Train (shuffled + flips), val and test loaders, like train.py in the exercise."""
    cls = PH2 if name == 'PH2' else DRIVE
    train_loader = DataLoader(cls('train', augment=True), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(cls('val'), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(cls('test'), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ---------------- Check the loaders ----------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # no screen on the cluster -> save to file
    import matplotlib.pyplot as plt

    for name, bs in [('PH2', 8), ('DRIVE', 2)]:
        train_loader, val_loader, test_loader = get_loaders(name, bs, num_workers=0)
        print(f'\n{name}: {len(train_loader.dataset)} train, '
              f'{len(val_loader.dataset)} val, {len(test_loader.dataset)} test')

        X, Y, FOV = next(iter(train_loader))
        print('  X:  ', tuple(X.shape), 'range', round(X.min().item(), 3), '-', round(X.max().item(), 3))
        print('  Y:  ', tuple(Y.shape), 'values', torch.unique(Y).tolist())
        print('  FOV:', tuple(FOV.shape), 'values', torch.unique(FOV).tolist())

        # Foreground fraction inside the FOV over the training set (no flips).
        # Used later for the weighted loss: pos_weight = negatives / positives.
        cls = PH2 if name == 'PH2' else DRIVE
        pos = tot = 0
        for _, y, f in DataLoader(cls('train'), batch_size=bs):
            pos += (y * f).sum().item()
            tot += f.sum().item()
        frac = pos / tot
        print(f'  foreground fraction: {frac:.3f}  -> pos_weight ~ {(1 - frac) / frac:.1f}')

        # Picture: image | mask | mask outline on image
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(X[0].permute(1, 2, 0)); ax[0].set_title('image')
        ax[1].imshow(Y[0, 0], cmap='gray'); ax[1].set_title('mask')
        ax[2].imshow(X[0].permute(1, 2, 0))
        ax[2].contour(Y[0, 0], levels=[0.5], colors='lime', linewidths=0.8)
        ax[2].set_title('overlay')
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(f'check_{name}.png', dpi=100)
        print(f'  saved check_{name}.png')
