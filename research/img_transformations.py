import albumentations as A
import torch


def train_transform(is_crop=True):
    transforms = A.Sequential([
        A.Resize(320, 320),
        A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 2.), saturation=(0.5, 2.), hue=(-0.2, 0.2), p=0.7),
        A.InvertImg(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], p=1)
    if is_crop:
        return A.Compose([
            A.PadIfNeeded(1008, 1008),
            A.RandomCrop(1008, 1008),
            transforms,
        ])
    else:
        return A.Compose([
            transforms
        ])

def evaluate_transform(is_crop=True):
    transforms = A.Sequential([
        A.Resize(320, 320),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], p=1)
    if is_crop:
        return A.Compose([
            A.PadIfNeeded(1008, 1008),
            A.RandomCrop(1008, 1008),
            transforms,
        ])
    else:
        return A.Compose([
            transforms
        ])


def separate_image_on_steps(img, steps_row, steps_col, mask=None):
    imgs = []
    M = img.shape[0]
    N = img.shape[1]
    step_row = M // steps_row
    step_col = N // steps_col

    for i in range(steps_row):
        for j in range(steps_col):
            imgs.append(img[i * step_row : (i + 1) * step_row, j * step_col : (j + 1) * step_col])

    if mask is not None:
        masks = []
        for i in range(steps_row):
            for j in range(steps_col):
                masks.append(mask[i * step_row: (i + 1) * step_row, j * step_col: (j + 1) * step_col])
        return imgs, masks
    else:
        return imgs

def gather_image_from_pieces(img_pieces, steps_row, steps_col, mask_pieces=None):
    img_rows = []
    for i in range(steps_row):
        img_rows.append(torch.cat(img_pieces[i * steps_col : (i + 1) * steps_col], 1))

    img = torch.cat(img_rows, 0)


    if mask_pieces is not None:
        mask_rows = []
        for i in range(steps_row):
            mask_rows.append(torch.cat(mask_pieces[i * steps_col: (i + 1) * steps_col], 2))

        mask = torch.cat(mask_rows, 1)
        return img, mask
    else:
        return img