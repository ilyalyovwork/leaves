import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from img_transformations import separate_image_on_steps

#binary_factor = 255

mask_folder = 'segmented'

class LeavesDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train', steps_row=1, steps_col=1):
        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        # steps - на какое количество окон будет разбито изображение
        self.steps_row = steps_row
        self.steps_col = steps_col

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        if self.mode == 'train':
            mask = load_mask(img_file_name)
            imgs, masks = separate_image_on_steps(image, self.steps_row, self.steps_col, mask=mask)
            augmented_imgs = []
            augmented_masks = []
            for img, mask in zip(imgs, masks):
                data = {"image": img, "mask": mask}
                augmented = data if not self.transform else self.transform(**data)
                img, mask = img_to_tensor(augmented["image"]),\
                              torch.from_numpy(np.expand_dims(augmented["mask"], 0)).float()

                augmented_imgs.append(img)
                augmented_masks.append(mask)
            return torch.stack(augmented_imgs), torch.stack(augmented_masks)
        else:
            imgs = separate_image_on_steps(image, self.steps_row, self.steps_col)
            augmented_imgs = []
            for img in imgs:
                data = {"image": img}
                augmented = data if not self.transform else self.transform(**data)
                img = img_to_tensor(augmented["image"])

                augmented_imgs.append(img)

            return torch.stack(augmented_imgs), str(img_file_name)



def load_image(path):
    try:
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        print(f'Error in cv2, path:{path}')


def load_mask(path):

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask > 0).astype(np.uint8)
