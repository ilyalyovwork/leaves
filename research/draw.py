import numpy as np
import albumentations as A
import cv2


# overlayed white mask over image
def mask_overlay(image, mask, color=(0, 0, 255), resize=(320, 320)):
    """
    Helper function to visualize mask on the top of the car
    """
    if resize:
      resizer = A.Resize(*resize)
      image = resizer(image=image)['image']
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 2] > 0
    img[ind] = weighted_sum[ind]
    return img