import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

"""
Dictionary that determines the class remapping of the cityscapes labels:

0 - void
1 - road
2 - sidewalk
3 - construction
4 - street object
5 - traffic sign/light
6 - nature
7 - sky
8 - person
9 - vehicle
"""
mapping_10 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 3,
    13: 3,
    14: 0,
    15: 0,
    16: 0,
    17: 4,
    18: 0,
    19: 5,
    20: 5,
    21: 6,
    22: 6,
    23: 7,
    24: 8,
    25: 8,
    26: 9,
    27: 9,
    28: 9,
    29: 0,
    30: 0,
    31: 9,
    32: 9,
    33: 9,
    -1: 0
}

"""
Class remapping function.
"""
def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_10:
        label_mask[mask == k] = mapping_10[k]
    return label_mask

"""
Input pre-processing function. Resizes the image and remaps the labels
"""
def get_test_inputs(img, labels):
    img = transforms.Resize((256, 512))(img)
    labels = transforms.Resize((256, 512), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(labels)
    img = transforms.ToTensor()(img)
    labels = np.array(labels)
    encoded_labels = encode_labels(labels)
    labels_tensor = torch.from_numpy(encoded_labels).type(torch.LongTensor)

    return img, labels_tensor

"""
Normalizes the input image, based on pre-computed means and stdevs
"""
def normalize(img):
    return transforms.Normalize(mean=[0.28689554, 0.32513303, 0.28389177],
                                std=[0.18696375, 0.19017339, 0.18720214])(img)