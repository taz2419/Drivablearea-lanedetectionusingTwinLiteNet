import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def random_perspective(combination, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)
    return (img, gray, line)

class MyDataset(torch.utils.data.Dataset):
    '''
    Dataset loader that works with configurable root path.
    Expects structure:
      <root>/
        images/{train,val}/*.jpg
        segments/{train,val}/*.png
        lane/{train,val}/*.png
    
    Usage:
      - Pass data_root="/kaggle/input/bdd100k_validation_only" explicitly, OR
      - Set environment variable BDD100K_ROOT="/kaggle/input/...", OR
      - Falls back to /data/bdd100k (local default)
    '''
    def __init__(self, transform=None, valid=False, data_root=None):
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid

        # Resolve dataset root: CLI arg > ENV > default
        BDD_ROOT_DEFAULT = "data/bdd100k"  # or current default
        BDD_ROOT = os.getenv("BDD100K_ROOT", BDD_ROOT_DEFAULT)

        base = data_root or BDD_ROOT
        split = 'val' if valid else 'train'


        self.img_root = os.path.join(base, 'images', split)
        self.seg_root = os.path.join(base, 'segments', split)
        self.lane_root = os.path.join(base, 'lane', split)

        # Validate directories exist
        for d in (self.img_root, self.seg_root, self.lane_root):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Required directory not found: {d}\n"
                    f"Expected structure: {base}/{{images,segments,lane}}/{split}/"
                    f"Available: {os.listdir(base) if os.path.isdir(base) else 'base not found'}"
                )
                

        # Collect image filenames
        self.names = sorted([n for n in os.listdir(self.img_root) if n.lower().endswith('.jpg')])

        if len(self.names) == 0:
            raise FileNotFoundError(f"No .jpg images found in {self.img_root}")

        print(f"Loaded {len(self.names)} images from {self.img_root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        W_, H_ = 640, 360
        name = self.names[idx]

        image_path = os.path.join(self.img_root, name)
        seg_path = os.path.join(self.seg_root, name.replace('.jpg', '.png'))
        lane_path = os.path.join(self.lane_root, name.replace('.jpg', '.png'))

        image = cv2.imread(image_path)
        label1 = cv2.imread(seg_path, 0)
        label2 = cv2.imread(lane_path, 0)

        if image is None or label1 is None or label2 is None:
            missing = []
            if image is None: missing.append(image_path)
            if label1 is None: missing.append(seg_path)
            if label2 is None: missing.append(lane_path)
            raise FileNotFoundError(f"Missing or unreadable file(s): {missing}")

        # Apply augmentations only during training
        if not self.valid:
            if random.random() < 0.5:
                (image, label1, label2) = random_perspective(
                    combination=(image, label1, label2),
                    degrees=10, translate=0.1, scale=0.25, shear=0.0
                )
            if random.random() < 0.5:
                augment_hsv(image)
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label1 = np.fliplr(label1).copy()
                label2 = np.fliplr(label2).copy()

        # Resize to target dimensions
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))

        # Create binary masks for drivable area and lane
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)

        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)  # [2, H, W]
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)  # [2, H, W]

        # Convert BGR to RGB and HWC to CHW
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        # CHANGED: Concatenate targets into single tensor instead of tuple
        target = torch.cat([seg_da, seg_ll], dim=0)  # [4, H, W]
        return image_path, torch.from_numpy(image), target




