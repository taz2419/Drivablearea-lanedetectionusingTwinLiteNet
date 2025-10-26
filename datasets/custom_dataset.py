import os
from glob import glob
from typing import List, Tuple, Sequence, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def _parse_ids(ids: Optional[Sequence[int] | str], default: Sequence[int]) -> List[int]:
    if ids is None:
        return list(default)
    if isinstance(ids, str):
        ids = ids.strip()
        if not ids:
            return []
        return [int(x) for x in ids.split(",")]
    return [int(x) for x in ids]

class CustomTwinLiteDataset(Dataset):
    """
    Layout:
      root/
        images/val/*.png|jpg
        segments/val/*.png  (per-pixel class ids or palette)
      Optional:
        segments/val/da/*.png and segments/val/ll/*.png (binary masks)
    """
    def __init__(
        self,
        root: str,
        split: str = "val",
        img_size: Optional[Tuple[int, int]] = None,  # (W,H)
        da_ids: Optional[Sequence[int] | str] = None,
        ll_ids: Optional[Sequence[int] | str] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size

        # Defaults (Cityscapes trainIds commonly use road=0)
        env_da = os.getenv("DA_IDS", "0")
        env_ll = os.getenv("LL_IDS", "")
        self.da_ids = _parse_ids(da_ids, default=[int(x) for x in env_da.split(",") if x.strip().isdigit()] or [0])
        self.ll_ids = _parse_ids(ll_ids, default=[int(x) for x in env_ll.split(",") if x.strip().isdigit()])

        self.img_dir = os.path.join(root, "images", split)
        self.seg_dir = os.path.join(root, "segments", split)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Images dir not found: {self.img_dir}")
        if not os.path.isdir(self.seg_dir):
            raise FileNotFoundError(f"Segments dir not found: {self.seg_dir}")

        self.seg_da_dir = os.path.join(self.seg_dir, "da")
        self.seg_ll_dir = os.path.join(self.seg_dir, "ll")
        self.use_separate_da_ll = os.path.isdir(self.seg_da_dir) and os.path.isdir(self.seg_ll_dir)

        files: List[str] = []
        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(glob(os.path.join(self.img_dir, e)))
        files.sort()
        if not files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        self.items: List[Tuple[str, str]] = []
        for ip in files:
            stem = os.path.splitext(os.path.basename(ip))[0]
            if self.use_separate_da_ll:
                da_p = self._find_mask(self.seg_da_dir, stem)
                ll_p = self._find_mask(self.seg_ll_dir, stem)
                if da_p and ll_p:
                    self.items.append((ip, stem))
            else:
                seg_p = self._find_mask(self.seg_dir, stem)
                if seg_p:
                    self.items.append((ip, seg_p))
        if not self.items:
            raise FileNotFoundError("No matching image/mask pairs found.")

    def _find_mask(self, base: str, stem: str) -> Optional[str]:
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = os.path.join(base, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.items)

    def _load_image(self, p: str) -> Image.Image:
        img = Image.open(p).convert("RGB")
        if self.img_size is not None:
            img = img.resize(self.img_size, Image.BILINEAR)
        return img

    def _load_seg(self, p: str) -> np.ndarray:
        im = Image.open(p)
        if self.img_size is not None:
            im = im.resize(self.img_size, Image.NEAREST)
        if im.mode in ("P", "L", "I"):
            return np.array(im)
        return np.array(im.convert("L"))

    def __getitem__(self, idx: int):
        ip, seg_info = self.items[idx]
        name = os.path.splitext(os.path.basename(ip))[0]

        img = np.array(self._load_image(ip), dtype=np.uint8)  # H,W,3

        if self.use_separate_da_ll:
            da_path = self._find_mask(self.seg_da_dir, name)
            ll_path = self._find_mask(self.seg_ll_dir, name)
            if da_path is None or ll_path is None:
                raise FileNotFoundError(f"Missing da/ll mask for {name}")
            da = self._load_seg(da_path)
            ll = self._load_seg(ll_path)
            da = (da > 127).astype(np.uint8) if da.max() > 1 else da.astype(np.uint8)
            ll = (ll > 127).astype(np.uint8) if ll.max() > 1 else ll.astype(np.uint8)
        else:
            seg_path = seg_info
            seg = self._load_seg(seg_path)
            da = np.isin(seg, self.da_ids).astype(np.uint8)
            ll = np.isin(seg, self.ll_ids).astype(np.uint8) if self.ll_ids else np.zeros_like(seg, dtype=np.uint8)

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [3,H,W], uint8
        target = np.stack([da, ll], axis=0).astype(np.int64)         # [2,H,W]
        target_t = torch.from_numpy(target)
        return name, img_t, target_t