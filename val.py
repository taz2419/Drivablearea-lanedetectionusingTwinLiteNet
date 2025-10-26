from argparse import ArgumentParser, Namespace
import os, glob, importlib, inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import val as run_val, custom_collate_fn, netParams, LOGGER
from datasets.custom_dataset import CustomTwinLiteDataset

def _load_model_class():
    m = importlib.import_module("model.TwinLite")
    for name in ("TwinLite", "TwinLiteNet", "Net", "Model"):
        if hasattr(m, name) and inspect.isclass(getattr(m, name)) and issubclass(getattr(m, name), nn.Module):
            return getattr(m, name)
    for name, obj in inspect.getmembers(m, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == m.__name__:
            return obj
    raise ImportError("No nn.Module subclass found in model/TwinLite.py")
Net = _load_model_class()

def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    new_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing: LOGGER.warning(f"Missing keys: {missing}")
    if unexpected: LOGGER.warning(f"Unexpected keys: {unexpected}")
    return model

def _is_kaggle() -> bool:
    return os.path.isdir("/kaggle/input")

def _find_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

def _detect_cityscapes_root():
    base = "/kaggle/input"
    for d in sorted(os.listdir(base)):
        root = os.path.join(base, d)
        if not os.path.isdir(root):
            continue
        if d.lower() == "cityscapes_val":
            img_val = os.path.join(root, "images", "val")
            seg_val = os.path.join(root, "segments", "val")
            if os.path.isdir(img_val) and os.path.isdir(seg_val):
                return root
        # fallback: any input with images/val + segments/val
        img_val = os.path.join(root, "images", "val")
        seg_val = os.path.join(root, "segments", "val")
        if os.path.isdir(img_val) and os.path.isdir(seg_val):
            return root
    return None

def build_loader(args):
    ds = CustomTwinLiteDataset(
        root=args.custom_root,
        split=args.split,
        img_size=(args.img_w, args.img_h) if (args.img_w and args.img_h) else None,
        da_ids=args.da_ids,
        ll_ids=args.ll_ids,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=custom_collate_fn
    )

def _parse_ids_str(s: str) -> list[int]:
    s = (s or "").strip()
    return [int(x) for x in s.split(",") if x.strip().isdigit()]

def build_args_kaggle() -> Namespace:
    weights = _find_first(["/kaggle/input/**/*.pth", "/kaggle/input/**/*.pt"]) or "pretrained/best.pth"
    root = _detect_cityscapes_root()
    if not root:
        raise FileNotFoundError("cityscapes_val dataset not found under /kaggle/input (expected images/val and segments/val).")

    da_ids = _parse_ids_str(os.getenv("DA_IDS", "0"))  # default road=0
    ll_ids = _parse_ids_str(os.getenv("LL_IDS", ""))   # set if you have lane ids

    return Namespace(
        custom_root=root,
        split="val",
        weights=weights,
        batch_size=16,
        num_workers=2,
        cpu=False,
        da_classes=2,
        ll_classes=2,
        da_ids=da_ids,
        ll_ids=ll_ids,
        img_w=0, img_h=0,
    )

def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Net().to(device)
    LOGGER.info(f"Model params: {netParams(model):,}")

    if args.weights and os.path.isfile(args.weights):
        load_weights(model, args.weights, device)
        LOGGER.info(f"Loaded weights: {args.weights}")
    else:
        LOGGER.warning(f"Weights not found: {args.weights}")

    val_loader = build_loader(args)
    da_metric, ll_metric = run_val(
        val_loader, model, device,
        num_classes_da=args.da_classes, num_classes_ll=args.ll_classes
    )
    print(f"DA  - mIoU: {da_metric.meanIntersectionOverUnion():.4f}, Acc: {da_metric.pixelAccuracy():.4f}")
    print(f"Lane- mIoU: {ll_metric.meanIntersectionOverUnion():.4f}, Acc: {ll_metric.pixelAccuracy():.4f}")

if __name__ == "__main__":
    if _is_kaggle():
        main(build_args_kaggle())
    else:
        p = ArgumentParser()
        p.add_argument("--custom_root", type=str, default="data/cityscapes_val")
        p.add_argument("--split", type=str, default="val")
        p.add_argument("--weights", type=str, default="pretrained/best.pth")
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--num_workers", type=int, default=2)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--da_classes", type=int, default=2)
        p.add_argument("--ll_classes", type=int, default=2)
        p.add_argument("--da_ids", type=str, default="0")
        p.add_argument("--ll_ids", type=str, default="")
        p.add_argument("--img_w", type=int, default=0)
        p.add_argument("--img_h", type=int, default=0)
        args = p.parse_args()
        args.da_ids = [int(x) for x in args.da_ids.split(",") if x.strip().isdigit()]
        args.ll_ids = [int(x) for x in args.ll_ids.split(",") if x.strip().isdigit()]
        main(args)