from argparse import ArgumentParser, Namespace
import os
import glob
import torch
from torch.utils.data import DataLoader

from model import TwinLite as net
from utils import val as run_val, custom_collate_fn, netParams, LOGGER
import DataSet as BDD
from datasets.custom_dataset import CustomTwinLiteDataset

def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
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

def _detect_custom_dataset_root():
    # Look for any input that has images/, da_masks/, ll_masks/
    base = "/kaggle/input"
    for d in sorted(os.listdir(base)):
        root = os.path.join(base, d)
        if not os.path.isdir(root):
            continue
        if all(os.path.isdir(os.path.join(root, x)) for x in ("images", "da_masks", "ll_masks")):
            return root
    return None

def _detect_bdd_root():
    # Heuristic: pick an input whose name contains "bdd" or has a "bdd100k" folder
    base = "/kaggle/input"
    for d in sorted(os.listdir(base)):
        root = os.path.join(base, d)
        if not os.path.isdir(root):
            continue
        if "bdd" in d.lower() or os.path.isdir(os.path.join(root, "bdd100k")):
            # Export for DataSet.py to consume if it supports env-based roots
            os.environ.setdefault("BDD100K_ROOT", root)
            return root
    return None

def build_loader(args):
    if args.dataset == "bdd100k":
        ds = BDD.MyDataset(valid=True)
    elif args.dataset == "custom":
        ds = CustomTwinLiteDataset(
            img_dir=args.img_dir,
            da_mask_dir=args.da_mask_dir,
            ll_mask_dir=args.ll_mask_dir,
            img_size=(args.img_w, args.img_h) if (args.img_w and args.img_h) else None
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=custom_collate_fn
    )

def build_args_kaggle() -> Namespace:
    # Find weights anywhere under /kaggle/input
    weights = _find_first([
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
    ]) or "pretrained/best.pth"

    # Prefer custom dataset if present, else BDD
    custom_root = _detect_custom_dataset_root()
    bdd_root = _detect_bdd_root()

    if custom_root:
        return Namespace(
            dataset="custom",
            weights=weights,
            batch_size=16,
            num_workers=2,
            cpu=False,
            da_classes=2,
            ll_classes=2,
            img_dir=os.path.join(custom_root, "images"),
            da_mask_dir=os.path.join(custom_root, "da_masks"),
            ll_mask_dir=os.path.join(custom_root, "ll_masks"),
            img_w=0, img_h=0,
        )

    # Fall back to BDD if found
    if bdd_root:
        return Namespace(
            dataset="bdd100k",
            weights=weights,
            batch_size=16,
            num_workers=2,
            cpu=False,
            da_classes=2,
            ll_classes=2,
            img_dir="", da_mask_dir="", ll_mask_dir="",
            img_w=0, img_h=0,
        )

    # If nothing detected, still return defaults (will likely error until inputs are added)
    return Namespace(
        dataset="bdd100k",
        weights=weights,
        batch_size=16,
        num_workers=2,
        cpu=False,
        da_classes=2,
        ll_classes=2,
        img_dir="", da_mask_dir="", ll_mask_dir="",
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

    da_iou = da_metric.meanIntersectionOverUnion()
    ll_iou = ll_metric.meanIntersectionOverUnion()
    da_acc = da_metric.pixelAccuracy()
    ll_acc = ll_metric.pixelAccuracy()
    print(f"DA  - mIoU: {da_iou:.4f}, Acc: {da_acc:.4f}")
    print(f"Lane- mIoU: {ll_iou:.4f}, Acc: {ll_acc:.4f}")

if __name__ == "__main__":
    if _is_kaggle():
        args = build_args_kaggle()
        main(args)
    else:
        p = ArgumentParser()
        p.add_argument("--dataset", choices=["bdd100k", "custom"], default="bdd100k")
        p.add_argument("--weights", type=str, default="pretrained/best.pth")
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--num_workers", type=int, default=2)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--da_classes", type=int, default=2)
        p.add_argument("--ll_classes", type=int, default=2)
        p.add_argument("--img_dir", type=str, default="data/custom/images")
        p.add_argument("--da_mask_dir", type=str, default="data/custom/da_masks")
        p.add_argument("--ll_mask_dir", type=str, default="data/custom/ll_masks")
        p.add_argument("--img_w", type=int, default=0)
        p.add_argument("--img_h", type=int, default=0)
        main(p.parse_args())