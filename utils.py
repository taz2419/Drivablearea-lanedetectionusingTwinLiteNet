import logging
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from typing import Tuple
from IOUEval import SegmentationMetric

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

LOGGING_NAME="validator"
def set_logging(name=LOGGING_NAME, verbose=True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)

def netParams(model) -> int:
    return sum(p.numel() for p in model.parameters())

def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, img, target = batch
            return img, target
        if len(batch) == 2:
            img, target = batch
            return img, target
    raise RuntimeError("Unexpected batch structure.")

@torch.no_grad()
def val(val_loader, model, device, num_classes_da=2, num_classes_ll=2):
    model.eval()
    da_metric = SegmentationMetric(num_classes_da)
    ll_metric = SegmentationMetric(num_classes_ll)

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if batch is None:
            continue
        img, target = _unpack_batch(batch)

        img = img.to(device).float() / 255.0
        target = target.to(device).long()
        da_target = target[:, 0]
        ll_target = target[:, 1]

        da_logits, ll_logits = model(img)
        da_pred = torch.argmax(da_logits, dim=1)
        ll_pred = torch.argmax(ll_logits, dim=1)

        da_metric.addBatch(da_pred.cpu().numpy().astype(np.int64),
                           da_target.cpu().numpy().astype(np.int64))
        ll_metric.addBatch(ll_pred.cpu().numpy().astype(np.int64),
                           ll_target.cpu().numpy().astype(np.int64))
    return da_metric, ll_metric
