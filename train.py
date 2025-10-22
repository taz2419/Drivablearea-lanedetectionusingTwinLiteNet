import os
import torch
from argparse import ArgumentParser
import DataSet as myDataLoader
from utils import val
from model import TwinLite as net

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = net.TwinLiteNet()
    model = model.to(device)

    # AMP for GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Dataloaders (configurable root)
    train_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=False, data_root=args.data_root),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True, data_root=args.data_root),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = YourLossClass(...)  # use your existing loss

    for epoch in range(args.epochs):
        model.train()
        for _, inputs, target in train_loader:
            inputs = inputs.to(device, non_blocking=(device.type == 'cuda')).float() / 255.0
            seg_da, seg_ll = target
            seg_da = seg_da.to(device, non_blocking=(device.type == 'cuda')).float()
            seg_ll = seg_ll.to(device, non_blocking=(device.type == 'cuda')).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_da, out_ll = model(inputs)
                loss = criterion(out_da, out_ll, seg_da, seg_ll)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # validate
        da_res, ll_res = val(val_loader, model, device=device)
        # ...save best to /kaggle/working...

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None, help='Root with images/, segments/, lane/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main()

