import torch
from model import TwinLite as net
from argparse import ArgumentParser
from utils import val, netParams
from const import *
import DataSet as myDataLoader  
from collections import OrderedDict


def validation(args):
    '''
    Main function for validation
    :param args: global arguments
    :return: None
    '''
    # Select device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Build model
    print("Building model...")
    model = net.TwinLiteNet()
    model = model.to(device)
    
    # Load pretrained weights
    if args.weight:
        print(f"Loading weights from {args.weight}")
        checkpoint = torch.load(args.weight, map_location=device)
         # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix from DataParallel checkpoints
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.' prefix (7 characters)
            else:
                name = k
            new_state_dict[name] = v
        
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict)
        print(f"✓ Successfully loaded weights")
    
    print(f"Total parameters: {netParams(model):,}")
    
    # Prepare data loader
    print("Loading validation dataset...")
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"Total validation samples: {len(valLoader.dataset)}")
    
    # Run validation
    print("Starting validation...")
    da_segment_results, ll_segment_results = val(valLoader, model, device)
    
    # Print results
    print("\n" + "="*50)
    print("Validation Results:")
    print("="*50)
    print("\nDrivable Area Segmentation:")
    print(f"  IoU: {da_segment_results.IntersectionOverUnion():.4f}")
    print(f"  Accuracy: {da_segment_results.Accuracy():.4f}")
    
    print("\nLane Line Segmentation:")
    print(f"  IoU: {ll_segment_results.IntersectionOverUnion():.4f}")
    print(f"  Accuracy: {ll_segment_results.Accuracy():.4f}")
    print("="*50)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='', help='Path to pretrained weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    
    args = parser.parse_args()
    validation(args)