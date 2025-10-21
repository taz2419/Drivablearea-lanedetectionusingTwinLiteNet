import torch
from model import TwinLite as net
from argparse import ArgumentParser
from utils import val, netParams
from const import *
import DataSet as myDataLoader  


def validation(args):
    '''
    Main function for training and validation
    :param args: global arguments
    :return: None
    '''
    # Select device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build model
    model = net.TwinLiteNet()

    # Load weights mapped to the current device (CPU-safe)
    print(f'Loading weights from: {args.weight}')
    checkpoint = torch.load(args.weight, map_location=device)

    #the next two section are dedicated to load models on CPU only device 
    #The error occurs because the pre-trained weights were saved with DataParallel 
    #(which adds a "module." prefix to all layer names), but you're loading them into 
    # a regular model (without DataParallel).

    #Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    #Removing 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k #remove `module.` prefix
        new_state_dict[name] = v


    #Loading cleaned state dict
    model.load_state_dict(new_state_dict)

    # Move model to device
    model = model.to(device)

    # Use DataParallel and cuDNN only on CUDA with multiple GPUs
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    model.eval()

    # DataLoader (pin_memory only helps on CUDA)
    pin_mem = (device.type == 'cuda')
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=pin_mem)
    
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # JIT trace on the selected device
    example = torch.rand(1, 3, 360, 640, device=device)
    model = torch.jit.trace(model, example)

    # Run validation (pass device so utils.val doesn’t call .cuda())
    da_segment_results, ll_segment_results = val(valLoader, model, device=device)

    # Print results
    msg = 'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
          'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
              da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
              ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
    print(msg)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', default="pretrained/best.pth", help='Path to pre-trained weights')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads (use 0-2 for CPU)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (reduce for CPU or limited GPU memory)')

    validation(parser.parse_args())


