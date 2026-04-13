from models.sam import SamPredictor, sam_model_registry
from models.sam_LoRa import LoRA_Sam
import numpy as np
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import Public_dataset
from utils.dsc import dice_coeff_multi_class
import monai
import json
import logging
from argparse import Namespace, ArgumentParser
from datetime import datetime


def parse_test_args():
    parser = ArgumentParser(description='Test script for SAM fine-tuning')
    parser.add_argument('-dir_checkpoint', type=str, required=True,
                        help='Path to the checkpoint directory (must contain args.json and checkpoint_best.pth)')
    parser.add_argument('-ckpt_name', type=str, default='checkpoint_best.pth',
                        help='Checkpoint filename to load (default: checkpoint_best.pth)')
    parser.add_argument('-gpu', type=int, default=0,
                        help='GPU device id to use (default: 0)')
    return parser.parse_args()


def run_test(args, ckpt_path):
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
    criterion2 = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Build model (mirrors the training script's model construction)
    # ------------------------------------------------------------------
    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls)
        sam.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls)
        sam = LoRA_Sam(args, sam, r=4).sam
        sam.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    else:
        raise ValueError(f'Unknown finetune_type: {args.finetune_type}')

    sam = sam.to('cuda')
    sam.eval()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.proposal_folder,
        args.val_img_list,
        phase='val',
        targets=[args.targets],
        normalize_type=args.normalize_type,
        if_prompt=False,
    )
    testloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=4)

    logging.info(f'Test set size: {len(test_dataset)} samples')

    # ------------------------------------------------------------------
    # Evaluation loop  (directly copied from training validate section)
    # ------------------------------------------------------------------
    eval_loss = 0
    dsc = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader, desc='Testing')):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask'])
            msks = msks.cuda()

            img_emb = sam.image_encoder(imgs)
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, mask_emb, iou_pred = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            loss = criterion1(pred, msks.float()) + criterion2(pred, torch.squeeze(msks.long(), 1))
            eval_loss += loss.item()

            dsc_batch = dice_coeff_multi_class(
                pred.argmax(dim=1).cpu(),
                torch.squeeze(msks.long(), 1).cpu().long(),
                args.num_cls,
            )
            dsc += dsc_batch

    eval_loss /= (i + 1)
    dsc /= (i + 1)

    logging.info('=' * 60)
    logging.info(f'Checkpoint : {ckpt_path}')
    logging.info(f'Test loss  : {eval_loss:.6f}')
    logging.info(f'Mean DSC   : {dsc:.6f}')
    logging.info('=' * 60)

    print(f'\nTest loss : {eval_loss:.6f}')
    print(f'Mean DSC  : {dsc:.6f}')


if __name__ == '__main__':
    test_args = parse_test_args()

    # ------------------------------------------------------------------
    # Set GPU
    # ------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = str(test_args.gpu)

    # ------------------------------------------------------------------
    # Load training args from checkpoint directory
    # ------------------------------------------------------------------
    args_path = os.path.join(test_args.dir_checkpoint, 'args.json')
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    args = Namespace(**args_dict)

    # Override checkpoint dir in case the saved path differs from current
    args.dir_checkpoint = test_args.dir_checkpoint

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

    # ------------------------------------------------------------------
    # Resolve checkpoint file and test image list
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(test_args.dir_checkpoint, test_args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    logging.info(f'Val img list   : {args.val_img_list}')
    logging.info(f'Dataset        : {args.dataset_name}')
    logging.info(f'Finetune type  : {args.finetune_type}')
    logging.info(f'Arch           : {args.arch}')
    logging.info(f'Num classes    : {args.num_cls}')

    run_test(args, ckpt_path)
