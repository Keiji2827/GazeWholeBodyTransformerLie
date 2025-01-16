"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image

python ./metro/tools/end2end_inference_bodymesh.py 
       --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
       --image_file_or_path ./samples/human-body
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import time
import datetime
import torch
#import cv2
from torch.utils.data import DataLoader
from models.dataloader.gafa_loader import create_gafa_dataset
from models.utils.logger import setup_logger
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import set_seed

from torchvision import transforms



class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        l2 = torch.linalg.norm(outputs, ord=2, axis=1)
        outputs = outputs/l2[:,None]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        cos =  torch.sum(outputs*targets,dim=-1)
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss

def run_test(args, test_dataloader):

    print("len of dataset:", len(test_dataloader))

    #_gaze_network.eval()

    criterion_mse = CosLoss().cuda(args.device)
        
    val = run_validate(args, test_dataloader, 
                        criterion_mse
                        )

    print(args.dataset)
    print("test:", val)

def run_validate(args, val_dataloader, criterion_mse):
    log_losses = AverageMeter()

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    max_iter = len(val_dataloader)

    with torch.no_grad():        
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            #_gaze_network.eval()

            image = batch["image"].cuda(args.device)

            gaze_dir = batch["gaze_dir"].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)

            #batch_imgs = image
            batch_size = gaze_dir.size(0)
            data_time.update(time.time() - end)

            # forward-pass
            #direction = _gaze_network(batch_imgs,batch_images, smpl, mesh_sampler)
            #print(direction.shape)

            loss = criterion_mse(head_dir,gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if(iteration%20==0):
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    ' '.join(
                    ['eta: {eta}', 'iter: {iter}']
                    ).format(eta=eta_string, iter=iteration)
                    + ", loss:{:.4f}".format(log_losses.avg) 
                )

    return log_losses.avg

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--model_checkpoint", default='output/checkpoint-6-54572/state_dict.bin', type=str, required=False,
                        help="Path to wholebodygaze checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument('--dataset', type=str, nargs='*', default="", 
                        help="use test scene.")

    args = parser.parse_args()
    return args

# 最初はここから
def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    # default='output/'
    #mkdir(args.output_dir)
    logger = setup_logger("WholeBodyGaze Test", args.output_dir, 0)
    # randomのシード
    # default=88
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh


    # Load pretrained model
    # --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))
    logger.info("Run Test")

    logger.info("Run test")
    exp_names = [
        'library/1029_2', #
        'lab/1013_2',
        'kitchen/1022_2',
        'living_room/006',
        'courtyard/002',
        'courtyard/003',
    ]

    if args.dataset:
        exp_names = args.dataset

    dset = create_gafa_dataset(n_frames=1 ,exp_names=exp_names)
    test_dataloader = DataLoader(
        #dset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
        dset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True
    )

    run_test(args, test_dataloader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
