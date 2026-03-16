import argparse
import numpy as np
import torch

from eval_jux import Solver as eval_jux_solver


# set the random seed manually for reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser()


parser.add_argument('--device', default='cuda:5', type=str,
                    help='cpu/cuda')

# training hyperparameters
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size')


# saving directories and checkpoint/sample iterations
parser.add_argument('--max_iter', default=0, type=float,
                    help='maximum number of batch iterations')
parser.add_argument('--ckpt_dir', default='ckpts', type=str)

# Dataset options

parser.add_argument('--dt', default=0.5, type=float)
parser.add_argument('--obs_len', default=4, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--dataset_dir', default='', type=str, help='dataset directory')
parser.add_argument('--dataset_name', default='nuScenes', type=str,
                    help='dataset name')
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--heatmap_size', default=256, type=int)


# Macro
parser.add_argument('--num_goal', default=3, type=int)


# Diffusion Micro
parser.add_argument('--diffusion_micro_cfg_id', default='nuscenes_micro_diffuse', type=str)

# Evaluation
parser.add_argument('--n_w', default=10, type=int)
parser.add_argument('--n_z', default=1, type=int)

parser.add_argument('--sanity_check', default=False, type=bool)

args = parser.parse_args()

solver = eval_jux_solver(args)
solver.load_checkpoint()
muse_outputs, diffuse_outputs, guided_outputs, gt_futures = solver.all_compute(sanity_check=args.sanity_check)


torch.save(muse_outputs, 'output/nuscenes/normal_10/muse_outputs.pt')
torch.save(diffuse_outputs, 'output/nuscenes/normal_10/diffuse_outputs.pt')
torch.save(guided_outputs, 'output/nuscenes/normal_10/guided_outputs.pt')
torch.save(gt_futures, 'output/nuscenes/normal_10/gt_futures.pt')