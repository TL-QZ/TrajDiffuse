import argparse
import numpy as np
import torch

from eval_all import Solver as eval_solver


# set the random seed manually for reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser()


parser.add_argument('--device', default='cuda:6', type=str,
                    help='cpu/cuda')

# training hyperparameters
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')


# saving directories and checkpoint/sample iterations
parser.add_argument('--max_iter', default=0, type=float,
                    help='maximum number of batch iterations')
parser.add_argument('--ckpt_dir', default='ckpts', type=str)

# Dataset options
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--dt', default=0.4, type=float)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--dataset_dir', default='datasets/pfsd', type=str, help='dataset directory')
parser.add_argument('--dataset_name', default='pfsd', type=str,
                    help='dataset name')
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--heatmap_size', default=160, type=int)


# Macro
parser.add_argument('--num_goal', default=3, type=int)


# Diffusion Micro
parser.add_argument('--diffusion_micro_cfg_id', default='pfsd_micro_diffuse', type=str)

# Evaluation
parser.add_argument('--n_w', default=20, type=int)
parser.add_argument('--n_z', default=1, type=int)

parser.add_argument('--sanity_check', default=False, type=bool)
parser.add_argument('--to_eval', default='all', type=str)



args = parser.parse_args()

if args.to_eval == 'all':
    all, muse, diffuse, guide = True, True, True, True
elif args.to_eval == 'quick':
    all, muse, diffuse, guide = False, True, True, False
elif args.to_eval == 'guide':
    all, muse, diffuse, guide = False, False, False, True
else:
    raise ValueError("mode error, available mode: [all, quick, guide]")


solver = eval_solver(args)
solver.load_checkpoint()
outputs = solver.all_compute(sanity_check=args.sanity_check, to_eval=args.to_eval)

if args.to_eval == 'all':
    muse_outputs, diffuse_outputs, guided_outputs, gt_futures = outputs
    torch.save(muse_outputs, 'output/pfsd/normal/muse_outputs.pt')
    torch.save(diffuse_outputs, 'output/pfsd/normal/diffuse_outputs.pt')
    torch.save(guided_outputs, 'output/pfsd/normal/guided_outputs.pt')
    torch.save(gt_futures, 'output/pfsd/normal/gt_futures.pt')

elif args.to_eval == 'quick':
    muse_outputs, diffuse_outputs, gt_futures = outputs
    torch.save(muse_outputs, 'output/pfsd/normal/muse_outputs.pt')
    torch.save(diffuse_outputs, 'output/pfsd/normal/diffuse_outputs.pt')
    torch.save(gt_futures, 'output/pfsd/normal/gt_futures.pt')

elif args.to_eval == 'guide':
    guided_outputs, gt_futures = outputs
    torch.save(guided_outputs, 'output/pfsd/normal/guided_outputs.pt')
    torch.save(gt_futures, 'output/pfsd/normal/gt_futures.pt')