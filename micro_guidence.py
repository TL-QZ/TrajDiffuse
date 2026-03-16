from collections import namedtuple
from torch.distributions import kl_divergence
import torch.optim as optim
from util import *
from model import *
from data.loader import data_loader
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np
import copy
import cv2
from tqdm import tqdm

from diffusion_models.temporal import TemporalUnet
from diffusion_models.diffusion import GaussianDiffusion, ValueDiffusion
from diffusion_models.value_function import ValueFunction


class Solver(object):

    def __init__(self, args):
        self.args = args
        self.name = '%s_%s_z_%s_enc_hD_%s_dec_hD_%s_mlpD_%s_map_featD_%s_map_mlpD_%s_lr_%s_klw_%s_fb_%s_scale_%s_n_goal_%s_run_%s' % \
                    (args.dataset_name, args.model_name, args.z_dim, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.map_mlp_dim,
                     args.lr, args.kl_weight, args.fb, args.scale, args.num_goal, args.run_id)

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len


        self.sg_idx = np.array(range(self.pred_len))
        self.sg_idx = np.flip(self.pred_len-1-self.sg_idx[::(self.pred_len//args.num_goal)])

        self.max_iter = int(args.max_iter)
        self.scale = args.scale
        self.lr = args.lr

        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        self.ckpt_load_iter = args.ckpt_load_iter

        self.log_freq = 1498/2

        mkdirs(self.ckpt_dir)

        diffuse_cfg = Config(args.diffusion_micro_cfg_id, False, create_dirs=False)

        self.map_size = tuple(diffuse_cfg.map_size)

        # NOTE: at this stage these following steps to involve the unconditional diffusion is not necessary
        # we will preceed with  training the micro-guidence model by train on noised forward process

        # # unconditional diffusion
        # model = TemporalUnet(horizon=diffuse_cfg.horizon, transition_dim=diffuse_cfg.transition_dim, 
        #                 cond_dim=diffuse_cfg.cond_dim, dim=diffuse_cfg.dim, 
        #                 dim_mults=diffuse_cfg.dim_mults, attention=diffuse_cfg.attention).to(self.device)

        # self.uncond_diffusion = GaussianDiffusion(model=model,
        #                 horizon=diffuse_cfg.horizon, observation_dim=diffuse_cfg.observation_dim, 
        #                 action_dim=diffuse_cfg.action_dim,
        #                 n_timesteps=diffuse_cfg.n_time_steps, loss_type=diffuse_cfg.loss_type, 
        #                 clip_denoised=diffuse_cfg.clip_denoised,
        #                 predict_epsilon=diffuse_cfg.predict_epsilon, action_weight=diffuse_cfg.action_weight, 
        #                 loss_discount=diffuse_cfg.loss_discount,
        #                 loss_weights=diffuse_cfg.loss_weights).to(self.device)
        # self.load_checkpoint()

        valueFunc = ValueFunction(**diffuse_cfg.yml_dict).to(self.device)
        self.model = ValueDiffusion(model=valueFunc, 
                            horizon=diffuse_cfg.horizon, 
                            observation_dim=diffuse_cfg.observation_dim, 
                            action_dim=diffuse_cfg.action_dim,
                            n_timesteps=diffuse_cfg.n_time_steps, 
                            loss_type=diffuse_cfg.value_loss_type).to(self.device)


        #  TODO uncomment this when valueFunc is ready

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(diffuse_cfg.value_learning_rate))

        self.ema = EMA(beta=diffuse_cfg.value_ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = diffuse_cfg.value_update_ema_every
        self.step_start_ema  = diffuse_cfg.value_step_start_ema

        # dataset loading
        print('Start loading data...')
        if self.dataset_name == 'nuScenes':
            cfg = Config('nuscenes_train', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)
            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.val_loader = data_generator(cfg, log, split='test', phase='testing',
                                             batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)
            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.idx_list))
            )
        else:
            _, self.train_loader = data_loader(self.args, 'train', shuffle=True)
            _, self.val_loader = data_loader(self.args, 'val', shuffle=True)
            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')
    

    def reset_parameters(self):
            self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self):
        if self.step_in_epoch < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.ckpt_dir,'iter_%s_state.pt' % epoch)
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)


    def train(self):

        data_loader = self.train_loader
        if self.dataset_name == 'nuScenes':
            iter_per_epoch = len(data_loader.idx_list)
        else:
            iterator = iter(data_loader)
            iter_per_epoch = len(iterator)
        start_iter = 1
        epoch = int(start_iter / iter_per_epoch)
        self.step_in_epoch = 0

        for iteration in tqdm(range(start_iter, self.max_iter + 1)):
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch +=1
                self.step_in_epoch = 0
                if self.dataset_name == 'nuScenes':
                    data_loader.is_epoch_end()
                else:
                    iterator = iter(data_loader)

            if self.dataset_name == 'nuScenes':
                data = data_loader.next_sample()
                if data is None:
                    continue
            else:
                data = next(iterator)

            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             map_info, inv_h_t,
             local_map, local_ic, local_homo) = data

            # Standardize the local_ic to [-1, 1] range by just specifies a scale of max(256, max_location_in_this_batch) 
            trajectories = torch.from_numpy(local_ic).to(torch.float32)
            local_ic_st = trajectories - trajectories[:,self.obs_len-1,:].unsqueeze(1)
            # std_scale = max(128.0, local_ic_st.abs().max())
            std_scale = 128.0
            local_ic_st /= std_scale
            local_ic_st = local_ic_st.to(self.device)

            condition = {}
            # Setup Condition, observations and the predicted waypoint and end goals
            for i in range(self.obs_len):
                condition[i] = local_ic_st[:,i,:]

            for i in range(len(local_map)):
                if local_map[i].shape != self.map_size:
                    local_map[i] = cv2.resize(local_map[i], self.map_size)
                local_map[i] = torch.from_numpy(local_map[i])            
            local_map = torch.stack(local_map).unsqueeze(1).to(self.device)
            batch = ValueBatch(local_ic_st, condition, local_map, trajectories[:,self.obs_len-1,:].unsqueeze(1).to(self.device), std_scale)

            loss, infos = self.model.loss(*batch)
            # Underconstruction below uncomment after value function being implemented

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.step_in_epoch % self.update_ema_every == 0:
                self.step_ema()
            
            if iteration % (iter_per_epoch*5) == 0:
                self.save(epoch)

            if  self.step_in_epoch % self.log_freq == 0:
                info_str = 'epoch {} iter {} loss {}'.format(epoch, self.step_in_epoch, loss.item())
                print(info_str, flush=True)
            
            self.step_in_epoch += 1



    def load_checkpoint(self):

        diffusion_path = os.path.join(
            self.ckpt_dir,
            'diffusion_weight.pt'
        )
        diffusion_dict = torch.load(diffusion_path)
        self.uncond_diffusion.load_state_dict(diffusion_dict['ema'])
        # self.micro_diffusion.load_state_dict(diffusion_dict['model'])
        print('ckpt loaded from ', self.ckpt_dir)