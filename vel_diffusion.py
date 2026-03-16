from collections import namedtuple
from torch.distributions import kl_divergence
import torch.optim as optim
import torch.nn.functional as nnf
from util import *
from model import *
from data.loader import data_loader
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np
import copy
import random

from tqdm import tqdm

from diffusion_models.temporal import TemporalUnet
from diffusion_models.diffusion import GaussianDiffusion

from torch.utils.tensorboard import SummaryWriter


class Solver(object):

    def __init__(self, args):

        self.args = args
        self.name = '%s_%s_run_%s' % \
                    (args.dataset_name, args.model_name, args.run_id)

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

        self.use_sg_net = args.use_sg_net
        if self.use_sg_net:
            sg_unet_path = os.path.join(args.ckpt_dir, "pretrained_models_" + args.dataset_name, 'sg_net.pt')
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu').to(self.device)
            self.sg_unet.eval()
            hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=self.sg_idx, device=self.device)
            self.make_heatmap = hg.make_heatmap



        # model and diffusion initialization
        diffuse_cfg = Config(args.diffusion_micro_cfg_id, False, create_dirs=False)

        model = TemporalUnet(horizon=diffuse_cfg.horizon, transition_dim=diffuse_cfg.transition_dim, 
                        cond_dim=diffuse_cfg.cond_dim, dim=diffuse_cfg.dim, 
                        dim_mults=diffuse_cfg.dim_mults, attention=diffuse_cfg.attention).to(self.device)

        self.model = GaussianDiffusion(model=model,
                        horizon=diffuse_cfg.horizon, observation_dim=diffuse_cfg.observation_dim, 
                        action_dim=diffuse_cfg.action_dim,
                        n_timesteps=diffuse_cfg.n_time_steps, loss_type=diffuse_cfg.loss_type, 
                        clip_denoised=diffuse_cfg.clip_denoised,
                        predict_epsilon=diffuse_cfg.predict_epsilon, action_weight=diffuse_cfg.action_weight, 
                        loss_discount=diffuse_cfg.loss_discount,
                        loss_weights=diffuse_cfg.loss_weights).to(self.device)
        self.action_dim = diffuse_cfg.action_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(diffuse_cfg.learning_rate))


        self.ema = EMA(beta=diffuse_cfg.ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = diffuse_cfg.update_ema_every
        self.step_start_ema  = diffuse_cfg.step_start_ema
        self.step_in_epoch = 0

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
            self.val_loader = data_generator(cfg, log, split='val', phase='testing',
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


    def train(self):
        writer = SummaryWriter(self.ckpt_dir)

        data_loader = self.train_loader
        if self.dataset_name == 'nuScenes':
            iter_per_epoch = len(data_loader.idx_list)
        else:
            iterator = iter(data_loader)
            iter_per_epoch = len(iterator)

        start_iter = 1
        epoch = int(start_iter / iter_per_epoch)

        epoch_traj_loss = 0.0

        for iteration in tqdm(range(start_iter, self.max_iter + 1)):
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                # summerize epoch loss for tensorboard
                writer.add_scalar('epoch_traj_loss', epoch_traj_loss / iter_per_epoch, epoch)
                epoch_traj_loss = 0.0

                epoch +=1
                self.step_in_epoch = 0

                # reset dataloader
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

            self.step_in_epoch += 1

            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, c,
             map_info, inv_h_t,
             local_map, local_ic, local_homo) = data
            
            batch_size = obs_traj.size(1)
            pred_sg_ic_list = None
            if self.use_sg_net:
                # 20% chance using sg_net output rather than ground truth
                if random.uniform(0,1) < 0.2:
                    obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map, aug=False)
                    recon_sg_heat = self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map], dim=1))
                    pred_sg_heat = F.sigmoid(recon_sg_heat)
                    pred_sg_ic_list = []
                    for i in range(batch_size):
                        pred_sg_ic = []
                        if self.dataset_name == 'pfsd':
                            map_size = local_map[i].shape
                            for heat_map in pred_sg_heat[i]:
                                argmax_idx = heat_map.argmax()
                                argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                                pred_sg_ic.append(argmax_idx)
                        else:
                            map_size = local_map[i].shape
                            for heat_map in pred_sg_heat[i]:
                                heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                        size=map_size, mode='bicubic',
                                                        align_corners=False).squeeze(0).squeeze(0)
                                argmax_idx = heat_map.argmax()
                                argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                                pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                        pred_sg_ic_list.append(pred_sg_ic)
                    pred_sg_ic_list = torch.stack(pred_sg_ic_list)
                    
            # Standardize the local_ic to [-1, 1] range by just specifies a scale of max(256, max_location_in_this_batch) 
            trajectories = torch.from_numpy(local_ic).to(torch.float32)
            std_scale = []
            local_ic_st = []
            normed_pred_sg_ic = []
            for b in range(batch_size):
                map_size = local_map[b].shape
                local_ic_st.append((trajectories[b]-map_size[0]/2)/(map_size[0]/2))
                if pred_sg_ic_list is not None:
                    normed_pred_sg_ic.append((pred_sg_ic_list[b]-map_size[0]/2)/(map_size[0]/2))
                std_scale.append(map_size[0]/2)
            local_ic_st = torch.stack(local_ic_st).to(self.device)
            if pred_sg_ic_list is not None:
                normed_pred_sg_ic = torch.stack(normed_pred_sg_ic).to(self.device)

            local_ic_st_array = local_ic_st.clone().cpu().detach().numpy()
            local_vel = np.zeros_like(local_ic_st_array)
            for b in range(batch_size):
                local_vel[b,:,0] = derivative_of(local_ic_st_array[b,:,0], dt=self.dt)
                local_vel[b,:,1] = derivative_of(local_ic_st_array[b,:,1], dt=self.dt)
            local_vel = torch.from_numpy(local_vel).to(torch.float32).to(self.device)
            train_input = torch.cat([local_vel, local_ic_st],dim=-1)

            condition = {}
            # Setup Condition, observations and the predicted waypoint and end goals
            for i in range(self.obs_len):
                condition[i] = train_input[:,i,:]
            
            # note in this setting local_ic_st dim is `self.obs_len+self.pred_len`
            if pred_sg_ic_list is not None:
                for idx, i in enumerate(self.sg_idx):
                    cond_idx = i+self.obs_len
                    condition[cond_idx] = normed_pred_sg_ic[:, idx,:]
            else:
                for i in self.sg_idx:
                    cond_idx = i+self.obs_len
                    condition[cond_idx] = local_ic_st[:, cond_idx,:]

            # Setup Batch
            batch = Batch(train_input, condition)

            loss, infos = self.model.loss(*batch)
            epoch_traj_loss += loss.item()
            writer.add_scalar('traj_loss', loss.item(), iteration)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if iteration % self.update_ema_every == 0:
                self.step_ema()

            if iteration % (iter_per_epoch*5) == 0:
                self.save(epoch)
            
            if iteration % (iter_per_epoch*5) == 0:
                self.validate(epoch, writer)

    def validate(self, epoch:int, writer) -> None:

        data_loader = self.val_loader
        if self.dataset_name == 'nuScenes':
            iter_per_epoch = len(data_loader.idx_list)
        else:
            iterator = iter(data_loader)
            iter_per_epoch = len(iterator)

        start_iter = 0
        local_de = []
        global_de = []
        for iteration in tqdm(range(start_iter, iter_per_epoch )):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)

                # reset dataloader
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
            
            batch_size = obs_traj.size(1)

            # Standardize the local_ic to [-1, 1] range by just specifies a scale of max(256, max_location_in_this_batch) 
            trajectories = torch.from_numpy(local_ic).to(torch.float32)
            std_scale = []
            local_ic_st = []
            for b in range(batch_size):
                map_size = local_map[b].shape
                local_ic_st.append((trajectories[b]-map_size[0]/2)/(map_size[0]/2))
                std_scale.append(map_size[0]/2)
            local_ic_st = torch.stack(local_ic_st).to(self.device)
            local_ic_st_array = local_ic_st.clone().cpu().detach().numpy()
            local_vel = np.zeros_like(local_ic_st_array)
            for b in range(batch_size):
                local_vel[b,:,0] = derivative_of(local_ic_st_array[b,:,0], dt=self.dt)
                local_vel[b,:,1] = derivative_of(local_ic_st_array[b,:,1], dt=self.dt)
            local_vel = torch.from_numpy(local_vel).to(torch.float32).to(self.device)
            local_cond = torch.cat([local_vel, local_ic_st],dim=-1)


            condition = {}
            # Setup Condition, observations and the predicted waypoint and end goals
            for i in range(self.obs_len):
                condition[i] = local_cond[:,i,:]
            
            # note in this setting local_ic_st dim is `self.obs_len+self.pred_len`
            for i in self.sg_idx:
                cond_idx = i+self.obs_len
                condition[cond_idx] = local_ic_st[:, cond_idx,:]

            samples = self.model(condition, verbose=False)
            vel_samples = samples[0][:,(self.obs_len):,:self.action_dim].permute(1, 0, 2)
            traj_recon = integrate_samples(vel_samples, local_ic_st[:,(self.obs_len-1),:], dt=self.dt)
            traj_recon = traj_recon.permute(1, 0, 2)
            # local_preds = []
            global_preds = []
            for b in range(batch_size):
                pred_local = traj_recon[b]*std_scale[b]+std_scale[b]
                
                back_wc = torch.matmul(
                    torch.cat([pred_local, torch.ones((len(pred_local), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[b].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                # local_preds.append(pred_local)
                global_preds.append(back_wc[:,:2])
            global_preds = torch.stack(global_preds).transpose(0,1)
            global_de.extend(displacement_error(global_preds, fut_traj[:, :, :2], mode='raw'))
        global_de = torch.stack(global_de)/self.pred_len
        global_ade = global_de.mean()
        writer.add_scalar('epoch_ade_global', global_ade, epoch)


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

            
            