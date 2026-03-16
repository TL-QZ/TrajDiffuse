from model import *
import torch.nn.functional as nnf

import matplotlib.pyplot as plt
from util import *
import numpy as np
import torch.nn.functional as F
from data.loader import data_loader
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator

from diffusion_models.temporal import TemporalUnet
from diffusion_models.diffusion import GaussianDiffusion, ValueDiffusion

from diffusion_models.value_function import ValueFunction


from diffusion_sampling.guides import MapGuide
from diffusion_sampling.functions import guided_micro_sampling, guided_excess_sampling

from tqdm import tqdm

class Solver(object):

    def __init__(self, args):
        self.image_counter = 0

        self.args = args
        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.heatmap_size = args.heatmap_size
        self.dataset_name = args.dataset_name
        self.scale = args.scale
        self.n_w = args.n_w
        self.n_z = args.n_z

        self.eps=1e-9
        self.sg_idx = np.array(range(self.pred_len))
        self.sg_idx = np.flip(self.pred_len-1-self.sg_idx[::(self.pred_len//args.num_goal)])

        self.ckpt_dir = os.path.join(args.ckpt_dir, "pretrained_models_" + args.dataset_name)

        if args.dataset_name == 'nuScenes':
            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.data_loader = data_generator(cfg, log, split='test', phase='testing',
                                         batch_size=args.batch_size, device=args.device, scale=args.scale,
                                         shuffle=False)
        else:
            _, self.data_loader = data_loader(self.args, 'test', shuffle=False)

        hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=None, device=self.device)
        self.make_heatmap = hg.make_heatmap
        self.make_one_heatmap = hg.make_one_heatmap

        # model and diffusion initialization
        diffuse_cfg = Config(args.diffusion_micro_cfg_id, False, create_dirs=False)


        model = TemporalUnet(horizon=diffuse_cfg.horizon, transition_dim=diffuse_cfg.transition_dim, 
                        cond_dim=diffuse_cfg.cond_dim, dim=diffuse_cfg.dim, 
                        dim_mults=diffuse_cfg.dim_mults, attention=diffuse_cfg.attention).to(self.device)

        self.micro_diffusion = GaussianDiffusion(model=model,
                        horizon=diffuse_cfg.horizon, observation_dim=diffuse_cfg.observation_dim, 
                        action_dim=diffuse_cfg.action_dim,
                        n_timesteps=diffuse_cfg.n_time_steps, loss_type=diffuse_cfg.loss_type, 
                        clip_denoised=diffuse_cfg.clip_denoised,
                        predict_epsilon=diffuse_cfg.predict_epsilon, action_weight=diffuse_cfg.action_weight, 
                        loss_discount=diffuse_cfg.loss_discount,
                        loss_weights=diffuse_cfg.loss_weights).to(self.device)
        
        self.guide = MapGuide(self.micro_diffusion).to(self.device)
        
        self.eval_kwarg = Config('guidence_eval', False, create_dirs=False)



    def all_evaluation(self):
        self.set_mode(train=False)
        muse_all_ade =[]
        muse_all_fde =[]
        muse_ECFL_sum = 0.0
        diffuse_all_ade = []
        diffuse_all_fde = []
        diffuse_ECFL_sum = 0.0
        total_num_instances = 0
        with torch.no_grad():
            if self.dataset_name == 'nuScenes':
                while not self.data_loader.is_epoch_end():
                    batch = self.data_loader.next_sample()
                    if batch is None:
                        continue
                    print('compute {} out of {}'.format(self.data_loader.index, len(self.data_loader.idx_list)))
                    muse_ade, muse_fde, muse_ECFL, diffuse_ade, diffuse_fde, diffuse_ECFL, num_instances = self.compute(batch)
                    muse_all_ade.append(muse_ade)
                    muse_all_fde.append(muse_fde)
                    muse_ECFL_sum += muse_ECFL
                    diffuse_all_ade.append(diffuse_ade)
                    diffuse_all_fde.append(diffuse_fde)
                    diffuse_ECFL_sum += diffuse_ECFL
                    total_num_instances += num_instances

            # else:
            #     for batch in self.data_loader:
            #         ade, fde = self.compute(batch)
            #         all_ade.append(ade)
            #         all_fde.append(fde)

        muse_all_ade = torch.cat(muse_all_ade, dim=1).cpu().numpy()
        muse_all_fde = torch.cat(muse_all_fde, dim=1).cpu().numpy()
        diffuse_all_ade = torch.cat(diffuse_all_ade, dim=1).cpu().numpy()
        diffuse_all_fde = torch.cat(diffuse_all_fde, dim=1).cpu().numpy()

        muse_min_ade = muse_all_ade.min(axis=0).mean()/self.pred_len
        muse_min_fde = muse_all_fde.min(axis=0).mean()

        diffuse_min_ade = diffuse_all_ade.min(axis=0).mean()/self.pred_len
        diffuse_min_fde = diffuse_all_fde.min(axis=0).mean()

        muse_ECFL = muse_ECFL_sum/total_num_instances
        diffuse_ECFL = diffuse_ECFL_sum/total_num_instances

        print('------------------------------------------')
        print('dataset name: ', self.dataset_name)
        print('muse_min_ade: ', muse_min_ade)
        print('muse_min_fde: ', muse_min_fde)
        print('muse_ECFL: ', muse_ECFL)
        print('diffuse_min_ade: ', diffuse_min_ade)
        print('diffuse_min_fde: ', diffuse_min_fde)
        print('diffuse_ECFL: ', diffuse_ECFL)
        print('------------------------------------------')


        return 0, 0


    def compute(self, batch):
        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)

        obs_heat_map, _ = self.make_heatmap(local_ic, local_map)

        self.lg_cvae.forward(obs_heat_map, None, training=False)
        
        pred_lg_wcs = []
        pred_sg_wcs = []
        pred_sg_ics = []

        ####### long term goals and the corresponding (deterministic) short term goals ########
        w_priors = []
        for _ in range(self.n_w):
            w_priors.append(self.lg_cvae.prior_latent_space.sample())

        lg_heatmap = []
        sg_heatmaps = []
        for w_prior in w_priors:
            # -------- long term goal --------
            pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

            pred_lg_wc = []
            pred_lg_ics = []
            for i in range(batch_size):
                map_size = local_map[i].shape
                pred_lg_ic = []
                if self.dataset_name == 'pfsd':
                    for heat_map in pred_lg_heat[i]:
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)
                else:
                    for heat_map in pred_lg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        if i == 0:
                            lg_heatmap.append(heat_map)

                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                pred_lg_ics.append(pred_lg_ic)
                back_wc = torch.matmul(
                    torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
            pred_lg_wc = torch.stack(pred_lg_wc)
            pred_lg_wcs.append(pred_lg_wc)

            # -------- short term goal --------
            pred_lg_heat_from_ic = []
            if self.dataset_name == 'pfsd':
                for coord in pred_lg_ics:
                    heat_map_traj = np.zeros((self.heatmap_size, self.heatmap_size))
                    heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    pred_lg_heat_from_ic.append(heat_map_traj)
            else:
                for i in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                        0].detach().cpu().numpy().astype(int)))
            pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                self.device)
            pred_sg_heat = F.sigmoid(
                self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

            pred_sg_wc = []
            pred_sg_ic_list = []
            sg_heatmap = []
            for i in range(batch_size):
                pred_sg_ic = []
                if self.dataset_name == 'pfsd':
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
                        if i == 0:
                            sg_heatmap.append(heat_map)

                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)

                pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                back_wc = torch.matmul(
                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_sg_wc.append(back_wc[:, :2])
                pred_sg_ic_list.append(pred_sg_ic)
            pred_sg_wc = torch.stack(pred_sg_wc)
            pred_sg_wcs.append(pred_sg_wc)
            pred_sg_ic_list = torch.stack(pred_sg_ic_list)
            pred_sg_ics.append(pred_sg_ic_list)

            sg_heatmap = torch.stack(sg_heatmap)
            sg_heatmaps.append(sg_heatmap)

        ##### trajectories per long&short goal ####
        # muse_pred_trajs: list len 5, of tensors with shape 12 x batch_size x 2
        # diffuse_global_pred, diffuse_local_pred: list len 5, of tensors with shape 16 x batch_size x 2
        fut_rel_pos_dists, muse_pred_trajs = self.muse_micro(obs_traj, obs_traj_st, seq_start_end, local_homo, pred_sg_wcs)
        diffuse_global_pred, diffuse_local_preds = self.diffuse_micro(batch_size, local_ic, local_homo, pred_sg_ics)

        # compute ADE FDE
        muse_ade, muse_fde = [], []
        diffuse_ade, diffuse_fde = [], []

        for m in range(len(w_priors)):
            muse_ade.append(displacement_error(
                muse_pred_trajs[m], fut_traj[:, :, :2], mode='raw'
            ))
            muse_fde.append(final_displacement_error(
                muse_pred_trajs[m][-1], fut_traj[-1, :, :2], mode='raw'
            ))
            diffuse_ade.append(displacement_error(
                diffuse_global_pred[m][self.obs_len:], fut_traj[:, :, :2], mode='raw'
            ))
            diffuse_fde.append(final_displacement_error(
                diffuse_global_pred[m][-1], fut_traj[-1, :, :2], mode='raw'
            ))
        muse_ade = torch.stack(muse_ade)
        muse_fde = torch.stack(muse_fde)
        diffuse_ade = torch.stack(diffuse_ade)
        diffuse_fde = torch.stack(diffuse_fde)

        # Reshape and compute ECFL
        muse_ECFL = 0.0
        diffuse_ECFL = 0.0
        muse_global_trajs = torch.stack(muse_pred_trajs).permute((2,0,1,3))
        diffuse_global_trajs = torch.stack(diffuse_global_pred)[:,self.obs_len:].permute((2,0,1,3))
        for i, rng_idx in enumerate(seq_start_end):
            map_obj = map_info[i]
            muse_ECFL += ECFL(muse_global_trajs[rng_idx[0]:rng_idx[1]], map_obj)
            diffuse_ECFL += ECFL(diffuse_global_trajs[rng_idx[0]:rng_idx[1]], map_obj)
        num_instances = muse_global_trajs.shape[0]*muse_global_trajs.shape[1]

        diffuse_local_collect = [diffuse_local_pred[self.obs_len:,0] for diffuse_local_pred in diffuse_local_preds]

        muse_local_trajs = muse_global_trajs[0].clone().cpu().numpy()
        muse_local_trajs = np.concatenate((muse_local_trajs, np.ones((len(w_priors), self.pred_len, 1))), axis=2)
        muse_local_trajs = np.matmul(muse_local_trajs, np.linalg.pinv(np.transpose(local_homo[0].cpu().numpy())))
        muse_local_trajs /= np.expand_dims(muse_local_trajs[:, :,2], 2)
        muse_local_trajs = torch.tensor(np.round(muse_local_trajs).astype(int)[:,:,:2])

        self.visualize(lg_heatmap, sg_heatmaps, muse_local_trajs, local_ic[0], local_map[0], 'muse')
        self.visualize(lg_heatmap, sg_heatmaps, diffuse_local_collect, local_ic[0], local_map[0], 'diffuse')
        self.image_counter += 1
        return muse_ade, muse_fde, muse_ECFL, diffuse_ade, diffuse_fde, diffuse_ECFL, num_instances


    def macro_stage(self, batch):
        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)

        obs_heat_map, _ = self.make_heatmap(local_ic, local_map)

        self.lg_cvae.forward(obs_heat_map, None, training=False)
        
        pred_lg_wcs = []
        pred_sg_wcs = []
        pred_sg_ics = []

        ####### long term goals and the corresponding (deterministic) short term goals ########
        w_priors = []
        for _ in range(self.n_w):
            w_priors.append(self.lg_cvae.prior_latent_space.sample())

        lg_heatmap = []
        sg_heatmaps = []

        for w_prior in w_priors:
            # -------- long term goal --------
            pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

            pred_lg_wc = []
            pred_lg_ics = []
            for i in range(batch_size):
                map_size = local_map[i].shape
                pred_lg_ic = []
                if self.dataset_name == 'pfsd':
                    for heat_map in pred_lg_heat[i]:
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)
                else:
                    for heat_map in pred_lg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        if i == 0:
                            lg_heatmap.append(heat_map)

                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                pred_lg_ics.append(pred_lg_ic)
                back_wc = torch.matmul(
                    torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
            pred_lg_wc = torch.stack(pred_lg_wc)
            pred_lg_wcs.append(pred_lg_wc)

            # -------- short term goal --------
            pred_lg_heat_from_ic = []
            if self.dataset_name == 'pfsd':
                for coord in pred_lg_ics:
                    heat_map_traj = np.zeros((self.heatmap_size, self.heatmap_size))
                    heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    pred_lg_heat_from_ic.append(heat_map_traj)
            else:
                for i in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                        0].detach().cpu().numpy().astype(int)))
            pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                self.device)
            pred_sg_heat = F.sigmoid(
                self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

            pred_sg_wc = []
            pred_sg_ic_list = []
            sg_heatmap = []
            for i in range(batch_size):
                pred_sg_ic = []
                if self.dataset_name == 'pfsd':
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
                        if i == 0:
                            sg_heatmap.append(heat_map)

                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)

                pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                back_wc = torch.matmul(
                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_sg_wc.append(back_wc[:, :2])
                pred_sg_ic_list.append(pred_sg_ic)
            pred_sg_wc = torch.stack(pred_sg_wc)
            pred_sg_wcs.append(pred_sg_wc)
            pred_sg_ic_list = torch.stack(pred_sg_ic_list)
            pred_sg_ics.append(pred_sg_ic_list)

            # sg_heatmap = torch.stack(sg_heatmap)
            # sg_heatmaps.append(sg_heatmap)
    
        return pred_sg_ics, pred_sg_wcs


    def diffuse_micro(self, batch_size, local_ic, local_homo, pred_sg_ics, local_map, sanity_check = False):

        local_preds = []
        global_preds = []
        trajectories = torch.from_numpy(local_ic).to(torch.float32).to(self.device)
        if not sanity_check:
            trajectories = trajectories[:,:self.obs_len,:]
        
        for pred_sg_ic in pred_sg_ics:
            obs_waypoint = torch.cat((trajectories, pred_sg_ic), dim = 1)
            normed_local_ic = []
            std_scale = []
            for b in range(batch_size):
                map_size = local_map[b].shape
                normed_local_ic.append((obs_waypoint[b]-map_size[0]/2)/(map_size[0]/2))
                std_scale.append(map_size[0]/2)
            normed_local_ic = torch.stack(normed_local_ic).to(self.device)

            condition = {}
            for i in range(self.obs_len):
                condition[i] = normed_local_ic[:,i,:]
            if sanity_check:
                for i in range(len(self.sg_idx)):
                    cond_idx = self.sg_idx[i]+self.obs_len
                    condition[cond_idx] = normed_local_ic[:,cond_idx,:]
            else: 
                for i in range(len(self.sg_idx)):
                    cond_idx = self.sg_idx[i]+self.obs_len
                    condition[cond_idx] = normed_local_ic[:, i+self.obs_len,:]

            # TODO continue here
            samples = self.micro_diffusion(condition, verbose=False)
            # samples = self.micro_diffusion(condition, guide=None, 
            #                     verbose=False, sample_fn=guided_excess_sampling, 
            #                     local_map=None, **self.eval_kwarg.yml_dict)
            pred_global = []
            local_output_collect = []
            for i in range(batch_size):
                pred_local = samples[0][i]* std_scale[i] + std_scale[i]
                local_output_collect.append(pred_local)

                back_wc = torch.matmul(
                    torch.cat([pred_local, torch.ones((len(pred_local), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_global.append(back_wc[:,:2])
            pred_global = torch.stack(pred_global)
            global_preds.append(pred_global.transpose(0,1))
            local_output_collect = torch.stack(local_output_collect)
            local_preds.append(local_output_collect.transpose(0,1))

        return global_preds, local_preds

    
    def guided_diffuse_micro(self, batch_size, local_ic, local_homo, resized_map, pred_sg_ics, local_map, sanity_check = False):

        local_preds = []
        global_preds = []
        
        trajectories = torch.from_numpy(local_ic).to(torch.float32).to(self.device)
        if not sanity_check:
            trajectories = trajectories[:,:self.obs_len,:]

        for pred_sg_ic in pred_sg_ics:
            obs_waypoint = torch.cat((trajectories, pred_sg_ic), dim = 1)
            normed_local_ic = []
            std_scale = []
            for b in range(batch_size):
                map_size = local_map[b].shape
                normed_local_ic.append((obs_waypoint[b]-map_size[0]/2)/(map_size[0]/2))
                std_scale.append(map_size[0]/2)
            normed_local_ic = torch.stack(normed_local_ic).to(self.device)

            condition = {}
            for i in range(self.obs_len):
                condition[i] = normed_local_ic[:,i,:]
            if sanity_check:
                for i in range(len(self.sg_idx)):
                    cond_idx = self.sg_idx[i]+self.obs_len
                    condition[cond_idx] = normed_local_ic[:, cond_idx,:]                
            else:
                for i in range(len(self.sg_idx)):
                    cond_idx = self.sg_idx[i]+self.obs_len
                    condition[cond_idx] = normed_local_ic[:, i+self.obs_len,:]
            
            self.eval_kwarg.yml_dict['center'] = std_scale
            self.eval_kwarg.yml_dict['std_scale'] = std_scale
            self.eval_kwarg.yml_dict['obs_len'] = self.obs_len
            
            samples = self.micro_diffusion(condition, guide=self.guide, 
                                verbose=False, sample_fn=guided_micro_sampling, 
                                local_map=resized_map, **self.eval_kwarg.yml_dict)
            pred_global = []
            local_output_collect = []
            for i in range(batch_size):
                pred_local = samples[0][i]* std_scale[i] + std_scale[i]
                local_output_collect.append(pred_local)
                back_wc = torch.matmul(
                    torch.cat([pred_local, torch.ones((len(pred_local), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_global.append(back_wc[:,:2])
            pred_global = torch.stack(pred_global)
            global_preds.append(pred_global.transpose(0,1))
            local_output_collect = torch.stack(local_output_collect)
            local_preds.append(local_output_collect.transpose(0,1))
        return global_preds, local_preds


    def muse_micro(self, obs_traj, obs_traj_st, seq_start_end, local_homo, pred_sg_wcs):

        fut_rel_pos_dists = []
        pred_fut_trajs = []
        # -------- Micro --------
        (hx, mux, log_varx) \
            = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

        p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
        z_priors = []
        for _ in range(self.n_z):
            z_priors.append(p_dist.sample())

        for pred_sg_wc in pred_sg_wcs:
            for z_prior in z_priors:
                fut_rel_pos_dist_prior = self.decoderMy(
                    obs_traj_st[-1],
                    obs_traj[-1, :, :2],
                    hx,
                    z_prior,
                    pred_sg_wc,  # goal prediction
                    self.sg_idx
                )
                fut_rel_pos_dists.append(fut_rel_pos_dist_prior)
        
        for dist in fut_rel_pos_dists:
            pred_fut_traj = integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)
            pred_fut_trajs.append(pred_fut_traj)
        
        return fut_rel_pos_dists, pred_fut_trajs


    def load_checkpoint(self):
        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'sg_net.pt'
        )
        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'encoderMx.pt'
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'encoderMy.pt'
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'decoderMy.pt'
        )
        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'lg_cvae.pt'
        )

        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
            self.lg_cvae = torch.load(lg_cvae_path)
            self.sg_unet = torch.load(sg_unet_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu').to(self.device)
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu').to(self.device)
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu').to(self.device)
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu').to(self.device)
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu').to(self.device)

        
        diffusion_path = os.path.join(
            self.ckpt_dir,
            'diffusion_weight.pt'
        )

        diffusion_dict = torch.load(diffusion_path,map_location='cpu')
        # self.micro_diffusion.load_state_dict(diffusion_dict['ema'])
        self.micro_diffusion.load_state_dict(diffusion_dict['model'])
        self.micro_diffusion.to(self.device)
        
        
        print('ckpt loaded from ', self.ckpt_dir)


    def set_mode(self, train=True):
        if train:
            self.sg_unet.train()
            self.lg_cvae.train()
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.sg_unet.eval()
            self.lg_cvae.eval()
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()


    def visualize(self, lg_pred, sg_pred, traj_pred, traj_obs,local_map, model='muse'):
        save_dir = "visualization/" + model + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        obs_color = (255,255,0)
        pred_color = (255,0,255)
        thickness = 1

        nav_map = np.zeros((local_map.shape[0], local_map.shape[1], 3))
        nav_map[:,:,0] = local_map
        nav_map[:,:,1] = local_map
        nav_map[:,:,2] = local_map

        nav_map = np.uint8(nav_map * 255)

        num_samples = len(lg_pred)

        one_image = np.zeros((local_map.shape[0], local_map.shape[1], 3))

        for i in range(num_samples):
            color_map = np.zeros((local_map.shape[0], local_map.shape[1], 3), dtype=np.uint8)

            lg_heat = lg_pred[i].detach().cpu().numpy()
            sg_heat = sg_pred[i].detach().cpu().numpy()
            traj = traj_pred[i].detach().cpu().numpy()

            lg_heat /= lg_heat.max()
            for j in range(len(sg_heat)):
                sg_heat[j] /= sg_heat[j].max()
            sg_heat = sg_heat.sum(0)
            sg_heat /= sg_heat.max()

            one_image[:,:,0] += lg_heat
            one_image[:,:,2] += sg_heat

            lg_heat = np.uint8(lg_heat * 255)
            sg_heat = np.uint8(sg_heat * 255)
            
            color_map[:,:,0] = lg_heat
            color_map[:,:,2] = sg_heat

            traj = traj * self.scale
            traj = traj.astype(np.int32)
    
            color_map = cv2.addWeighted(color_map, 0.5, nav_map, 0.5, 0)

            for j in range(len(traj_obs)):
                cv2.circle(color_map, (traj_obs[j,1], traj_obs[j,0]), 1, obs_color, thickness)
            for j in range(self.pred_len):
                if j in self.sg_idx:
                    cv2.circle(color_map, (traj[j,1], traj[j,0]), 1, (0,255,255), thickness)
                else:
                    cv2.circle(color_map, (traj[j,1], traj[j,0]), 1, pred_color, thickness)

            cv2.imwrite(save_dir + 'agent_' + str(self.image_counter) + "_mode_" + str(i) + ".png", color_map)

        one_image = np.clip(one_image, 0, 1)
        one_image = np.uint8(one_image * 255)
        one_image = cv2.addWeighted(one_image, 0.5, nav_map, 0.5, 0)
        for i in range(len(traj_obs)):
            cv2.circle(one_image, (traj_obs[i,1], traj_obs[i,0]), 1, obs_color, thickness)
        for i in range(num_samples):
            traj = traj_pred[i].detach().cpu().numpy()
            traj = traj.astype(np.int32)
            one_image = cv2.polylines(one_image, [traj[:,::-1]], False, pred_color, thickness)
        
        cv2.imwrite(save_dir + 'agent_' + str(self.image_counter) + "_all.png", one_image)

        
    def all_compute(self, sanity_check = False):
        self.set_mode(train=False)

        muse_outputs = []
        diffuse_outputs = []
        guided_outputs = []
        gt_futures = []
        with torch.no_grad():
            if self.dataset_name == 'nuScenes':
                while not self.data_loader.is_epoch_end():
                    batch = self.data_loader.next_sample()
                    if batch is None:
                        continue
                    print('compute {} out of {}'.format(self.data_loader.index, len(self.data_loader.idx_list)))
                    muse_global, diffuse_global, guided_global, gt_global= self.compute_output(batch, sanity_check=sanity_check)
                    muse_outputs.append(muse_global.transpose(2,1).cpu())
                    diffuse_outputs.append(diffuse_global.transpose(2,1).cpu())
                    guided_outputs.append(guided_global.transpose(2,1).cpu())
                    gt_futures.append(gt_global.transpose(1,0).cpu())
            else:
                i=0
                for batch in tqdm(self.data_loader):
                    # print('compute {} out of {}'.format(i, len(self.data_loader)))
                    i+=1
                    muse_global, diffuse_global, guided_global, gt_global= self.compute_output(batch, sanity_check=sanity_check)
                    muse_outputs.append(muse_global.transpose(2,1).cpu())
                    diffuse_outputs.append(diffuse_global.transpose(2,1).cpu())
                    guided_outputs.append(guided_global.transpose(2,1).cpu())
                    gt_futures.append(gt_global.transpose(1,0).cpu())
        muse_outputs = torch.cat(muse_outputs, dim=1)
        diffuse_outputs = torch.cat(diffuse_outputs, dim=1)
        guided_outputs = torch.cat(guided_outputs, dim=1)
        gt_futures = torch.cat(gt_futures, dim=0)

        return muse_outputs, diffuse_outputs, guided_outputs, gt_futures
        # return None, diffuse_outputs, None, None
        # return muse_outputs, diffuse_outputs, None, gt_futures


    def compute_output(self, batch, sanity_check = False):

        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)
        pred_sg_ics, pred_sg_wcs = self.macro_stage(batch)

        fut_rel_pos_dists, muse_pred_trajs = self.muse_micro(obs_traj, obs_traj_st, seq_start_end, local_homo, pred_sg_wcs)

        diffuse_global_pred, diffuse_local_preds = self.diffuse_micro(batch_size, local_ic, local_homo, pred_sg_ics,local_map, sanity_check=sanity_check)

        guided_global_pred, guided_local_preds = self.guided_diffuse_micro(batch_size, local_ic, local_homo, local_map, pred_sg_ics, local_map, sanity_check=sanity_check)

        muse_pred_trajs = torch.stack(muse_pred_trajs)
        diffuse_global_pred = torch.stack(diffuse_global_pred)[:,self.obs_len:]
        guided_global_pred = torch.stack(guided_global_pred)[:,self.obs_len:]
        return muse_pred_trajs, diffuse_global_pred, guided_global_pred, fut_traj[:,:,:2]
        # return None, diffuse_global_pred, None, fut_traj[:,:,:2]
        # return muse_pred_trajs, diffuse_global_pred, None, fut_traj[:,:,:2]