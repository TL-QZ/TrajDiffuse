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


    def macro_stage(self, batch, visualize=False):
        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)
        local_ic = local_ic.astype(int)
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
                    sg_heatmap_inter = []
                    for heat_map in pred_sg_heat[i]:
                        sg_heatmap_inter.append(heat_map)
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        # if i == 0:
                        # sg_heatmap_inter.append(heat_map)

                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)
                    sg_heatmap.append(torch.stack(sg_heatmap_inter))
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

        #     sg_heatmap = torch.stack(sg_heatmap)
        #     sg_heatmaps.append(sg_heatmap)
        # if visualize:
        #     return pred_sg_ics, pred_sg_wcs, torch.stack(sg_heatmaps)
        # else:
        return pred_sg_ics, pred_sg_wcs


    def diffuse_micro(self, batch_size, local_ic, local_homo, pred_sg_ics, local_map, sanity_check = False, visualize = False):

        local_preds = []
        global_preds = []
        inter_local_preds = []
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
            if visualize:
                samples = self.micro_diffusion(condition, verbose=False, return_chain=True)
            else:
                samples = self.micro_diffusion(condition, verbose=False)
            # samples = self.micro_diffusion(condition, guide=None, 
            #                     verbose=False, sample_fn=guided_excess_sampling, 
            #                     local_map=None, **self.eval_kwarg.yml_dict)
            pred_global = []
            local_output_collect = []
            inter_output_collect = []
            for i in range(batch_size):
                pred_local = samples[0][i]* std_scale[i] + std_scale[i]
                local_output_collect.append(pred_local)
                back_wc = torch.matmul(
                    torch.cat([pred_local, torch.ones((len(pred_local), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_global.append(back_wc[:,:2])
                if visualize:
                    inter_local = samples[2][i]* std_scale[i] + std_scale[i]
                    inter_output_collect.append(inter_local)
            if visualize:
                inter_output_collect = torch.stack(inter_output_collect)
                inter_local_preds.append(inter_output_collect.transpose(0,2))
            pred_global = torch.stack(pred_global)
            global_preds.append(pred_global.transpose(0,1))
            local_output_collect = torch.stack(local_output_collect)
            local_preds.append(local_output_collect.transpose(0,1))
            
        if visualize:
            return global_preds, local_preds, inter_local_preds
        else:
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

            # TODO: if sdd then make the 
            if self.dataset_name == 'sdd':
                binarized_map = []
                for b in range(batch_size):
                    binarized_map.append(np.where(resized_map[b] == 3, 1,0))
                resized_map = binarized_map
            
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


    def all_compute(self, sanity_check = False, to_eval = 'all'):

        if to_eval == 'all':
            all, muse, diffuse, guide = True, True, True, True
        elif to_eval == 'quick':
            all, muse, diffuse, guide = False, True, True, False
        elif to_eval == 'diffuse':
            all, muse, diffuse, guide = False, False, True, False
        elif to_eval == 'guide':
            all, muse, diffuse, guide = False, False, False, True
        else:
            raise ValueError("mode error, available mode: [all, diffuse, quick, guide]")
        
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
                    muse_global, diffuse_global, guided_global, gt_global= self.compute_output(batch, sanity_check=sanity_check, to_eval = to_eval)
                    if all or muse:
                        muse_outputs.append(muse_global.transpose(2,1).cpu())
                    if all or diffuse:
                        diffuse_outputs.append(diffuse_global.transpose(2,1).cpu())
                    if all or guide:
                        guided_outputs.append(guided_global.transpose(2,1).cpu())
                    gt_futures.append(gt_global.transpose(1,0).cpu())
            else:
                for batch in tqdm(self.data_loader):
                    muse_global, diffuse_global, guided_global, gt_global= self.compute_output(batch, sanity_check=sanity_check, to_eval = to_eval)
                    if all or muse:
                        muse_outputs.append(muse_global.transpose(2,1).cpu())
                    if all or diffuse:
                        diffuse_outputs.append(diffuse_global.transpose(2,1).cpu())
                    if all or guide:    
                        guided_outputs.append(guided_global.transpose(2,1).cpu())
                    gt_futures.append(gt_global.transpose(1,0).cpu())
        
        if to_eval == 'all':
            muse_outputs = torch.cat(muse_outputs, dim=1)
            diffuse_outputs = torch.cat(diffuse_outputs, dim=1)
            guided_outputs = torch.cat(guided_outputs, dim=1)
            gt_futures = torch.cat(gt_futures, dim=0)
            return muse_outputs, diffuse_outputs, guided_outputs, gt_futures

        elif to_eval == 'quick':
            muse_outputs = torch.cat(muse_outputs, dim=1)
            diffuse_outputs = torch.cat(diffuse_outputs, dim=1)
            gt_futures = torch.cat(gt_futures, dim=0)
            return muse_outputs, diffuse_outputs, gt_futures
        
        elif to_eval == 'diffuse':
            diffuse_outputs = torch.cat(diffuse_outputs, dim=1)
            gt_futures = torch.cat(gt_futures, dim=0)
            return diffuse_outputs, gt_futures
        
        elif to_eval == 'guide':
            guided_outputs = torch.cat(guided_outputs, dim=1)
            gt_futures = torch.cat(gt_futures, dim=0)
            return guided_outputs, gt_futures
        else:
            raise ValueError("mode error, available mode: [all, quick, guide]")


    def compute_output(self, batch, sanity_check = False, to_eval='all'):

        if to_eval == 'all':
            all, muse, diffuse, guide = True, True, True, True
        elif to_eval == 'quick':
            all, muse, diffuse, guide = False, True, True, False
        elif to_eval == 'diffuse':
            all, muse, diffuse, guide = False, False, True, False
        elif to_eval == 'guide':
            all, muse, diffuse, guide = False, False, False, True
        else:
            raise ValueError("mode error, available mode: [all, quick, guide]")


        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)
        pred_sg_ics, pred_sg_wcs = self.macro_stage(batch)

        muse_pred_trajs= None
        diffuse_global_pred = None
        guided_global_pred = None

        if all or muse:
            fut_rel_pos_dists, muse_pred_trajs = self.muse_micro(obs_traj, obs_traj_st, seq_start_end, local_homo, pred_sg_wcs)
            muse_pred_trajs = torch.stack(muse_pred_trajs)
        if all or diffuse:
            diffuse_global_pred, diffuse_local_preds = self.diffuse_micro(batch_size, local_ic, local_homo, pred_sg_ics,local_map, sanity_check=sanity_check)
            diffuse_global_pred = torch.stack(diffuse_global_pred)[:,self.obs_len:]
        if all or guide:
            guided_global_pred, guided_local_preds = self.guided_diffuse_micro(batch_size, local_ic, local_homo, local_map, pred_sg_ics, local_map, sanity_check=sanity_check)
            guided_global_pred = torch.stack(guided_global_pred)[:,self.obs_len:]
        
        return muse_pred_trajs, diffuse_global_pred, guided_global_pred, fut_traj[:,:,:2]

    
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
