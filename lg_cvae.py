import torch.optim as optim
from data.loader import data_loader
from util import *

from tqdm import tqdm

from unet.probabilistic_unet import ProbabilisticUnet
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import torch.nn.functional as F
from unet.utils import init_weights

from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, args):

        self.args = args

        if args.model_name == 'lg_ae':
            self.name = '%s_%s_wD_%s_lr_%s_fcomb_%s' % \
                        (args.dataset_name, args.model_name, args.w_dim, args.lr, args.fcomb)
        elif args.model_name == 'ig_ae':
            self.name = '%s_%s_wD_%s_lr_%s_fcomb_%s' % \
                        (args.dataset_name, args.model_name, args.w_dim, args.lr, args.fcomb)
        elif args.model_name == 'ig_cvae':
            self.name = '%s_%s_wD_%s_lr_%s_fb_%s_anneal_e_%s_fcomb_%s' % \
                        (args.dataset_name, args.model_name, args.w_dim, args.lr,
                         args.fb, args.anneal_epoch, args.fcomb)
        else:
            self.name = '%s_%s_wD_%s_lr_%s_fb_%s_anneal_e_%s_fcomb_%s' % \
                        (args.dataset_name, args.model_name, args.w_dim, args.lr,
                         args.fb, args.anneal_epoch, args.fcomb)
        self.name = self.name + '_run_' + str(args.run_id)
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        if args.alternative_lg != -1:
            self.lg_idx = np.array([args.alternative_lg])
        else:
            self.lg_idx = None

        self.alpha = 0.25
        self.gamma = 2
        self.eps=1e-9
        self.lg_kl_weight=1

        self.w_dim = args.w_dim

        self.max_iter = int(args.max_iter)
        self.lr = args.lr


        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        self.ckpt_load_iter = args.ckpt_load_iter
        mkdirs(self.ckpt_dir)




        if self.ckpt_load_iter == 0:  # create a new model
            if args.model_name == 'lg_ae' or args.model_name == 'ig_ae':
                # input = env + past trajectories / output = env + lg
                num_filters = [32, 32, 64, 64, 64]
                self.lg_cvae = ProbabilisticUnet(input_channels=2, num_classes=1, num_filters=num_filters,
                                                 latent_dim=self.w_dim,
                                                 no_convs_fcomb=args.fcomb,
                                                 no_convs_per_block=1, beta=self.lg_kl_weight).to(self.device)
            elif args.model_name == 'lg_cvae' or args.model_name == 'ig_cvae':
                if self.device == 'cuda':
                    self.lg_cvae = torch.load(args.pretrained_lg_path).to(self.device)
                else:
                    self.lg_cvae = torch.load(args.pretrained_lg_path, map_location='cpu')
                print('>>> lg_ae loaded from ', args.pretrained_lg_path)
                ## random init after latent space
                for m in self.lg_cvae.unet.upsampling_path:
                    m.apply(init_weights)
                self.lg_cvae.fcomb.apply(init_weights)
                self.lg_cvae.beta = self.lg_kl_weight

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')

        self.optim_vae = optim.Adam(
            list(self.lg_cvae.parameters()),
            lr=self.lr,
        )

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

        hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=self.lg_idx, device=self.device)
        self.make_heatmap = hg.make_heatmap



    ####
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        if self.dataset_name == 'nuScenes':
            iter_per_epoch = len(data_loader.idx_list)
        else:
            iterator = iter(data_loader)
            iter_per_epoch = len(iterator)
        start_iter = 1
        epoch = int(start_iter / iter_per_epoch)


        lg_kl_weight = self.lg_kl_weight
        if self.anneal_epoch > 0:
            lg_kl_weight = 0

        writer = SummaryWriter(self.ckpt_dir)

        epoch_lg_likelihood = 0.0

        for iteration in tqdm(range(start_iter, self.max_iter + 1)):
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch +=1
                if self.anneal_epoch > 0:
                    lg_kl_weight = min(self.lg_kl_weight * (epoch / self.anneal_epoch), self.lg_kl_weight)
                    print('kl_w: ', lg_kl_weight)
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

            if self.lg_idx is not None:
                obs_heat_map, lg_heat_map, _ = self.make_heatmap(local_ic, local_map, aug=True)
            else:
                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map, aug=True)

            recon_lg_heat = self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)
            recon_lg_heat = F.normalize(F.sigmoid(recon_lg_heat).view(recon_lg_heat.shape[0],-1), p=1)
            lg_heat_map= lg_heat_map.view(lg_heat_map.shape[0], -1)

            # Focal loss:
            lg_likelihood = (self.alpha * lg_heat_map * torch.log(recon_lg_heat + self.eps) * ((1 - recon_lg_heat) ** self.gamma) \
                         + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - recon_lg_heat + self.eps) * (
                recon_lg_heat ** self.gamma)).sum().div(batch_size)

            if self.model_name == 'lg_cvae' or self.model_name == 'ig_cvae':
                lg_kl = self.lg_cvae.kl_divergence(analytic=True)
                lg_kl = torch.clamp(lg_kl, self.fb).sum().div(batch_size)

                lg_elbo = lg_likelihood - lg_kl_weight * lg_kl
                loss = - lg_elbo

                writer.add_scalar('lg_kl', lg_kl.item(), iteration)
                writer.add_scalar('lg_elbo', lg_elbo.item(), iteration)
                writer.add_scalar('lg_likelihood', -1*lg_likelihood.item(), iteration)

                epoch_lg_likelihood += -1*lg_likelihood.item()

            else:
                loss = - lg_likelihood
                writer.add_scalar('lg_likelihood', -1*lg_likelihood.item(), iteration)
                epoch_lg_likelihood += -1*lg_likelihood.item()

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()

            # save model parameters
            if iteration % (iter_per_epoch*10) == 0:
                self.save_checkpoint(iteration)
            if iteration % (iter_per_epoch) == 0:
                writer.add_scalar('epoch_lg_likelihood', epoch_lg_likelihood/iter_per_epoch, iteration/iter_per_epoch)
                epoch_lg_likelihood = 0.0

        writer.flush()
        writer.close()
        # save model parameters
        try:
            self.save_checkpoint(self.max_iter)
        except:
            print('save model error')


    def set_mode(self, train=True):
        if train:
            self.lg_cvae.train()
        else:
            self.lg_cvae.eval()


    ####
    def save_checkpoint(self, iteration):
        path = os.path.join(
            self.ckpt_dir,
            'iter_%s_%s.pt' % (iteration, self.model_name))
        del self.lg_cvae.unet.blocks
        torch.save(self.lg_cvae, path)

    ####
    def load_checkpoint(self):
        path = os.path.join(
            self.ckpt_dir,
            'iter_%s_%s.pt' % (self.ckpt_load_iter, self.model_name)
        )
        if self.device == 'cuda':
            self.lg_cvae = torch.load(path)

        else:
            self.lg_cvae = torch.load(path, map_location='cpu')
