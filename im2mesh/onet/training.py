import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import random


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''

        self.model.train()
        self.optimizer.zero_grad()
        loss_out_concat, loss_out_xfg1, loss_out_xfg2, loss_out, loss_kl_fxg1, loss_kl_fxg2, loss_kl_out, loss_kl_concat_out, loss_all = self.compute_loss(
            data)
        loss_all.backward()
        self.optimizer.step()

        return loss_out_concat.item(), loss_out_xfg1.item(), loss_out_xfg2.item(), loss_out.item(), loss_kl_fxg1.item(), loss_kl_fxg2.item(), loss_kl_out.item(), loss_kl_concat_out.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        inputs_xfg1 = self.jigsaw_generator(inputs, 4)
        inputs_xfg2 = self.jigsaw_generator(inputs, 2)
        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, inputs_xfg1, inputs_xfg2, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out,out = self.model(points_iou, inputs, inputs_xfg1, inputs_xfg2,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        
        occ_iou_hat_np_ori = (out.probs >= threshold).cpu().numpy()
        iou1 = compute_iou(occ_iou_np, occ_iou_hat_np_ori).mean()
        eval_dict['iou'] = iou
        eval_dict['iou_ori'] = iou1

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out,_ = self.model(points_voxels, inputs, inputs_xfg1, inputs_xfg2,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels
            
            

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())
        inputs_xfg1 = self.jigsaw_generator(inputs, 4)
        inputs_xfg2 = self.jigsaw_generator(inputs, 2)
        kwargs = {}
        with torch.no_grad():
            p_r,_ = self.model(p, inputs, inputs_xfg1, inputs_xfg2, sample=self.eval_sample,
                             **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        # print(inputs.size())
        inputs_xfg1 = self.jigsaw_generator(inputs, 4)
        # print('1111',inputs_xfg1.size())
        inputs_xfg2 = self.jigsaw_generator(inputs, 2)
        kwargs = {}

        xfg1, xfg2, x, concate_out = self.model.encode_inputs(inputs, inputs_xfg1, inputs_xfg2)

        q_z_xfg1 = self.model.infer_z(p, occ, xfg1, **kwargs)

        q_z_xfg2 = self.model.infer_z(p, occ, xfg2, **kwargs)

        q_z_x = self.model.infer_z(p, occ, x, **kwargs)

        q_z_concat = self.model.infer_z(p, occ, concate_out, **kwargs)

        z_xfg1 = q_z_xfg1.rsample()
        z_xfg2 = q_z_xfg2.rsample()
        z_x0 = q_z_x.rsample()
        z_concate_out = q_z_concat.rsample()

        # KL-divergence
        kl_xfg1 = dist.kl_divergence(q_z_xfg1, self.model.p0_z).sum(dim=-1)
        kl_xfg2 = dist.kl_divergence(q_z_xfg2, self.model.p0_z).sum(dim=-1)
        kl_x = dist.kl_divergence(q_z_x, self.model.p0_z).sum(dim=-1)
        kl_concat = dist.kl_divergence(q_z_concat, self.model.p0_z).sum(dim=-1)
        loss_xfg1 = kl_xfg1.mean()
        loss_xfg2 = kl_xfg2.mean()
        loss_x = kl_x.mean()
        loss_concat = kl_concat.mean()

        # General points
        out_concat, out_xfg1, out_xfg2, out = self.model.decode(p, z_xfg1, z_xfg2, z_x0, z_concate_out, xfg1, xfg2, x,
                                                                concate_out,train_step=1, **kwargs)

        loss_out_concat = F.binary_cross_entropy_with_logits(
            out_concat.logits, occ, reduction='none')

        loss_out_xfg1 = F.binary_cross_entropy_with_logits(
            out_xfg1.logits, occ, reduction='none')

        loss_out_xfg2 = F.binary_cross_entropy_with_logits(
            out_xfg2.logits, occ, reduction='none')

        loss_out = F.binary_cross_entropy_with_logits(
            out.logits, occ, reduction='none')
        # print(out_concat.logits.size())
        loss_kl_xfg1 = F.kl_div(F.log_softmax(out_xfg1.logits, dim=-1), F.softmax(out_xfg2.logits, dim=-1),
                                reduction='none')
        loss_kl_xfg2 = F.kl_div(F.log_softmax(out_xfg2.logits, dim=-1), F.softmax(out.logits, dim=-1), reduction='none')
        loss_kl_out = F.kl_div(F.log_softmax(out.logits, dim=-1), F.softmax(out_concat.logits, dim=-1),
                               reduction='none')
        # torch.nn.KLDivLoss
        loss_kl_concat_xfg1 = F.kl_div(F.log_softmax(out_concat.logits, dim=-1), F.softmax(out_xfg1.logits, dim=-1),
                                       reduction='none')
        loss_kl_concat_xfg2 = F.kl_div(F.log_softmax(out_concat.logits, dim=-1), F.softmax(out_xfg2.logits, dim=-1),
                                       reduction='none')
        loss_kl_concat_out = F.kl_div(F.log_softmax(out_concat.logits, dim=-1), F.softmax(out.logits, dim=-1),
                                      reduction='none')

        loss_out_concat = loss_concat + loss_out_concat.sum(-1).mean()
        loss_out_xfg1 = loss_xfg1 + loss_out_xfg1.sum(-1).mean()
        loss_out_xfg2 = loss_xfg2 + loss_out_xfg2.sum(-1).mean()
        loss_out = loss_x + loss_out.sum(-1).mean()

        loss_kl_fxg1 = 1 * (loss_kl_xfg1.sum(-1).mean())
        loss_kl_fxg2 = 1 * (loss_kl_xfg2.sum(-1).mean())
        loss_kl_out = 1 * (loss_kl_out.sum(-1).mean())
        loss_kl_concat_out = 1 * (0.1 * loss_kl_concat_xfg1.sum(-1).mean() + 0.2 * loss_kl_concat_xfg2.sum(
            -1).mean() + 0.7 * loss_kl_concat_out.sum(-1).mean())

        loss_all = loss_out_concat + loss_out_xfg1 + loss_out_xfg2 + loss_out + loss_kl_fxg1 + loss_kl_fxg2 + loss_kl_out + loss_kl_concat_out
        return loss_out_concat, loss_out_xfg1, loss_out_xfg2, loss_out, loss_kl_fxg1, loss_kl_fxg2, loss_kl_out, loss_kl_concat_out, loss_all

    def jigsaw_generator(self, inputs, n):
        l = []
        for a in range(n):
            for b in range(n):
                l.append([a, b])
        block_size = 224 // n
        rounds = n ** 2
        random.shuffle(l)
        jigsaws = inputs.clone()
        for i in range(rounds):
            x, y = l[i]
            temp = jigsaws[..., 0:block_size, 0:block_size].clone()
            # print(temp.size())
            jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                       y * block_size:(y + 1) * block_size].clone()
            jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

        return jigsaws
