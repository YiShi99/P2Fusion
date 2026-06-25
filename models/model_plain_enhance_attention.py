from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn.functional as F
from .select_network import define_G
from .model_base import ModelBase
import os
from torch.utils.tensorboard import SummaryWriter
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from .loss_total import FusionLoss

import pyiqa
import numpy as np
import torchvision
import cv2
from nets.segformer import SegFormer


class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']  # training option

        self.netG = None
        self.tensorboard_path = os.path.join(self.opt['path']['root'], 'Tensorboard')
        self.netE = None

        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt)
        # ------------------------------------

        self.teacher_ir = SegFormer(num_classes=2, phi="b2").to(self.device)

        self.teacher_vis = pyiqa.create_metric('brisque').to(self.device)
        self.writer = None

        self.fusion_loss = FusionLoss(
            lambda_distill_ir=opt['train'].get('lambda_ir_prompt', 10),
            lambda_distill_vis=opt['train'].get('lambda_vis_prompt', 5),
            lambda_adaptive_intensity=opt['train'].get('lambda_adaptive_intensity', 20),
            lambda_ssim_global=opt['train'].get('lambda_ssim_global', 10),
            lambda_gradient=opt['train'].get('lambda_gradient', 20),
            device=self.device
        )
        self.fixed_ir = None
        self.fixed_vis = None

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        local_rank = self.opt['local_rank']
        self.device = torch.device(f'cuda:{local_rank}')
        # ---------- GPU封装与DDP ----------
        self.netG = define_G(self.opt)
        self.netG = self.netG.to(self.device)
        # 断言检查（调试用，确认模型参数都在本地GPU）
        for name, param in self.netG.named_parameters():
            assert param.device == self.device, f'[DeviceMismatch] Param {name} on {param.device}, expected {self.device}'
        for name, buf in self.netG.named_buffers():
            assert buf.device == self.device, f'[DeviceMismatch] Buffer {name} on {buf.device}, expected {self.device}'

        if self.opt['dist']:
            self.netG = torch.nn.parallel.DistributedDataParallel(
                self.netG,
                device_ids=[self.opt['local_rank']],
                output_device=self.opt['local_rank'],
                find_unused_parameters=False
            )
        elif torch.cuda.device_count() > 1:
            self.netG = nn.DataParallel(self.netG, device_ids=self.opt['gpu_ids'])

        if self.netE is not None:
            self.netE = self.netE.to(self.device).eval()

        ir_weight_path = 'nets/best_epoch_weights.pth'
        state_dict = torch.load(ir_weight_path, map_location=self.device)
        self.teacher_ir.load_state_dict(state_dict, strict=False)

        self.teacher_ir.eval()
        self.teacher_vis.eval()

        for p in self.teacher_ir.parameters():
            p.requires_grad = False
        for p in self.teacher_vis.parameters():
            p.requires_grad = False

        # ---------- TensorBoard ----------
        if self.opt['rank'] == 0:
            os.makedirs(self.tensorboard_path, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_path)
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log
        self.set_fixed_images()

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def set_fixed_images(self):
        pass

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            if self.opt['rank'] == 0:
                print('Loading model for G [{:s}] ...'.format(load_path_G))

            if self.opt['dist'] and isinstance(self.netG, torch.nn.parallel.DistributedDataParallel):
                self.load_network(load_path_G, self.netG.module, strict=self.opt_train['G_param_strict'],
                                  param_key='params')
            else:

                self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                if self.opt['rank'] == 0:
                    print('Loading model for E [{:s}] ...'.format(load_path_E))
                # 同样，如果 netE 也被 DDP 包装了，这里需要 self.netE.module
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                if self.opt['rank'] == 0:
                    print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------

    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))


    def feed_data(self, data, phase='train'):
        self.A = data['A'].to(self.device)
        self.B = data['B'].to(self.device)
        if self.fixed_ir is None or self.fixed_vis is None:
            self.fixed_ir = self.A.clone()
            self.fixed_vis = self.B.clone()
            print("[Info] Fixed images set from feed_data.")

        if phase != 'test' and self.opt_train.get('use_teacher_loss', True):
            with torch.no_grad():
                self.teacher_ir.eval()
                self.teacher_vis.eval()

                self.teacher_ir_logits = self.teacher_ir(self.A)
                teacher_ir_probs = F.softmax(self.teacher_ir_logits, dim=1)
                self.teacher_ir_prompt_raw = teacher_ir_probs[:, 1, :, :]

                target_size = (self.teacher_ir_prompt_raw.shape[1] // 4, self.teacher_ir_prompt_raw.shape[2] // 4)


                teacher_ir_prompt_4d = self.teacher_ir_prompt_raw.unsqueeze(1)

                self.teacher_ir_prompt_downsampled = F.interpolate(
                    teacher_ir_prompt_4d,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

                self.teacher_ir_prompt = self.teacher_ir_prompt_downsampled.squeeze(1)

                self.raw_teacher_vis_score = self.teacher_vis(self.B)  # (B, 1)
                teacher_score_float = self.raw_teacher_vis_score.float()
                high_score_percentage = (100.0 - teacher_score_float)
                self.teacher_vis_score = high_score_percentage / 100.0

                # ----------------------------------------

    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, phase='test'):
        outputs = self.netG(self.A, self.B)

        if isinstance(outputs, tuple):
            self.E = outputs[0]
        else:
            self.E = outputs

    # ----------------------------------------
    def sigmoid_schedule(self, step, mid_step, steep=1.0, max_val=20):
        return float(max_val / (1 + torch.exp(torch.tensor(-steep * (step - mid_step)))))

    def log_fixed_images(self, current_step):
        if self.fixed_ir is not None and self.fixed_vis is not None and self.writer is not None:
            with torch.no_grad():
                outputs = self.netG(self.fixed_ir, self.fixed_vis)


            if isinstance(outputs, tuple):
                fused_img_batch = outputs[0]
            else:
                fused_img_batch = outputs


            self.writer.add_image('fixed/I_ir', self.fixed_ir[0], current_step, dataformats='CHW')
            self.writer.add_image('fixed/I_vis', self.fixed_vis[0], current_step, dataformats='CHW')

            self.writer.add_image('fixed/I_fused', fused_img_batch[0], current_step, dataformats='CHW')

    def optimize_parameters(self, current_step):
        assert hasattr(self, 'teacher_ir_prompt'), "Missing teacher_ir_prompt, call feed_data() first."
        assert hasattr(self, 'teacher_vis_score'), "Missing teacher_vis_score, call feed_data() first."
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            netG_base = self.netG.module
        else:
            netG_base = self.netG
        self.G_optimizer.zero_grad()
        self.netG_forward()



        F_ir, P_ir = netG_base.forward_features_ir(self.A)
        F_vis, P_vis_map, P_vis_score = netG_base.forward_features_vis(self.B)
        F_fused_ir_final, F_fused_vis_final, F_ir_fused, F_vis_fused, enhance_weight_ir, modality_weight_ir, gate_weights_ir, enhance_weight_vis, modality_weight_vis, gate_weights_vis = netG_base.forward_features_Fusion(
            F_ir, F_vis, P_ir, P_vis_map)
        I_f = self.E
        I_ir = self.A
        I_vis = self.B

        I_ir_gray = I_ir[:, :1, :, :]
        I_vis_gray = I_vis[:, :1, :, :]


        loss_dict = self.fusion_loss(
            P_ir=P_ir,
            GT_P_ir=self.teacher_ir_prompt,
            P_vis=P_vis_score,
            GT_P_vis=self.teacher_vis_score,
            I_f=I_f,
            I_vis=I_vis_gray,
            I_ir=I_ir_gray

        )

        G_loss = loss_dict['total']
        G_loss.backward()

        # ========== Clip Grad ==========
        clip_grad = self.opt_train.get('G_optimizer_clipgrad', 0)
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad, norm_type=2)

        self.G_optimizer.step()

        # ========== 正则项 ==========
        save_step = self.opt['train']['checkpoint_save']
        orth_step = self.opt_train.get('G_regularizer_orthstep', 0)
        clip_step = self.opt_train.get('G_regularizer_clipstep', 0)

        if orth_step > 0 and current_step % orth_step == 0 and current_step % save_step != 0:
            self.netG.apply(regularizer_orth)
        if clip_step > 0 and current_step % clip_step == 0 and current_step % save_step != 0:
            self.netG.apply(regularizer_clip)

        # ========== 日志记录 ==========
        self.log_dict['G_loss'] = G_loss.item()
        for k, v in loss_dict.items():
            if k != 'total':
                self.log_dict[f'loss_{k}'] = v.item()
                if self.opt['rank'] == 0:
                    self.writer.add_scalar(f'loss/{k}', v.item(), current_step)

        # ========== EMA 更新 ==========
        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])


        current_lr = self.G_optimizer.param_groups[0]['lr']
        if self.opt['rank'] == 0:
            self.writer.add_scalar('lr/generator', current_lr, current_step)


        if current_step % 1000 == 0 and self.opt['rank'] == 0:
            # 创建文本摘要
            summary = f"Step {current_step} Summary:\n"
            summary += f"Total Loss: {G_loss.item():.4f}\n"
            summary += f"Distill IR: {loss_dict['distill_ir'].item():.4f}\n"
            summary += f"Distill VIS: {loss_dict['distill_vis'].item():.4f}\n"
            summary += f"SSIM Loss: {loss_dict['ssim'].item():.4f}\n"
            summary += f"gradient Loss: {loss_dict['gradient'].item():.4f}\n"
            summary += f"adaptive_intensity Loss: {loss_dict['adaptive_intensity'].item():.4f}\n"

            # 添加梯度范数信息
            grad_norms = []
            for name, param in self.netG.named_parameters():
                if param.grad is not None:
                    grad_norms.append(f"{name}: {param.grad.norm().item():.4f}")

            summary += "Grad Norms:\n" + "\n".join(grad_norms[:5]) + "\n..."  # 只显示前5个

            self.writer.add_text('summary', summary, current_step)



    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward(phase='test')
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self):
        out_dict = OrderedDict()

        out_dict['A'] = self.A.detach()[0].float().cpu()
        out_dict['B'] = self.B.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()

        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['A'] = self.A.detach().float().cpu()
        out_dict['BL'] = self.B.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['GT'] = self.GT.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        # print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        # print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
