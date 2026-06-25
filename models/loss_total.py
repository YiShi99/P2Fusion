import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class FusionLoss(nn.Module):
    """
    Unified fusion loss combining distillation, intensity, SSIM, and gradient components.
    """

    def __init__(self,
                 lambda_distill_ir=1.0,
                 lambda_distill_vis=6.0,
                 lambda_adaptive_intensity=5.0,
                 lambda_ssim_global=150.0,
                 lambda_gradient=3.0,
                 type_distill_ir='BCE',
                 type_distill_vis='MSE',
                 device='cuda'):
        super(FusionLoss, self).__init__()
        self.lambda_distill_ir = lambda_distill_ir
        self.lambda_distill_vis = lambda_distill_vis
        self.lambda_adaptive_intensity = lambda_adaptive_intensity
        self.lambda_ssim_global = lambda_ssim_global
        self.lambda_gradient = lambda_gradient

        self.L_Grad = L_Grad().to(device)
        self.cal_intensity_loss = L_Intensity().to(device)
        self.L_SSIM = L_SSIM().to(device)

        self.type_distill_ir = type_distill_ir.upper()
        self.type_distill_vis = type_distill_vis.upper()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.device = device

    def compute_distill_loss(self, pred, target, loss_type):
        if loss_type == 'BCE':
            return self.bce_loss(pred, target)
        elif loss_type == 'MSE':
            return self.mse_loss(pred, target)
        elif loss_type == 'L1':
            return self.l1_loss(pred, target)
        else:
            raise ValueError(f'Unsupported distillation loss type: {loss_type}')

    def forward(self, P_ir, GT_P_ir, P_vis, GT_P_vis, I_f, I_vis, I_ir):
        loss_dict = {}
        total_loss = 0.0
        GT_P_ir = GT_P_ir.float()
        GT_P_vis = GT_P_vis.float()

        # 1. Distillation Loss
        loss_distill_ir = self.compute_distill_loss(P_ir[:, 0], GT_P_ir, self.type_distill_ir)
        loss_distill_vis = self.compute_distill_loss(P_vis, GT_P_vis, self.type_distill_vis)

        loss_dict['distill_ir'] = loss_distill_ir
        loss_dict['distill_vis'] = loss_distill_vis
        total_loss += self.lambda_distill_ir * loss_distill_ir + self.lambda_distill_vis * loss_distill_vis

        # 2. Adaptive Intensity Loss
        loss_adaptive_intensity = self.cal_intensity_loss(I_ir, I_vis, I_f)
        loss_dict['adaptive_intensity'] = loss_adaptive_intensity
        total_loss += self.lambda_adaptive_intensity * loss_adaptive_intensity

        # 3. Global SSIM Loss
        loss_ssim = (1 - self.L_SSIM(I_ir, I_vis, I_f))
        loss_dict['ssim'] = loss_ssim
        total_loss += self.lambda_ssim_global * loss_ssim

        # 4. Gradient Loss
        loss_gradient = self.L_Grad(I_ir, I_vis, I_f, P_vis)
        loss_dict['gradient'] = loss_gradient
        total_loss += self.lambda_gradient * loss_gradient

        loss_dict['total'] = total_loss
        return loss_dict


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused, P_vis):
        grad_A = self.sobelconv(image_A)
        grad_B = self.sobelconv(image_B)
        grad_F = self.sobelconv(image_fused)

        grad_joint_max = torch.max(grad_A, grad_B)
        return F.l1_loss(grad_F, grad_joint_max)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)

        mean_A = torch.mean(gradient_A)
        mean_B = torch.mean(gradient_B)

        weight_A = mean_A / (mean_A + mean_B + 1e-8)
        weight_B = mean_B / (mean_A + mean_B + 1e-8)

        return weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        return F.l1_loss(image_fused, intensity_joint)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernely = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        C = x.shape[1]
        grad_x = F.conv2d(x, self.weightx.expand(C, 1, 3, 3), padding=1, groups=C)
        grad_y = F.conv2d(x, self.weighty.expand(C, 1, 3, 3), padding=1, groups=C)
        return torch.abs(grad_x) + torch.abs(grad_y)