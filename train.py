import os
import math
import argparse
import random
import logging
import warnings

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from models.model_plain_enhance_attention import ModelPlain

warnings.filterwarnings("ignore")


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])

    def simulate_ir_temperature(self, img):
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    def simulate_vis_as_ir(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def misalignment(self, img, max_shift=4):
        h, w, _ = img.shape
        tx = np.random.randint(-max_shift, max_shift + 1)
        ty = np.random.randint(-max_shift, max_shift + 1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def random_occlusion(self, img, min_frac=0.05, max_frac=0.2):
        h, w, _ = img.shape
        occ_area = random.uniform(min_frac, max_frac) * h * w
        occ_size = int(np.sqrt(occ_area))
        top = random.randint(0, h - occ_size)
        left = random.randint(0, w - occ_size)
        if random.random() < 0.5:
            img[top:top + occ_size, left:left + occ_size, :] = 0
        else:
            mean_val = img.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
            img[top:top + occ_size, left:left + occ_size, :] = mean_val
        return img

    def random_grayscale(self, img):
        if random.random() < 0.2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img

    def apply_augmentations(self, img_A, img_B):
        img_A = cv2.resize(img_A, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_B = cv2.resize(img_B, (512, 512), interpolation=cv2.INTER_LINEAR)

        mode = random.randint(0, 7)
        img_A = util.augment_img(img_A, mode)
        img_B = util.augment_img(img_B, mode)

        if random.random() < 0.5:
            img_A = self.simulate_ir_temperature(img_A)

        if random.random() < 0.3:
            img_B = self.simulate_vis_as_ir(img_B)

        img_B = self.random_grayscale(img_B)

        if random.random() < 0.3:
            img_A = self.misalignment(img_A)
        if random.random() < 0.3:
            img_B = self.misalignment(img_B)

        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.85, 1.15)
        img_B = np.clip(img_B * brightness, 0, 255).astype(np.uint8)
        mean = np.mean(img_B, axis=(0, 1), keepdims=True)
        img_B = np.clip((img_B - mean) * contrast + mean, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            k = random.choice([3, 5])
            img_A = cv2.GaussianBlur(img_A, (k, k), 0)
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img_B = cv2.GaussianBlur(img_B, (k, k), 0)

        if random.random() < 0.2:
            noise_A = np.random.normal(0, random.uniform(1, 5), img_A.shape).astype(np.float32)
            noise_B = np.random.normal(0, random.uniform(1, 5), img_B.shape).astype(np.float32)

            img_A = img_A.astype(np.float32) + noise_A
            img_B = img_B.astype(np.float32) + noise_B

            img_A = np.clip(img_A, 0, 255).astype(np.uint8)
            img_B = np.clip(img_B, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            img_A = self.random_occlusion(img_A)
        if random.random() < 0.3:
            img_B = self.random_occlusion(img_B)

        return img_A, img_B

    def __getitem__(self, index):
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, self.n_channels)
        img_B = util.imread_uint(B_path, self.n_channels)

        if self.opt['phase'] == 'train':
            img_A, img_B = self.apply_augmentations(img_A, img_B)
            img_A = util.uint2tensor3(img_A)
            img_B = util.uint2tensor3(img_B)
            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}
        else:
            img_A = util.uint2single(img_A)
            img_B = util.uint2single(img_B)
            img_A = util.single2tensor3(img_A)
            img_B = util.single2tensor3(img_B)
            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)


def main(json_path='options/train_p2fusion.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt['dist'] = True
        opt['local_rank'] = int(os.environ['LOCAL_RANK'])
    else:
        opt['dist'] = False
        opt['local_rank'] = args.local_rank

    if opt['dist']:
        torch.cuda.set_device(opt['local_rank'])
        dist.init_process_group(backend='nccl', init_method='env://')

    opt['rank'] = dist.get_rank() if opt['dist'] else 0
    opt['world_size'] = dist.get_world_size() if opt['dist'] else 1

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')

    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    iters = [init_iter_G, init_iter_E, init_iter_optimizerG]
    iters = [i for i in iters if i > 0]
    current_step = max(iters) if iters else 0

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if opt['rank'] == 0:
        print(f'Random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

            if opt['rank'] == 0:
                logger.info(f'Number of train images: {len(train_set):,d}, iters: {train_size:,d}')

            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['world_size'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['world_size'],
                                          drop_last=True,
                                          pin_memory=True,
                                          persistent_workers=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")

    model = ModelPlain(opt)
    model.init_train()

    best_loss = float('inf')

    for epoch in range(250):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        if opt['rank'] == 0:
            data_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/250', unit='batch')
        else:
            data_iter = train_loader

        for i, train_data in enumerate(data_iter):
            current_step += 1
            model.update_learning_rate(current_step)
            model.feed_data(train_data, phase='train')
            model.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = f'<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                for k, v in logs.items():
                    message += f'{k:s}: {v:.3e} '
                logger.info(message)
                data_iter.set_postfix({k: f'{v:.3e}' for k, v in logs.items()})

                if 'G_loss' in logs:
                    if logs['G_loss'] < best_loss:
                        best_loss = logs['G_loss']
                        model.save('best')
                        logger.info(f'>>> Saving best model at iter {current_step}, G_loss = {best_loss:.6f}')

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                save_dir = opt['path']['models']
                model.save(current_step)
                logger.info(f'Saving the model. Save dir is: {save_dir}')

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                test_iter = tqdm(test_loader, desc='Testing', unit='batch', leave=False)

                for test_data in test_iter:
                    image_name_ext = os.path.basename(test_data['A_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data, phase='test')
                    model.test()
                    visuals = model.current_visuals()

                    if 'E' in visuals:
                        E_img = util.tensor2uint(visuals['E'])
                        save_img_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                        util.imsave(E_img, save_img_path)
                    else:
                        continue


if __name__ == '__main__':
    main()