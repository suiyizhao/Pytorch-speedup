import sys
import glob
import torch
import random
import argparse

import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from time import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def set_random_seed(seed, deterministic=False):
    
    '''
    function: Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PairedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.transform = transforms.Compose([
            transforms.Resize(opt.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.img_paths = sorted(glob.glob(opt.data_source + '/train' + '/blurry' + '/*/*.*'))
        self.gt_paths = sorted(glob.glob(opt.data_source + '/train' + '/sharp' + '/*/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
        gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')
        
        img = self.transform(img)
        gt = self.transform(gt)
        
        # crop
        h, w = img.size(1), img.size(2)
        offset_h = random.randint(0, max(0, h - self.opt.crop - 1))
        offset_w = random.randint(0, max(0, w - self.opt.crop - 1))
        
        img = img[:, offset_h:offset_h + self.opt.crop, offset_w:offset_w + self.opt.crop]
        gt = gt[:, offset_h:offset_h + self.opt.crop, offset_w:offset_w + self.opt.crop]
        
        # flip
        # vertical flip
        if random.random() < 0.5:
            idx = [i for i in range(img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img = img.index_select(1, idx)
            gt = gt.index_select(1, idx)
        # horizontal flip
        if random.random() < 0.5:
            idx = [i for i in range(img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img = img.index_select(2, idx)
            gt = gt.index_select(2, idx)
        
        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))

def warmming_up(dataloader):
    print('warmming up begin...')
    for i, (imgs, gts) in tqdm(enumerate(dataloader)):
        imgs = imgs.cuda()
        gts = gts.cuda()
    print('warmming up end!')
    
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--data_source", type=str, default='/path/to/dataset', help="dataset root")
parser.add_argument("--resize", type=int, default=256, help="obtained image size after resize operation")
parser.add_argument("--crop", type=int, default=224, help="obtained image size after crop operation")
parser.add_argument("--result_dir", type=str, default='../result', help="dir for saving the results")
opt = parser.parse_args()
print(opt)

set_random_seed(opt.seed, deterministic=True)

dataset = PairedDataset(opt)
dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

print('---------------------------------------- running... ----------------------------------------------------')
f = open(opt.result_dir + '/log_' + sys.argv[0][0:-3] + '.txt', 'w')
f.write('Setting: batch_size: {}, num_workers: {}, resize: {}, crop: {}'.format(opt.batch_size, opt.num_workers, opt.resize, opt.crop) + '\n')
f.write('----------------------------' + '\n')
f.write('Running: ' + '\n')
f.write('----------------------------' + '\n')

warmming_up(dataloader)

time_begin = time()

time_shot = time()
for i, (imgs, gts) in enumerate(dataloader):
    imgs = imgs.cuda()
    gts = gts.cuda()
    print('Running: Iteration[{:0>4}/{:0>4}] Time: {:.4f}'.format(i + 1, len(dataloader), time()-time_shot))
    f.write('Running: Iteration[{:0>4}/{:0>4}] Time: {:.4f}'.format(i + 1, len(dataloader), time()-time_shot) + '\n')
    time_shot = time()
        
print('')
print('TOTAL TIME COST: {:.4f}'.format(time()-time_begin))
f.write('\n')
f.write('TOTAL TIME COST: {:.4f}'.format(time()-time_begin) + '\n')

f.close()