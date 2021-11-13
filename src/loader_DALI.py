import sys
import glob
import torch
import random
import argparse

import numpy as np
import nvidia.dali.fn as fn

from tqdm import tqdm
from time import time
from PIL import Image
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

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

class ExternalInputIterator(object):
    def __init__(self, data_source, batch_size, shuffle):
        self.batch_size = batch_size
        
        img_paths = sorted(glob.glob(data_source + '/train' + '/blurry' + '/*/*.*'))
        gt_paths = sorted(glob.glob(data_source + '/train' + '/sharp' + '/*/*.*'))
        self.paths = list(zip(*(img_paths,gt_paths)))
        if shuffle:
            random.shuffle(self.paths)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        imgs = []
        gts = []

        if self.i >= len(self.paths):
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            img_path, gt_path = self.paths[self.i % len(self.paths)]
            imgs.append(np.fromfile(img_path, dtype = np.uint8))
            gts.append(np.fromfile(gt_path, dtype = np.uint8))
            self.i += 1
        return (imgs, gts)

    def __len__(self):
        return len(self.paths)

    next = __next__

@pipeline_def
def externalSourcePipeline(external_data, resize, crop):
    imgs, gts = fn.external_source(source=external_data, num_outputs=2)
    
    crop_pos = (fn.random.uniform(range=(0., 1.)), fn.random.uniform(range=(0., 1.)))
    flip_p = (fn.random.coin_flip(), fn.random.coin_flip())
    
    imgs = transform(imgs, resize, crop, crop_pos, flip_p)
    gts = transform(gts, resize, crop, crop_pos, flip_p)
    return imgs, gts

def transform(imgs, resize, crop, crop_pos, flip_p):
    imgs = fn.decoders.image(imgs, device='mixed')
    imgs = fn.resize(imgs, resize_y=resize)
    imgs = fn.crop(imgs, crop=(crop,crop), crop_pos_x=crop_pos[0], crop_pos_y=crop_pos[1])
    imgs = fn.flip(imgs, horizontal=flip_p[0], vertical=flip_p[1])
    imgs = fn.transpose(imgs, perm=[2, 0, 1])
    imgs = imgs/127.5-1
    
    return imgs

def warmming_up(dataloader):
    print('warmming up begin...')
    for i, data in tqdm(enumerate(dataloader)):
        imgs = data[0]['imgs']
        gts = data[0]['gts']
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

eii = ExternalInputIterator(data_source=opt.data_source, batch_size=opt.batch_size, shuffle=True)
pipe = externalSourcePipeline(batch_size=opt.batch_size, num_threads=opt.num_workers, device_id=0, seed=opt.seed, external_data = eii, resize=opt.resize, crop=opt.crop)
dgi = DALIGenericIterator(pipe, output_map=["imgs", "gts"], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

print('---------------------------------------- running... ----------------------------------------------------')
f = open(opt.result_dir + '/log_' + sys.argv[0][0:-3] + '.txt', 'w')
f.write('Setting: batch_size: {}, num_workers: {}, resize: {}, crop: {}'.format(opt.batch_size, opt.num_workers, opt.resize, opt.crop) + '\n')
f.write('----------------------------' + '\n')
f.write('Running: ' + '\n')
f.write('----------------------------' + '\n')

warmming_up(dgi)

time_begin = time()

time_shot = time()
for i, data in enumerate(dgi):
    imgs = data[0]['imgs']
    gts = data[0]['gts']
        
    print('Running: Iteration[{:0>4}/{:0>4}] Time: {:.4f}'.format(i + 1, len(eii)//opt.batch_size+1, time()-time_shot))
    f.write('Running: Iteration[{:0>4}/{:0>4}] Time: {:.4f}'.format(i + 1, len(eii)//opt.batch_size+1, time()-time_shot) + '\n')
    time_shot = time()
        
print('')
print('TOTAL TIME COST: {:.4f}'.format(time()-time_begin))
f.write('\n')
f.write('TOTAL TIME COST: {:.4f}'.format(time()-time_begin) + '\n')

f.close()