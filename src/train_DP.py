import sys
import time
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader

def printParaNum(model):
    
    '''
    function: print the number of total parameters and trainable parameters
    '''
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    print('Trainable parameters: %d' % total_trainable_params)
    
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(1, 3, 3, 2), nn.BatchNorm2d(3), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(3, 3, 3, 1), nn.BatchNorm2d(3), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(3, 8, 3, 2), nn.BatchNorm2d(8), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(8, 8, 3, 1), nn.BatchNorm2d(8), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(8, 16, 3, 2), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 32, 3, 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(), nn.Linear(128, 10)
        )
        
        self.initialize_weights()
        
    def forward(self, img):
        out = self.model(img)

        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

time_begin = time.time()

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--workers", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--dataset", type=str, default='../dataset/mnist', help="dataset root")
parser.add_argument("--result_dir", type=str, default='../result', help="dir for saving the results")
opt = parser.parse_args()
print(opt)

set_random_seed(1234, deterministic=True)

time_1 = time.time()

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
dataset = datasets.MNIST(opt.dataset, train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
time_2 = time.time()

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
# new #
model = nn.DataParallel(Model().cuda())
printParaNum(model)
time_3 = time.time()

print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
# Loss function
loss_func = nn.CrossEntropyLoss()
# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
time_4 = time.time()

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
f = open(opt.result_dir + '/log_' + sys.argv[0][0:-3] + '.txt', 'w')
f.write('Type: single machine, multiple card, fixed precision' + '\n')
f.write('Parallel manner: DataParallel' + '\n')
f.write('Mixing manner: none' + '\n')
f.write('Setting: epochs: {}, lr: {}, batch_size: {}, workers: {}'.format(opt.epochs, opt.lr, opt.batch_size, opt.workers) + '\n')
f.write('----------------------------' + '\n')
f.write('Training: ' + '\n')
f.write('----------------------------' + '\n')
time_4_dataloading = 0
time_4_computing = 0
for epoch in range(opt.epochs):
    time_4_begin = time.time()
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        time_temp = time.time()
        time_4_dataloading += time_temp - time_4_begin
        
        optimizer.zero_grad()
        pred = model(imgs)
        loss = loss_func(pred, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(pred, 1)
        acc = (pred == labels).sum().item() / len(labels)
        
        print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>4}/{:0>4}] Loss: {:.4f} Acc: {:.4f}'.format(
                       epoch + 1, opt.epochs, i + 1, len(dataloader), loss, acc))
        f.write('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>4}/{:0>4}] Loss: {:.4f} Acc: {:.4f}'.format(
                       epoch + 1, opt.epochs, i + 1, len(dataloader), loss, acc) + '\n')
        time_4_computing += time.time() - time_temp
        time_4_begin = time.time()
        
time_5 = time.time()

f.write('\n')
f.write('TIME COST' + '\n')
f.write('Parameters preparing: {:.6f}(s)'.format(time_1 - time_begin) + '\n')
f.write('Data loading: {:.6f}(s)'.format(time_2 - time_1) + '\n')
f.write('Model defining: {:.6f}(s)'.format(time_3 - time_2) + '\n')
f.write('Requisites defining: {:.6f}(s)'.format(time_4 - time_3) + '\n')
f.write('Training: {:.6f}(s)'.format(time_5 - time_4) + '\n')
f.write('    Training (dataloading): {:.6f}(s)'.format(time_4_dataloading) + '\n')
f.write('    Training (computing): {:.6f}(s)'.format(time_4_computing) + '\n')
f.close()

# new #
torch.save(model.module.state_dict(), opt.result_dir + '/model_' + sys.argv[0][0:-3] + '.pkl')