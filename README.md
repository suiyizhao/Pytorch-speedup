# Pytorch-speedup
## Script Category Description
| Category | script |
| ---- | ---- |
| **comparison script** | **train.py, loader.py** |
| for single-machine-multiple-cards training | train_DP.py, train_DDP.py |
| for mixed-precision training | train_amp.py |
| for DALI data loading | loader_DALI.py |  

Note: The comment `# new #` in script represents newly added code block (compare to comparison script, e.g., train.py)
## Environment
- CPU: Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- GPU: RTX 2080Ti
- OS: Ubuntu 18.04.3 LTS
- DL framework: Pytorch 1.6.0, Torchvision 0.7.0
## Single-machine-multiple-cards training (two cards for example)
### *train_DP.py* -- Parallel computing using [nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel)
> **Usage:** 
> ```
> cd Template/src
> python train_DP.py
> ```
> **Superiority:**  
>     - Easy to use  
>     - Accelerate training (inconspicuous)  
> **Weakness:**  
>     - Unbalanced load  
> **Description:**   
> DataParallel is very convenient to use, we just need to use DataParallel to package the model:  
> ```
> model = ...
> model = nn.DataParallel(model)
> ```
### *train_DDP.py* -- Parallel computing using [torch.distributed](https://pytorch.org/docs/stable/distributed.html#)
> **Usage:** 
> ```
> cd Template/src
> CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_DDP.py
> ```
> **Superiority:**  
>     - balanced load  
>     - Accelerate training (conspicuous)  
> **Weakness:**  
>     - Hard to use  
> **Description:**  
> Unlike `DataParallel` who control multiple GPUs via single-process, `distributed` creates multiple process. we just need to accomplish one code and torch will automatically assign it to n processes, each running on corresponding GPU.  
> To config distributed model via `torch.distributed`, the following steps needed to be performed:  
> 1. Get current process index:  
> ```
> parser = argparse.ArgumentParser()
> parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
> opt = parser.parse_args()
> # print(opt.local_rank)
> ```
> 2. Set the backend and port used for communication between GPUs:  
> ```
> dist.init_process_group(backend='nccl')
> ```
> 3. Config current device according to the `local_rank`:  
> ```
> torch.cuda.set_device(opt.local_rank)
> ```
> 4. Config data sampler:
> ```
> dataset = ...
> sampler = distributed.DistributedSampler(dataset)
> dataloader = DataLoader(dataset=dataset, ..., sampler=sampler)
> ```
> 5. Package the model：
> ```
> model = ...
> model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
> model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[opt.local_rank])
> ```
## Mixed-precision training
### *train_amp.py* -- Mixed-precision training using [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html#)
> **Usage:** 
> ```
> cd Template/src
> python train_amp.py
> ```
> **Superiority:**  
>     - Easy to use  
>     - Accelerate training (conspicuous for heavy model)  
> **Weakness:**  
>     - Accelerate training (inconspicuous for light model)  
> **Description:**   
> Mixed-precision training is a set of techniques that allows us to use fp16 without causing our model training to diverge.  
> To config mixed-precision training via `torch.cuda.amp`, the following steps needed to be performed:
> 1. Instantiate `GradScaler` object:
> ```
> scaler = torch.cuda.amp.GradScaler()
> ```
> 2. Modify the traditional optimization process:  
> ```
> # Before:
> optimizer.zero_grad()
> preds = model(imgs)
> loss = loss_func(preds, labels)
> loss.backward()
> optimizer.step()
> 
> # After:
> optimizer.zero_grad()
> with torch.cuda.amp.autocast():
>     preds = model(imgs)
>     loss = loss_func(preds, labels)
> scaler.scale(loss).backward()
> scaler.step(optimizer)
> scaler.update()
> ```
## DALI data loading
### *loader_DALI.py* -- Data loading using [nvidia.dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
> **Prerequisite:**  
>     - [NVIDIA Driver](https://www.nvidia.com/drivers) supporting [CUDA 10.0](https://developer.nvidia.com/cuda-downloads) or later (i.e., 410.48 or later driver releases)  
>     - PyTorch 0.4 or later  
>     - Data organization format that matches the code, the format that matches the loader_DALI.py is as follows:  
>       &emsp;/dataset / train or test / img or gt / sub_dirs / imgs [[View]](https://github.com/suiyizhao/Template/blob/master/src/loader_DALI.py#:~:text=self.batch_size%20%3D%20batch_size-,img_paths%20%3D%20sorted(glob.glob(data_source%20%2B%20%27/train%27%20%2B%20%27/blurry%27%20%2B%20%27/*/*.*%27)),gt_paths%20%3D%20sorted(glob.glob(data_source%20%2B%20%27/train%27%20%2B%20%27/sharp%27%20%2B%20%27/*/*.*%27)),-self.paths%20%3D%20list)  
> **Usage:** 
> ```
> pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
> cd Template/src
> python loader_DALI.py --data_source /path/to/dataset
> ```
> **Superiority:**  
>     - Easy to use  
>     - Accelerate data loading  
> **Weakness:**  
>     - Occupy video memory  
> **Description:**   
> NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks and an execution engine that accelerates the data pipeline for computer vision and audio deep learning applications.  
> To load dataset using DALI, the following steps needed to be performed:  
> 1. Config external input iterator:  
> ```
> eii = ExternalInputIterator(data_source=opt.data_source, batch_size=opt.batch_size, shuffle=True)
> ```
> ```
> # A demo of external input iterator
> class ExternalInputIterator(object):
>     def __init__(self, data_source, batch_size, shuffle):
>         self.batch_size = batch_size
>         
>         img_paths = sorted(glob.glob(data_source + '/train' + '/blurry' + '/*/*.*'))
>         gt_paths = sorted(glob.glob(data_source + '/train' + '/sharp' + '/*/*.*'))
>         self.paths = list(zip(*(img_paths,gt_paths)))
>         if shuffle:
>             random.shuffle(self.paths)
> 
>     def __iter__(self):
>         self.i = 0
>         return self
> 
>     def __next__(self):
>         imgs = []
>         gts = []
> 
>         if self.i >= len(self.paths):
>             self.__iter__()
>             raise StopIteration
> 
>         for _ in range(self.batch_size):
>             img_path, gt_path = self.paths[self.i % len(self.paths)]
>             imgs.append(np.fromfile(img_path, dtype = np.uint8))
>             gts.append(np.fromfile(gt_path, dtype = np.uint8))
>             self.i += 1
>         return (imgs, gts)
> 
>     def __len__(self):
>         return len(self.paths)
> 
>     next = __next__
> ```
> 2. Config pipeline:
> ```
> pipe = externalSourcePipeline(batch_size=opt.batch_size, num_threads=opt.num_workers, device_id=0, seed=opt.seed, external_data = eii, resize=opt.resize, crop=opt.crop)
> ```
> ```
> # A demo of pipeline
> @pipeline_def
> def externalSourcePipeline(external_data, resize, crop):
>     imgs, gts = fn.external_source(source=external_data, num_outputs=2)
>     
>     crop_pos = (fn.random.uniform(range=(0., 1.)), fn.random.uniform(range=(0., 1.)))
>     flip_p = (fn.random.coin_flip(), fn.random.coin_flip())
>     
>     imgs = transform(imgs, resize, crop, crop_pos, flip_p)
>     gts = transform(gts, resize, crop, crop_pos, flip_p)
>     return imgs, gts
> 
> def transform(imgs, resize, crop, crop_pos, flip_p):
>     imgs = fn.decoders.image(imgs, device='mixed')
>     imgs = fn.resize(imgs, resize_y=resize)
>     imgs = fn.crop(imgs, crop=(crop,crop), crop_pos_x=crop_pos[0], crop_pos_y=crop_pos[1])
>     imgs = fn.flip(imgs, horizontal=flip_p[0], vertical=flip_p[1])
>     imgs = fn.transpose(imgs, perm=[2, 0, 1])
>     imgs = imgs/127.5-1
>     
>     return imgs
> ```
> 3. Instantiate DALIGenericIterator object:
> ```
> dgi = DALIGenericIterator(pipe, output_map=["imgs", "gts"], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
> ```
> 4. Read data:
> ```
> for i, data in enumerate(dgi):
>     imgs = data[0]['imgs']
>     gts = data[0]['gts']
> ```
