# Template
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
### *train_DP.py* -- Parallel computing using `nn.DataParallel`
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
### *train_DDP.py* -- Parallel computing using `torch.distributed`
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
> 5. Package model：
> ```
> model = ...
> model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
> model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[opt.local_rank])
> ```
## Mixed-precision training
### *train_amp.py* -- mixed-precision training using `torch.cuda.amp`
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
### *loader_DALI.py* -- Data loading using `nvidia.dali`
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
