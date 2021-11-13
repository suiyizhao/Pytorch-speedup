# Template
## Script Category Description
| Category | script |
| ---- | ---- |
| **comparison script** | **train.py, loader.py** |
| for single-machine-multiple-cards training | train_DP.py, train_DDP.py |
| for mix-precision training | train_amp.py |
| for DALI data-loader | loader_DALI.py |  

Note: The comment `# new #` in script represents newly added code block (compare to comparison script, e.g., train.py)
## Environment
- CPU: Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- GPU: RTX 2080Ti
- OS: Ubuntu 18.04.3 LTS
- DL framework: Pytorch 1.6.0, Torchvision 0.7.0
## Single-machine-multiple-cards training
### *train_DP.py* 
> **Usage:** 
> ```
> cd Template/src
> python train_DP.py
> ```
> **Description:**  
> Using `nn.DataParallel` in Pytorch:  
> DataParallel is very convenient to use, we just need to use DataParallel to package our model:
> ```
> model = nn.DataParallel(model)
> ```
### *train_DDP.py* 
> **Usage: (two cards for example)** 
> ```
> cd Template/src
> CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_DDP.py
> ```
