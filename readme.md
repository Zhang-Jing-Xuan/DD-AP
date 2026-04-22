**Distill the biased datasets**
 
* Step 1: Run ECS.py to obtain the biasd score
  
```python ECS.py  --dataset CMNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --data_path MNIST_align0.95_severity_4.pth --log_path ECS_50_CMNIST_align0.95_severity_4```

* Step 2: Run main_BAXX.py to distilled BA samples

```python main_BADM.py  --dataset CMNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --data_path MNIST_align0.95_severity_4.pth --log_path BADM_50_MNIST_align0.95_severity_4 --Iteration 20000```

* Step 3: Run main_BABCXX.py to get the final results
  
```python main_BABCDM.py  --dataset CMNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --data_path MNIST_align0.95_severity_4.pth --log_path BABCDM_50_CMNIST_align0.95_severity_4```