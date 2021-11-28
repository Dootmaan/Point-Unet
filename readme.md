# Point-Unet 

## Update 2021/11/28

It's weird that RandLA-Net can only achieve ~0.6 dice in the testset, which is inconsistent with the Point-Unet paper. I also tried to cut down the number of points to a half and uses the original Adam optimized and it is still around 0.6 dice. Please send me a pull request if you find any bugs in my code.

## Update 2021/11/23

Introduce Adam8bit for optmizer. More sampling points available now. please install bitsandbytes according to your cuda toolkit version before using. Now you must use Python3.6+ to run the code.

```
pip install bitsandbytes-cudaXXX
```

For me i use CUDA 11.1 so the command should be 'pip install bitsandbytes-cuda111'

**Please also note that maybe you have to install torch-points-kernels for yourself, since the library is compiled on my machine and may cause errors if you directly run the code. Remember to remove the torch_points_kernels folder in this repo after you installed the library.**

---

This is an unofficial implementation of the MICCAI 2021 paper *Point-Unet: A Context-Aware Point-Based Neural Network for Volumetric Segmentation*

The authors claim that they will provide their code in late November, so this repo will then be depreciated. I am currently also working on a similar field of this paper and want to compare our method with Point-Unet, so I implemented the method by myself to analyze the result early. 

Note that the method has been modified to make it runnable on a 11G memory graphic card, and the experiment is only conducted for BRATS WT segmentation. The experimental results are only for reference, and if you want to test the result for your own paper please wait for the official version of the code.

## How to use this repo?

Keep in mind that Point-Unet is a two-stage coarse-to-fine framework and cannot be trainned end-to-end. So, to run the code, you have to firstly train a Saliency Attention Network by running:

```
    CUDA_VISIBLE_DEVICES=0 python3 -u train_saliencyAttentionNet.py
```

This model uses the original BRATS dataset for training. Remember to specify the path to the dataset.

Secondly, we need to sample the 3D images into point clouds with the help of Saliency Attention Network. To realize this goal, run:

```
    CUDA_VISIBLE_DEVICES=0 python3 -u generate_pointSetDataset.py
```

You can adjust the density of sampling by changing the density parameter in this file. This python file will generate a new dataset for Point-Unet.

At last, we can train Point-Unet with:
```
    CUDA_VISIBLE_DEVICES=0 python3 -u train_pointunet.py
```

A simple testing function is integrated in this file and can be used as an insight to the model performance.

## Warnings

- This project is still under construction and may contain some errors.
- Point-Unet is not implemented currently and this repo currently only uses RandLA
- This implementation is not exactly the same as what is described in the paper.