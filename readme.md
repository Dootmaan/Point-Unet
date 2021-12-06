# Point-Unet 

This implementation conducts WT segmentation with T2 input by default, which is not the same of the original paper's experiment. A new branch that conducts the same experiment of the paper will be updated soon.

*This repo is basically built from scratch by myself and is only used for my personal experiment so it may cannot be directly used unless some key factors such as the dataset path is correctly configured. After i finish my current work i will try to sort out the code and make it easier to use. You are also welcome for proposing a pull request.* 

The update log below is very important and you are encouraged to read them before using the code.

## Update 2021/12/6

I have read the official code and find the primary difference between my implementation and the original ones is the sampling methods. It turns out that the Context-Aware Sampling will sample every lesion voxel into points and add up other random sampling points to get a total number of 365000 points. I have already changed my code to such sampling method and achieved a 0.81 DSC rn. The experiment is still running so the final result may be even better.

However by doing so it is just impossible to make it trainable on a 11G GTX 1080Ti, so I changed to use an Tesla A4000 instead. The memory cost during training is 15440M, which indicates that **you have to use a grahical cards with >=16G memory** instead. Adam8Bit is abandoned now since it doesnt help too much under this circumstances.

Besides there are also some other changes:
- i add a erosion step after the infaltion to eliminate some segmentation noise.
- the pretrained weights of Point-Unet is uploaded to [here](https://drive.google.com/file/d/1Ur-He3A7UwxqyBnthVTsIoyjUXPaH-PH/view?usp=sharing)

## Update 2021/12/4

The official code is realeased today. Please check [here](https://github.com/VinAIResearch/Point-Unet) to learn more. The official code is based on TensorFlow and I will modify this PyTorch implementation accordingly. Currently this repo is **depreciated** because of the poor performance. Please use the official code for your project. 

I will update the code after achieving a similar result with PyTorch.
 
## Update 2021/12/1

With Adam8bit, the model finally reaches a 0.66 dice. However, for a task using T2 as input and WT as label, the saliency attention network can already achieve a 0.815 DSC. The official code is still unavailable so i'll keep working on this project.

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
- This implementation conducts WT segmentation with T2 input by default, which is not the same of the original paper's experiment. If you want to conduct the same experiment as the paper, you have to modify the BraTSDataset3D.py to read other modalities' image and change the input channel of Saliency Attention Network to 4. You also have to generate the points for all these modalities and eventually change the output channel of Point-Unet to 4. I will add a branch of this after finish my current work.
- This implementation is not exactly the same as what is described in the paper (such as the channel numbers).