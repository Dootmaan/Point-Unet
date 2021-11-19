import torch
import glob
import SimpleITK as sitk
import torch.utils.data
import os
from scipy import ndimage
import numpy as np
import cv2
import random
# from config import config

class PCDataset3D(torch.utils.data.Dataset):
    def __init__(self,path,mode='train',augment=False):
        self.data=[]
        self.label=[]
        self.originLable=[]
        self.mode=mode
        self.aug=augment
        

        # Whether shuffle the dataset. Default not because we need to evaluate test dice for different models.
        # bundle = list(zip(images, coarse, labels))
        # random.shuffle(bundle)
        # images[:], coarse[:], labels[:] = zip(*bundle)


        # print("shape:",np.load(labels[0]).shape)
        
        # Accelarate by loading all data into memory
        if mode=='train':
            images=sorted(glob.glob(os.path.join(path, "PCdataset/*_Training_input.npy")))
            labels=sorted(glob.glob(os.path.join(path, "PCdataset/*_Training_label.npy")))
            print("train:",len(images), "folder:",path)
            for i in range(len(images)):
                print('Adding train sample:',images[i])
                image_arr=np.load(images[i])
                lesion_arr=np.load(labels[i])
                self.data.append(image_arr)
                self.label.append(lesion_arr)
            
        # elif mode=='val':
        #     print("val:", n_val, "folder:",path)
        #     images=images[n_train:n_train+n_val]
        #     labels=labels[n_train:n_train+n_val]
        #     for i in range(len(images)):
        #         print('Adding val sample:',images[i])
        #         image=sitk.ReadImage(images[i])
        #         image_arr=sitk.GetArrayFromImage(image)
        #         lesion=sitk.ReadImage(labels[i])
        #         lesion_arr=sitk.GetArrayFromImage(lesion)
        #         lesion_arr[lesion_arr>1]=1  # 只做WT分割
        #         img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
        #         label_seg[label_seg<0.5]=0.
        #         label_seg[label_seg>=0.5]=1.
        #         self.data.append(img)
        #         self.label.append(label_seg)
        #         self.coarse.append(label_sr)
        
        elif mode=='test':
            
            images=sorted(glob.glob(os.path.join(path, "PCdataset/*_Testing_input.npy")))
            labels=sorted(glob.glob(os.path.join(path, "PCdataset/*_Testing_label.npy")))
            originLabels=sorted(glob.glob(os.path.join(path, "PCdataset/*_Testing_originLabel.npy")))
            print("test:", len(images), "folder:",path)
            for i in range(len(images)):
                print('Adding test sample:',images[i])
                image_arr=np.load(images[i])
                lesion_arr=np.load(labels[i])
                originlabel_arr=np.load(originLabels[i])
                self.data.append(image_arr)
                self.label.append(lesion_arr)
                self.originLable.append(originlabel_arr)
            
        else:
            print("Not implemented for this dataset. (No need)")
            raise Exception()

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None
        
        if self.mode=='test':
            return self.data[index],self.label[index],self.originLable[index]
        else:
            return self.data[index],self.label[index]

if __name__=='__main__':
    dataset=PCDataset3D('/newdata/why/BraTS20/',mode='train')
    test=dataset.__getitem__(0)
    cv2.imwrite('img.png',test[0][32,:,:]*255)
    cv2.imwrite('seg.png',test[1][32,:,:]*255)
    cv2.imwrite('sr.png',test[2][32,:,:]*255)
    print(test)

