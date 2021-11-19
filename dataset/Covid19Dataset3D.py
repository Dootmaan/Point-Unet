# BraTS2020
import torch
import glob
import SimpleITK as sitk
import torch.utils.data
import os
from scipy import ndimage
import numpy as np
import cv2
import random
from config import config

img_size=config.input_img_size
crop_size=config.crop_size  # z,x,y
# img_size=[64,96,96]
# crop_size=[32,48,48]

print('Using patch size:', crop_size)

class Covid19Dataset3D(torch.utils.data.Dataset):
    def __init__(self,path,train=True,augment=True):
        self.data=[]
        self.label_seg=[]
        self.label_sr=[]
        self.train=train
        self.aug=augment
        images=sorted(glob.glob(path+"/*_ct.nii.gz"))
        labels=sorted(glob.glob(path+"/*_seg.nii.gz"))

        # Whether shuffle the dataset. Default not because we need to evaluate test dice for different models.
        # bundle = list(zip(images, labels))
        # random.shuffle(bundle)
        # images[:], labels[:] = zip(*bundle)

        train_frac, val_frac = 0.8, 0.2
        n_train = int(train_frac * len(images)) + 1
        n_val = min(len(images) - n_train, int(val_frac * len(images)))
        
        # Accelarate by loading all data into memory
        if train:
            print("train:",n_train, "folder:",path)
            images=images[:n_train]
            labels=labels[:n_train]
            for i in range(len(images)):
                print('Adding train sample:',images[i])
                image=sitk.ReadImage(images[i])
                image=self.reset_spacing(image,sitk.sitkBSpline)
                image_arr=sitk.GetArrayFromImage(image)
                image_arr[image_arr<-1000]=-1000
                image_arr[image_arr>500]=500
                lesion=sitk.ReadImage(labels[i])
                lesion=self.reset_spacing(lesion,sitk.sitkNearestNeighbor)
                lesion_arr=sitk.GetArrayFromImage(lesion)
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                if img.shape[1]!=img_size[1]:
                    print('skip',images[i])
                    continue
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)
            
        else:
            print("val:", n_val, "folder:",path)
            images=images[n_train:n_train+n_val]
            labels=labels[n_train:n_train+n_val]
            for i in range(len(images)):
                print('Adding val sample:',images[i])
                image=sitk.ReadImage(images[i])
                image=self.reset_spacing(image,sitk.sitkBSpline)
                image_arr=sitk.GetArrayFromImage(image)
                image_arr[image_arr<-1000]=-1000
                image_arr[image_arr>500]=500
                lesion=sitk.ReadImage(labels[i])
                lesion=self.reset_spacing(lesion,sitk.sitkNearestNeighbor)
                lesion_arr=sitk.GetArrayFromImage(lesion)
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                if img.shape[1]!=img_size[1]:
                    print('skip',images[i])
                    continue
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)

    def normalization(self,image_array):
        # image_array = np.int32(image_array)#整型
        max = image_array.max()
        min = image_array.min()
        #归一化
        image_array = 1.0*(image_array - min)/(max - min)
        #image_array = image_array.astype(int)#整型
        return image_array
    
    def cropMR(self,img,mask):
        # result=np.where(img!=0)
        # z_list=result[0]
        # x_list=result[1]
        # y_list=result[2]

        # x_max=x_list.max()
        # x_min=x_list.min()
        # y_max=y_list.max()
        # y_min=y_list.min()
        # z_max=z_list.max()
        # z_min=z_list.min()
        D,M,N=img.shape
        if M<192:
            return img,mask,img
        D=int(D/2)
        M=int(M/2)
        N=int(N/2)
        img=img[:,M-96:M+96,N-96:N+96]
        mask=mask[:,M-96:M+96,N-96:N+96]
        # img=img[D-64:D+64,M-96:M+96,N-96:N+96]
        # mask=mask[D-64:D+64,M-96:M+96,N-96:N+96]
        img=np.int32(img)

        return self.normalization(ndimage.interpolation.zoom(img,[img_size[0]/img.shape[0],img_size[1]/img.shape[1],img_size[2]/img.shape[2]],order=1)),self.normalization(ndimage.interpolation.zoom(mask,[2*img_size[0]/img.shape[0],2*img_size[1]/img.shape[1],2*img_size[2]/img.shape[2]],order=0)),self.normalization(ndimage.interpolation.zoom(img,[2*img_size[0]/img.shape[0],2*img_size[1]/img.shape[1],2*img_size[2]/img.shape[2]],order=1))

    def reset_spacing(self, itkimage,resamplemethod=sitk.sitkBSplineResamplerOrder3,newSpacing=[1.5,1.5,1.5]):

        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSpacing = np.array(newSpacing,float)
        factor = originSpacing / newSpacing
        newSize = originSize*factor
        newSize = newSize.astype(np.int) #spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        # sitk.WriteImage(itkimgResampled,'test.nii')
        return itkimgResampled

    def augment(self,img,label_seg,label_sr):
        # _img=img.copy()
        # _label_seg=label_seg.copy()
        # _label_sr=label_sr.copy()
        if random.random()<0.5:  #Flip
            if random.random()<0.5:
                for i in range(img.shape[0]):
                    img[i,:,:]=cv2.flip(img[i,:,:],0)
                for i in range(label_seg.shape[0]):
                    label_seg[i,:,:]=cv2.flip(label_seg[i,:,:],0)
                    label_sr[i,:,:]=cv2.flip(label_sr[i,:,:],0)
            else:
                for i in range(img.shape[0]):
                    img[i,:,:]=cv2.flip(img[i,:,:],1)
                for i in range(label_seg.shape[0]):
                    label_seg[i,:,:]=cv2.flip(label_seg[i,:,:],1)
                    label_sr[i,:,:]=cv2.flip(label_sr[i,:,:],1)
                    

        if random.random()<0.5:  #Shift
            vertical=np.random.randint(-img.shape[1]//8,img.shape[1]//8)
            horizon=np.random.randint(-img.shape[1]//8,img.shape[1]//8)
            M_img=np.float32([[0,1,horizon],[1,0,vertical]])
            M_label=np.float32([[0,1,2*horizon],[1,0,2*vertical]])
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))
        
        if random.random()<0.5: #Rotate
            degree=np.random.randint(0,360)
            M_img = cv2.getRotationMatrix2D(((img.shape[1]-1)/2.0,(img.shape[2]-1)/2.0),degree,1)
            M_label=cv2.getRotationMatrix2D(((label_seg.shape[1]-1)/2.0,(label_seg.shape[2]-1)/2.0),degree,1)
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))

        #random crop
        start_z=random.randint(0,img_size[0]-crop_size[0])
        start_x=random.randint(0,img_size[1]-crop_size[1])
        start_y=random.randint(0,img_size[2]-crop_size[2])

        return img[start_z:(start_z+crop_size[0]),start_x:(start_x+crop_size[1]),start_y:(start_y+crop_size[2])],label_seg[2*start_z:2*(start_z+crop_size[0]),2*start_x:2*(start_x+crop_size[1]),2*start_y:2*(start_y+crop_size[2])],label_sr[2*start_z:2*(start_z+crop_size[0]),2*start_x:2*(start_x+crop_size[1]),2*start_y:2*(start_y+crop_size[2])]

    def __len__(self):
        return len(self.label_seg)

    def __getitem__(self,index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None
            
        if self.train:
            if self.aug:
                return self.augment(self.data[index],self.label_seg[index],self.label_sr[index])
            else:
                return self.data[index],self.label_seg[index],self.label_sr[index]
        else:
            return self.data[index],self.label_seg[index],self.label_sr[index]

if __name__=='__main__':
    dataset=Covid19Dataset3D('/newdata/why/COVID-19-20_v2/Train',train=True)
    test=dataset.__getitem__(0)
    cv2.imwrite('img.png',test[0][32,:,:]*255)
    cv2.imwrite('seg.png',test[1][64,:,:]*255)
    cv2.imwrite('sr.png',test[2][64,:,:]*255)
    print(test)

