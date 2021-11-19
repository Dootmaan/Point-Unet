import torch
import torch.utils.data
import SimpleITK as sitk
import os
from scipy import ndimage
import numpy as np
import cv2
import random

class DSRLDataset3D(torch.utils.data.Dataset):
    def __init__(self, path, train=True,size=192,maskSR=False,slices=32):
        self.size=float(size)
        self.slices=slices
        self.data=list()
        self.label_seg=list()
        self.label_sr=list()

        if train == True:
            for i,folder in enumerate(os.listdir(path)):
                
                #88Z has no lesion mask(healthy),
                if folder=='88Z':
                    continue

                if os.path.exists(path+'/'+folder+'/'+'lesion.nrrd'):
                    if i<0.005*len(os.listdir(path)):
                        print('Processing training set:',folder)
                        image=sitk.ReadImage(path+'/'+folder+'/'+'image.nrrd')
                        image_arr=sitk.GetArrayFromImage(image)
                        lung=sitk.ReadImage(path+'/'+folder+'/'+'lung.nrrd')
                        lung_arr=sitk.GetArrayFromImage(lung)
                        lesion=sitk.ReadImage(path+'/'+folder+'/'+'lesion.nrrd')
                        lesion_arr=sitk.GetArrayFromImage(lesion)
                        # self.truncate_hu(image_arr)
                        # image_arr=self.normalization(image_arr)  #因为下面要和mask相乘
                        cropImg,cropMask,cropSR=self.cropCT(image_arr,lung_arr,lesion_arr)
                        if maskSR:
                            cropSR=cropSR*cropMask

                        # if cropImg.shape[0]==0.5*size and cropImg.shape[1]==size and cropImg.shape[2]==size:
                        self.data.append(self.normalization(cropImg))
                        self.label_seg.append(cropMask)
                        self.label_sr.append(self.normalization(cropSR))

        else:
            for i,folder in enumerate(os.listdir(path)):
                if os.path.exists(path+'/'+folder+'/'+'lesion.nrrd'):
                    if i>=0.8*len(os.listdir(path)):
                        print('Processing testing set:',folder)
                        image=sitk.ReadImage(path+'/'+folder+'/'+'image.nrrd')
                        image_arr=sitk.GetArrayFromImage(image)
                        lung=sitk.ReadImage(path+'/'+folder+'/'+'lung.nrrd')
                        lung_arr=sitk.GetArrayFromImage(lung)
                        lesion=sitk.ReadImage(path+'/'+folder+'/'+'lesion.nrrd')
                        lesion_arr=sitk.GetArrayFromImage(lesion)
                        # self.truncate_hu(image_arr)
                        # image_arr=self.normalization(image_arr)
                        cropImg,cropMask,cropSR=self.cropCT(image_arr,lung_arr,lesion_arr)
                        if maskSR:
                            cropSR=cropSR*cropMask

                        # if cropImg.shape[0]==0.5*size and cropImg.shape[1]==size and cropImg.shape[2]==size:
                        
                        self.data.append(self.normalization(cropImg))
                        self.label_seg.append(cropMask)
                        self.label_sr.append(self.normalization(cropSR))

    def truncate_hu(self,image_array):
        image_array[image_array > 500] = 500
        image_array[image_array <-1000] = -1000
    
    def normalization(self,image_array):
        max = image_array.max()
        min = image_array.min()
        #归一化
        image_array = 1.0*(image_array - min) / (max - min)
        #image_array = image_array.astype(int)#整型
        return image_array
    
    def cropCT(self,img,lung_mask,lesion_mask):
        result=np.where(lung_mask!=0)
        z_list=result[0]
        x_list=result[1]
        y_list=result[2]

        x_max=x_list.max()
        x_min=x_list.min()
        y_max=y_list.max()
        y_min=y_list.min()
        z_max=z_list.max()
        z_min=z_list.min()

        zoomed_lrimg=ndimage.interpolation.zoom(img[z_min:z_max,x_min:x_max,y_min:y_max],[self.slices/(z_max-z_min),self.size/(x_max-x_min),self.size/(y_max-y_min)],order=3)
        self.truncate_hu(zoomed_lrimg)
        zoomed_lrimg=self.normalization(zoomed_lrimg)
        zoomed_hrimg=ndimage.interpolation.zoom(img[z_min:z_max,x_min:x_max,y_min:y_max],[2*self.slices/(z_max-z_min),2*self.size/(x_max-x_min),2*self.size/(y_max-y_min)],order=3)
        self.truncate_hu(zoomed_hrimg)
        zoomed_hrimg=self.normalization(zoomed_hrimg)
        zoomed_lrlungmask=ndimage.interpolation.zoom(lung_mask[z_min:z_max,x_min:x_max,y_min:y_max],[self.slices/(z_max-z_min),self.size/(x_max-x_min),self.size/(y_max-y_min)],order=0)
        zoomed_hrlungmask=ndimage.interpolation.zoom(lung_mask[z_min:z_max,x_min:x_max,y_min:y_max],[2*self.slices/(z_max-z_min),2*self.size/(x_max-x_min),2*self.size/(y_max-y_min)],order=0)
        zoomed_lesionmask=ndimage.interpolation.zoom(lesion_mask[z_min:z_max,x_min:x_max,y_min:y_max],[2*self.slices/(z_max-z_min),2*self.size/(x_max-x_min),2*self.size/(y_max-y_min)],order=0)

        return zoomed_lrimg*zoomed_lrlungmask,zoomed_hrlungmask*zoomed_lesionmask,zoomed_hrimg*zoomed_hrlungmask
    
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
            vertical=random.randint(-img.shape[1]//8,img.shape[1]//8)
            horizon=random.randint(-img.shape[1]//8,img.shape[1]//8)
            M_img=np.float32([[0,1,horizon],[1,0,vertical]])
            M_label=np.float32([[0,1,2*horizon],[1,0,2*vertical]])
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))
        
        if random.random()<0.5: #Rotate
            degree=random.randint(0,360)
            M_img = cv2.getRotationMatrix2D(((img.shape[1]-1)/2.0,(img.shape[2]-1)/2.0),degree,1)
            M_label=cv2.getRotationMatrix2D(((label_seg.shape[1]-1)/2.0,(label_seg.shape[2]-1)/2.0),degree,1)
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))

        return img,label_seg,label_sr

    def __len__(self):
        return len(self.label_seg)

    def __getitem__(self, index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None

        return self.augment(self.data[index],self.label_seg[index],self.label_sr[index])  #KeyError

if __name__=='__main__':
    dataset=DSRLDataset3D('/newdata/zh/COVID19-ZJ/COVID19-Dataset-ALL-Formatted',train=True,size=192,slices=32)
    test=dataset.__getitem__(0)
    cv2.imwrite('img.png',test[0][15,:,:]*255)
    cv2.imwrite('seg.png',test[1][31,:,:]*255)
    cv2.imwrite('sr.png',test[2][31,:,:]*255)
    print(test)

