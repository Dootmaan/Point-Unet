import numpy as np
import random
import torch as pt
from config import config
from dataset.BraTSDataset3D import BraTSDataset3D
from dataset.MVILiverDataset3D import MVILiverDataset3D
from model.SaliencyAttentionNet import SaliencyAttentionNet
from model.PointUnet import PointUnet
from config import config
import time
model_path='/newdata/why/Saved_models'

crop_size = config.crop_size
size = crop_size[2] * 2  # 用于最后cv2显示
img_size = config.input_img_size

testset1 = BraTSDataset3D('/newdata/why/BraTS20', mode='test', augment=False)
testset2 = MVILiverDataset3D(
    '/newdata/why/MVI_Liver_Formatted', mode='test', augment=False)

test_dataset1 = pt.utils.data.DataLoader(testset1,
                                        batch_size=1,
                                        shuffle=True,
                                        drop_last=True)

test_dataset2 = pt.utils.data.DataLoader(testset2,
                                        batch_size=1,
                                        shuffle=True,
                                        drop_last=True)

device = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')
model1 = SaliencyAttentionNet().to(device)
model2 = PointUnet(4, 2,device=device).to(device)

model1.load_state_dict(pt.load(model_path +
            '/PointUnet/SaliencyAttentionNet_3D_BraTS_patch-free_bs1_best.pt',
            map_location='cpu'))
model2.load_state_dict(pt.load(model_path+'/PointUnet/PointUNet_3D_BraTS_patch-free_bs1_best.pt',map_location = 'cpu'))

model3 = SaliencyAttentionNet().to(device)
model4 = PointUnet(4, 2,device=device).to(device)

model3.load_state_dict(pt.load(model_path +
            '/PointUnet/SaliencyAttentionNet_3D_Liver_patch-free_bs1_best.pt',
            map_location='cpu'))
model4.load_state_dict(pt.load(model_path+'/PointUnet/PointUNet_3D_Liver_patch-free_bs1_best.pt',map_location = 'cpu'))


model1.eval()
model2.eval()
model3.eval()
model4.eval()


def TestModel():
    start_time = time.time()

    for whatever, data in enumerate(test_dataset1):
        print(whatever)
        (_, labels, inputs) = data  # use label_sr as input
        pointSet=[]
        inputs3D = pt.autograd.Variable(inputs).type(pt.FloatTensor).to(device).unsqueeze(1)
        with pt.no_grad():
            outputs3D = model1(inputs3D)
        output_list = np.array(outputs3D.squeeze(0).squeeze(0).cpu().data.numpy())
        # output_list[output_list<0.5]=0
        # output_list[output_list>=0.5]=1 

        image=inputs.squeeze(0).squeeze(0).cpu().data.numpy()
        for i in range(output_list.shape[0]):
            for j in range(output_list.shape[1]):
                for k in range(output_list.shape[2]):
                    pointSet.append([i, j, k, image[i, j, k]])
        output_list=output_list.flatten()
        pointSet=np.array(pointSet)
        none_tumor = list(np.where(output_list <0.5)[0])
        tumor = list(np.where(output_list >= 0.5)[0])
        print(len(tumor))
        queried_idx = tumor + random.sample(none_tumor, k=365000 - len(tumor))
        queried_idx = np.array(queried_idx)
        random.shuffle(queried_idx)

        queried_points=pointSet[queried_idx,...]
        queried_points[:,0:3]/=image.shape
        # queried_labels=output_list[queried_idx,...]

        # pointSet=np.array(pointSet)
        inputs3D = pt.autograd.Variable(pt.from_numpy(queried_points)).type(pt.FloatTensor).unsqueeze(0).to(device)
        with pt.no_grad():
            outputs3D = model2(inputs3D)
        
        outputs3D[outputs3D<0.5]=0
        outputs3D[outputs3D>=0.5]=1
        # output_list=outputs3D.squeeze(0).squeeze(0).cpu().data.numpy()

        # output_list[output_list < 0.5] = 0
        # output_list[output_list >= 0.5] = 1

        # final_img = np.zeros(shape=(2 * img_size[1], 2 * 2 * img_size[2]))
        # final_img[:, :2 * img_size[2]] = output_list[0, 0, 64, :, :] * 255
        # final_img[:, 2 * img_size[2]:] = label_list[0, 0, 64, :, :] * 255
        # cv2.imwrite('TestPhase_Res_patchfree_Liver.png', final_img)

        # pr_sum = output_list.sum()
        # gt_sum = label_list.sum()
        # pr_gt_sum = np.sum(output_list[label_list == 1])
        # dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        # dice_sum += dice
        # # print("dice:",dice)

        # try:
        #     hausdorff = hd95(
        #         output_list.squeeze(0).squeeze(0),
        #         label_list.squeeze(0).squeeze(0))
        # except:
        #     hausdorff = 0
        # jaccard = jc(
        #     output_list.squeeze(0).squeeze(0),
        #     label_list.squeeze(0).squeeze(0))

        # print("dice:", dice, ";hd95:", hausdorff, ";jaccard:", jaccard)

        # hd_sum += hausdorff
        # jc_sum += jaccard
    
    for whatever, data in enumerate(test_dataset2):
        print(whatever)
        (_, labels, inputs) = data  # use label_sr as input
        pointSet=[]
        inputs3D = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
        with pt.no_grad():
            outputs3D = model3(inputs3D)
        output_list = np.array(outputs3D.squeeze(0).squeeze(0).cpu().data.numpy())
        image=inputs.squeeze(0).squeeze(0).cpu().data.numpy()
        for i in range(output_list.shape[0]):
            for j in range(output_list.shape[1]):
                for k in range(output_list.shape[2]):
                    pointSet.append([i, j, k, image[i, j, k]])
        output_list=output_list.flatten()
        pointSet=np.array(pointSet)
        none_tumor = list(np.where(output_list <0.5)[0])
        tumor = list(np.where(output_list >= 0.5)[0])
        print(len(tumor))
        queried_idx = tumor + random.sample(none_tumor, k=800000 - len(tumor))
        queried_idx = np.array(queried_idx)
        random.shuffle(queried_idx)

        queried_points=pointSet[queried_idx,...]
        queried_points[:,0:3]/=image.shape

        inputs3D = pt.autograd.Variable(pt.from_numpy(queried_points)).type(pt.FloatTensor).unsqueeze(0).to(device)
        with pt.no_grad():
            outputs3D = model4(inputs3D)
        
        outputs3D[outputs3D<0.5]=0
        outputs3D[outputs3D>=0.5]=1
        # output_list=outputs3D.squeeze(0).squeeze(0).cpu().data.numpy()

    total_time=time.time()-start_time
    print(total_time / (len(test_dataset1)+len(test_dataset2)))
    return total_time / (len(test_dataset1)+len(test_dataset2))

TestModel()