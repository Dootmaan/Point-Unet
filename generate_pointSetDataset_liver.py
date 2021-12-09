import numpy as np
import random
import torch as pt
from config import config
from dataset.MVILiverDataset3D import MVILiverDataset3D
from model.SaliencyAttentionNet import SaliencyAttentionNet

total_num_points=800000
threshold=0.5

batch_size = 1
model_path = '/newdata/why/Saved_models'
crop_size = config.crop_size
size = crop_size[2] * 2  #用于最后cv2显示
img_size = config.input_img_size


def contextAwareSamplingTrain(image, label):
    pointSet = list()
    pointSet_label = list()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if not image[i, j, k]==0:
                    pointSet.append([i, j, k, image[i, j, k]])
                    pointSet_label.append(label[i,j,k])

    pointSet=np.array(pointSet)
    pointSet_label=np.array(pointSet_label)

    none_tumor = list(np.where(pointSet_label == 0)[0])
    tumor = list(np.where(pointSet_label > 0)[0])
    print(len(tumor))
    queried_idx = tumor + random.sample(none_tumor, k=total_num_points - len(tumor))
    queried_idx = np.array(queried_idx)
    random.shuffle(queried_idx)

    queried_points=pointSet[queried_idx,...]
    queried_points[:,0:3]/=image.shape
    queried_labels=pointSet_label[queried_idx,...]

    return queried_points, queried_labels


def contextAwareSamplingTest(image, attention_map):
    pointSet = list()
    pointSet_label = list()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if not image[i, j, k]==0:
                    pointSet.append([i, j, k, image[i, j, k]])
                    pointSet_label.append(attention_map[i,j,k])

    pointSet=np.array(pointSet)
    pointSet_label=np.array(pointSet_label)

    none_tumor = list(np.where(pointSet_label <threshold)[0])
    tumor = list(np.where(pointSet_label >= threshold)[0])
    queried_idx = tumor + random.sample(none_tumor, k=total_num_points - len(tumor))
    queried_idx = np.array(queried_idx)
    random.shuffle(queried_idx)

    queried_points=pointSet[queried_idx,...]
    queried_points[:,0:3]/=image.shape
    queried_labels=pointSet_label[queried_idx,...]

    return queried_points, queried_labels


trainset = MVILiverDataset3D('/newdata/why/MVI_Liver_Formatted', mode='train',augment=False)
# valset=BraTSDataset3D('/newdata/why/MVI_Liver_Formatted',mode='val')
testset = MVILiverDataset3D('/newdata/why/MVI_Liver_Formatted', mode='test')

train_dataset = pt.utils.data.DataLoader(trainset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)
# val_dataset=pt.utils.data.DataLoader(valset,batch_size=1,shuffle=True,drop_last=True)
test_dataset = pt.utils.data.DataLoader(testset,
                                        batch_size=1,
                                        shuffle=True,
                                        drop_last=True)

model = SaliencyAttentionNet(in_ch=1).cuda()
model.load_state_dict(
    pt.load(model_path +
            '/PointUnet/SaliencyAttentionNet_3D_Liver_patch-free_bs1_best.pt',
            map_location='cpu'))
model.eval()

print('generating training samples')
for i, data in enumerate(train_dataset):
    (_, labels_seg, inputs) = data
    # inputs = pt.autograd.Variable(inputs).type(
    #     pt.FloatTensor).cuda().unsqueeze(1)
    # labels_seg = pt.autograd.Variable(labels_seg).type(
    #     pt.FloatTensor).cuda().unsqueeze(1)
    # with pt.no_grad():
    #     outputs_seg = model(inputs)

    pointset, gt = contextAwareSamplingTrain(inputs.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        labels_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Training_input.npy'.
        format(i), pointset)
    np.save('/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Training_originLabel.npy'.
        format(i), labels_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Training_label.npy'.
        format(i), gt)

print('generating testing samples')
for i, data in enumerate(test_dataset):
    (_, labels_seg, inputs) = data
    inputs = pt.autograd.Variable(inputs).type(
        pt.FloatTensor).cuda().unsqueeze(1)
    labels_seg = pt.autograd.Variable(labels_seg).type(
        pt.FloatTensor).cuda().unsqueeze(1)
    with pt.no_grad():
        outputs_seg = model(inputs)


    pointset, gt = contextAwareSamplingTest(inputs.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        outputs_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Testing_input.npy'.
        format(i), pointset)
    np.save('/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Testing_originLabel.npy'.
        format(i), labels_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/MVI_Liver_Formatted/PCdataset/Liver{:03d}_Testing_label.npy'.
        format(i), gt)

print('finished')