import numpy as np
import random
import torch as pt
from config import config
from dataset.BraTSDataset3D import BraTSDataset3D
from model.SaliencyAttentionNet import SaliencyAttentionNet

density=0.5  # the higher the denser. 1 maximum.

batch_size = 1
model_path = '/newdata/why/Saved_models'
crop_size = config.crop_size
size = crop_size[2] * 2  #用于最后cv2显示
img_size = config.input_img_size


def contextAwareSampling(image, label, attention_map):
    pointSet = list()
    pointSet_label = list()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                # if attention_map[i, j, k] > dense_threshold:
                #     pointSet.append([i, j, k, image[i, j, k]])
                #     pointSet_label.append([i, j, k, label[i, j, k]])
                # else:
                #     if random.random() < random_sample_prob:
                #         pointSet.append([i, j, k, image[i, j, k]])
                #         pointSet_label.append([i, j, k, label[i, j, k]])

                if random.random() < density* attention_map[i, j, k]:
                    pointSet.append([i, j, k, image[i, j, k]])
                    pointSet_label.append(label[i, j, k])

    return np.array(pointSet), np.array(pointSet_label)


trainset = BraTSDataset3D('/newdata/why/BraTS20', mode='train',augment=False)
# valset=BraTSDataset3D('/newdata/why/BraTS20',mode='val')
testset = BraTSDataset3D('/newdata/why/BraTS20', mode='test')

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
            '/PointUnet/SaliencyAttentionNet_3D_BraTS_patch-free_bs1_best.pt',
            map_location='cpu'))
model.eval()

print('generating training samples')
for i, data in enumerate(train_dataset):
    (_, labels_seg, inputs) = data
    inputs = pt.autograd.Variable(inputs).type(
        pt.FloatTensor).cuda().unsqueeze(1)
    labels_seg = pt.autograd.Variable(labels_seg).type(
        pt.FloatTensor).cuda().unsqueeze(1)
    with pt.no_grad():
        outputs_seg = model(inputs)

    pointset, gt = contextAwareSampling(inputs.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        labels_seg.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        outputs_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/BraTS20/PCdataset/BraTS20_{:03d}_Training_input.npy'.
        format(i), pointset)
    np.save(
        '/newdata/why/BraTS20/PCdataset/BraTS20_{:03d}_Training_label.npy'.
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


    pointset, gt = contextAwareSampling(inputs.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        labels_seg.squeeze(0).squeeze(0).cpu().data.numpy(),
                                        outputs_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/BraTS20/PCdataset/BraTS20_{:03d}_Testing_input.npy'.
        format(i), pointset)
    np.save('/newdata/why/BraTS20/PCdataset/BraTS20_{:03d}_Testing_originLabel.npy'.
        format(i), labels_seg.squeeze(0).squeeze(0).cpu().data.numpy())
    np.save(
        '/newdata/why/BraTS20/PCdataset/BraTS20_{:03d}_Testing_label.npy'.
        format(i), gt)

print('finished')