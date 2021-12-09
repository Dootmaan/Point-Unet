from model.SaliencyAttentionNet import SaliencyAttentionNet
# from model.SaliencyAttentionNet import UNet
from dataset.MVILiverDataset3D import MVILiverDataset3D
from config import config
import torch as pt
import numpy as np
# from loss.FALoss3D import FALoss3D
import cv2
from medpy.metric.binary import jc, dc, hd95
from scipy import ndimage
from loss.DiceLoss import BinaryDiceLoss
import time

lr = 0.00001
epoch = 100
batch_size = 1
model_path = '/newdata/why/Saved_models'
crop_size = config.crop_size
size = crop_size[2] * 2  #用于最后cv2显示
img_size = config.input_img_size

print(
    'Please note that this experiment actually uses 2x larger patch than the displayed patch size(above).'
)

trainset = MVILiverDataset3D('/newdata/why/MVI_Liver_Formatted', mode='train')
# valset=LiverDataset3D('/newdata/why/MVI_Liver_Formatted',mode='val')
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

# lossfunc_sr=pt.nn.MSELoss()
lossfunc_seg = pt.nn.BCELoss()
# lossfunc_fa=FALoss3D()
optimizer = pt.optim.Adam(model.parameters(), lr=lr)
scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


def ValModel():
    model.eval()
    dice_sum = 0
    hd_sum = 0
    jc_sum = 0
    weight_map = np.zeros(
        (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))
    for a in range(0, img_size[0] - crop_size[0] + 1,
                   crop_size[0] // 2):  # overlap0.5
        for b in range(0, img_size[1] - crop_size[1] + 1, crop_size[1] // 2):
            for c in range(0, img_size[2] - crop_size[2] + 1,
                           crop_size[2] // 2):
                weight_map[:, :, (2 * a):(2 * (a + crop_size[0])),
                           (2 * b):(2 * (b + crop_size[1])),
                           (2 * c):(2 * (c + crop_size[2]))] += 1

    weight_map[weight_map == 0] = 1
    weight_map = 1. / weight_map
    for i, data in enumerate(val_dataset):
        output_list = np.zeros(
            (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))
        label_list = np.zeros(
            (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))

        (_, labels, inputs) = data  # use label_sr as input
        labels3D = pt.autograd.Variable(labels).type(
            pt.FloatTensor).cuda().unsqueeze(1)

        for a in range(0, img_size[0] - crop_size[0] + 1,
                       crop_size[0] // 2):  # overlap0.5
            for b in range(0, img_size[1] - crop_size[1] + 1,
                           crop_size[1] // 2):
                for c in range(0, img_size[2] - crop_size[2] + 1,
                               crop_size[2] // 2):
                    inputs3D = pt.autograd.Variable(
                        inputs[:, (2 * a):(2 * (a + crop_size[0])),
                               (2 * b):(2 * (b + crop_size[1])),
                               (2 * c):(2 * (c + crop_size[2]))]).type(
                                   pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D = model(inputs3D)
                    outputs3D = np.array(outputs3D.cpu().data.numpy())
                    # outputs3D=ndimage.interpolation.zoom(outputs3D,[1,1,2,2,2],order=3)
                    # outputs3D[outputs3D<0.5]=0
                    # outputs3D[outputs3D>=0.5]=1
                    output_list[:, :, (2 * a):(2 * (a + crop_size[0])),
                                (2 * b):(2 * (b + crop_size[1])),
                                (2 * c):(2 * (c + crop_size[2]))] += outputs3D

        label_list = np.array(labels3D.cpu().data.numpy())

        output_list = np.array(output_list) * weight_map

        # label_list=np.array(label_list)

        output_list[output_list < 0.5] = 0
        output_list[output_list >= 0.5] = 1

        final_img = np.zeros(shape=(2 * img_size[1], 2 * 2 * img_size[2]))
        final_img[:, :2 * img_size[2]] = output_list[0, 0, 64, :, :] * 255
        final_img[:, 2 * img_size[2]:] = label_list[0, 0, 64, :, :] * 255
        cv2.imwrite('TestPhase_Res_patchfree_Liver.png', final_img)

        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        print("dice:", dice)

        # hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        # jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        # hd_sum+=hausdorff
        # jc_sum+=jaccard

    print("Finished. Total dice: ", dice_sum / len(val_dataset), '\n')
    print("Finished. Avg Jaccard: ", jc_sum / len(val_dataset))
    print("Finished. Avg hausdorff: ", hd_sum / len(val_dataset))
    return dice_sum / len(val_dataset)


def TestModel():
    model.eval()
    dice_sum = 0
    hd_sum = 0
    jc_sum = 0
    total_time=0
    weight_map = np.zeros(
        (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))
    for a in range(0, img_size[0] - crop_size[0] + 1,
                   crop_size[0] // 2):  # overlap0.5
        for b in range(0, img_size[1] - crop_size[1] + 1, crop_size[1] // 2):
            for c in range(0, img_size[2] - crop_size[2] + 1,
                           crop_size[2] // 2):
                weight_map[:, :, (2 * a):(2 * (a + crop_size[0])),
                           (2 * b):(2 * (b + crop_size[1])),
                           (2 * c):(2 * (c + crop_size[2]))] += 1

    weight_map[weight_map == 0] = 1
    weight_map = 1. / weight_map
    for i, data in enumerate(test_dataset):
        output_list = np.zeros(
            (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))
        label_list = np.zeros(
            (1, 1, 2 * img_size[0], 2 * img_size[1], 2 * img_size[2]))

        (_, labels, inputs) = data  # use label_sr as input
        labels3D = pt.autograd.Variable(labels).type(
            pt.FloatTensor).cuda().unsqueeze(1)

        for a in range(0, img_size[0] - crop_size[0] + 1,
                       crop_size[0] // 2):  # overlap0.5
            for b in range(0, img_size[1] - crop_size[1] + 1,
                           crop_size[1] // 2):
                for c in range(0, img_size[2] - crop_size[2] + 1,
                               crop_size[2] // 2):
                    inputs3D = pt.autograd.Variable(
                        inputs[:, (2 * a):(2 * (a + crop_size[0])),
                               (2 * b):(2 * (b + crop_size[1])),
                               (2 * c):(2 * (c + crop_size[2]))]).type(
                                   pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D = model(inputs3D)
                    outputs3D = np.array(outputs3D.cpu().data.numpy())
                    # outputs3D=ndimage.interpolation.zoom(outputs3D,[1,1,2,2,2],order=3)
                    # outputs3D[outputs3D<0.5]=0
                    # outputs3D[outputs3D>=0.5]=1
                    output_list[:, :, (2 * a):(2 * (a + crop_size[0])),
                                (2 * b):(2 * (b + crop_size[1])),
                                (2 * c):(2 * (c + crop_size[2]))] += outputs3D

        label_list = np.array(labels3D.cpu().data.numpy())

        output_list = np.array(output_list) * weight_map

        # label_list=np.array(label_list)

        output_list[output_list < 0.5] = 0
        output_list[output_list >= 0.5] = 1

        final_img = np.zeros(shape=(2 * img_size[1], 2 * 2 * img_size[2]))
        final_img[:, :2 * img_size[2]] = output_list[0, 0, 64, :, :] * 255
        final_img[:, 2 * img_size[2]:] = label_list[0, 0, 64, :, :] * 255
        cv2.imwrite('TestPhase_Res_patchfree_Liver.png', final_img)

        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        # print("dice:",dice)

        try:
            hausdorff = hd95(
                output_list.squeeze(0).squeeze(0),
                label_list.squeeze(0).squeeze(0))
        except:
            hausdorff = 0
        jaccard = jc(
            output_list.squeeze(0).squeeze(0),
            label_list.squeeze(0).squeeze(0))

        print("dice:", dice, ";hd95:", hausdorff, ";jaccard:", jaccard)

        hd_sum += hausdorff
        jc_sum += jaccard

    print("Finished. Total dice: ", dice_sum / len(test_dataset), '\n')
    print("Finished. Avg Jaccard: ", jc_sum / len(test_dataset))
    print("Finished. Avg hausdorff: ", hd_sum / len(test_dataset))
    return dice_sum / len(test_dataset)


# TestModel()
# raise Exception("end of test")

best_dice = 0
for x in range(epoch):
    model.train()
    print('==>Epoch', x, ': lr=', optimizer.param_groups[0]['lr'], '==>\n')

    for i, data in enumerate(train_dataset):
        (_, labels_seg, inputs) = data
        optimizer.zero_grad()
        inputs = pt.autograd.Variable(inputs).type(
            pt.FloatTensor).cuda().unsqueeze(1)
        labels_seg = pt.autograd.Variable(labels_seg).type(
            pt.FloatTensor).cuda().unsqueeze(1)
        outputs_seg = model(inputs)
        loss_seg = lossfunc_seg(
            outputs_seg, labels_seg)  #+lossfunc_dice(outputs_seg,labels_seg)

        loss_seg.backward()
        optimizer.step()

        if i % 10 == 0:
            final_img = np.zeros(shape=(size, size * 3))
            print('[epoch {:3d},iter {:5d}]'.format(x, i), 'loss:',
                  loss_seg.item())
            final_img[:, 0:size] = outputs_seg.cpu().data.numpy()[
                0, 0, crop_size[0], :, :] * 255
            # final_img[:,128:256]=outputs_sr.cpu().data.numpy()[0,0,31,:,:]*255
            final_img[:, size:(
                2 * size
            )] = labels_seg.cpu().data.numpy()[0, 0, crop_size[0], :, :] * 255
            # final_img[:,384:512]=labels_sr.cpu().data.numpy()[0,0,31,:,:]*255
            final_img[:, (
                2 * size
            ):] = inputs.cpu().data.numpy()[0, 0, crop_size[0], :, :] * 255
            cv2.imwrite('resunet_3d_patchfree_combine.png', final_img)

    scheduler.step()

    print('==>End of epoch', x, '==>\n')

    print('===VAL===>')
    dice = TestModel()
    if dice > best_dice:
        best_dice = dice
        print(
            'New best dice! Model saved to', model_path +
            '/PointUnet/SaliencyAttentionNet_3D_Liver_patch-free_bs' +
            str(batch_size) + '_best.pt')
        pt.save(
            model.state_dict(), model_path +
            '/PointUnet/SaliencyAttentionNet_3D_Liver_patch-free_bs' +
            str(batch_size) + '_best.pt')
    # print('===TEST===>')
    # TestModel()

print('\nBest Dice:', best_dice)
