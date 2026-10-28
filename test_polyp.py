import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import random

from torch.backends import cudnn

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataloader import test_dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(Y_test, yp):
    jacard = 0
    dice = 0
    tanimoto = 0
    smooth = 1e-15
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        only_neg = y2 * (1 - yp_2)
        only_pos = (1 - y2) * yp_2

        if (np.sum(y2) == 0) and (np.sum(yp_2) == 0):
            tanimoto += 1.0
            jacard += 1.0
            dice += 1.0
        elif (np.sum(intersection) == 0):
            tanimoto += 0.0
            jacard += 0.0
            dice += 0.0
        else:
            tanimoto += ((np.sum(intersection) + smooth) / (
                    np.sum(intersection) + np.sum(only_neg) + np.sum(only_pos) + smooth))
            jacard += ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
            dice += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

    jacard /= len(Y_test)
    dice /= len(Y_test)
    tanimoto /= len(Y_test)

    return jacard, dice, tanimoto


def confusion_matrix_scorer(Y, Y_pred):
    Y = Y.astype(np.int8)
    Y_pred = Y_pred.astype(np.int8)
    P = len(np.where(Y == 1)[0])
    N = len(np.where(Y == 0)[0])

    FP = len(np.where(Y - Y_pred == -1)[0])
    FN = len(np.where(Y - Y_pred == 1)[0])
    TP = len(np.where(Y + Y_pred == 2)[0])
    TN = len(np.where(Y + Y_pred == 0)[0])

    return P, N, TN, FP, FN, TP


def get_metrics(Y, pred):
    Y = np.reshape(Y, pred.shape)
    smooth = 1e-15
    P = 0
    N = 0
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    sensitivity = 0
    specificity = 0
    accuracy = 0
    precision = 0
    F1 = 0
    MCC = 0
    for i in range(len(Y)):
        _p, _n, _tn, _fp, _fn, _tp = confusion_matrix_scorer(Y[i], pred[i])
        P += _p
        N += _n
        TN += _tn
        FP += _fp
        FN += _fn
        TP += _tp
        if (np.sum(Y[i]) == 0) and (np.sum(pred[i]) == 0):
            sensitivity += 1
            specificity += 1
            precision += 1
            F1 += 1
            MCC += 1
        else:
            if (_tp == 0):
                sensitivity += 0
                precision += 0
                F1 += 0.0
            else:
                sensitivity += (_tp / (_tp + _fn))
                precision += (_tp / (_tp + _fp))
                F1 += (2 * ((_tp / (_tp + _fp)) * (_tp / (_tp + _fn))) / ((_tp / (_tp + _fp)) + (_tp / (_tp + _fn))))
            if (_tn == 0):
                specificity += 0
            else:
                specificity += (_tn / (_tn + _fp))
            MCC += (_tp * _tn - _fp * _fn + smooth) / (
                    np.power((_tp + _fp) * (_tp + _fn) * (_tn + _fp) * (_tn + _fn), 0.5) + smooth)
        accuracy += ((_tp + _tn) / (_tp + _fn + _fp + _tn))

    return P, N, TN, FP, FN, TP, sensitivity / len(Y), specificity / len(Y), accuracy / len(Y), precision / len(
        Y), F1 / len(Y), MCC / len(Y)


def get_metrics_and_print(Y, yp, method="", testset='CVC-300', threshold=0.5, show=False, write=False,
                          save_path='.'):
    rs = []
    P, N, TN, FP, FN, TP, sensitivity, specificity, accuracy, precision, f1, mcc_cal = get_metrics(Y, yp)
    jacard, dice, tanimoto = calculate_metrics(Y, yp)

    cmat = [[TN, FP], [FN, TP]]
    cmat_score = [[TN / N, FP / N], [FN / P, TP / P]]
    print(cmat)
    if show:
        plt.figure(figsize=(6, 6))
        sns.heatmap(cmat_score, cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
        plt.xlabel("Predictions")
        plt.ylabel("True values")
        plt.show()

    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", sensitivity)
    print("F1 Score: ", f1)
    print("MCC: ", mcc_cal)
    print('Dice: ', dice)
    print('Jacard: ', jacard)
    print('Tanimoto: ', tanimoto)

    if write:
        results = pd.DataFrame([[method, TN, FP, FN, TP, jacard,
                                 dice, sensitivity, specificity,
                                 accuracy, precision, f1, mcc_cal]],
                               columns=['Method', 'TN', 'FP', 'FN', 'TP', 'mIoU/Jacard', 'DICE',
                                        'Sensitivity/Recall', 'Specificity', 'Accuracy', 'Precision',
                                        'F-score', 'MCC'])

        log_folder = './test_log/test_log_' + args.exp
        snapshot_subfolder = '_'.join(
            snapshot_path.split('/')[-1].split('_')[1:])
        log_folder = os.path.join(log_folder, snapshot_subfolder, epoch)
        os.makedirs(log_folder, exist_ok=True)
        result_path = os.path.join(log_folder, 'results_' + testset + '.csv')

        if not os.path.exists(result_path):
            results.to_csv(result_path, mode='w', index=False, header=True)
        else:
            results.to_csv(result_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Polyp',
                        help='experiment_name')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int, default=30000,
                        help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=0,
                        help='whether use deterministic training')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='select optimizer for training, SGD or AdamW')
    parser.add_argument("--alpha", type=float, default=0.5,
                        help='alpha for the mixup training')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=352,
                        help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='vit_patches_size, default is 16')
    parser.add_argument('--skip_aggregation', type=str, default='additive',
                        help='Type of skip-aggregation: additive or concatenation')


    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.is_pretrain = True
    model_name = 'TU_S2Att_PPA_clip'
    args.exp = model_name + '_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_deterministic' if args.deterministic != 0 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.optimizer) if args.optimizer != 'SGD' else snapshot_path
    snapshot_path = snapshot_path + '_a' + str(args.alpha) if args.alpha != 0.5 else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    print(snapshot_path)
    pth_files = [f for f in os.listdir(snapshot_path) if f.endswith('.pth')]
    pth_files.sort()

    for pth_file in pth_files:
        snapshot = os.path.join(snapshot_path, pth_file)
        model.load_state_dict(torch.load(snapshot))
        model.eval()

        epoch = pth_file.split('_')[-1].split('.')[0]

        print(f"Testing model: {pth_file} (Epoch {epoch})")

        for _data_name in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300']:
            ##### put data_path here #####
            data_path = '../data/polyp/TestDataset/{}'.format(_data_name)

            ##### save_path #####
            snapshot_subfolder = os.path.split(snapshot_path)[-1]
            model_name = os.path.split(os.path.dirname(snapshot_path))[-1]
            save_path = os.path.join('./result_map', model_name, snapshot_subfolder, epoch, _data_name)
            save_path = os.path.normpath(save_path)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print('Evaluating ' + data_path)

            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            num1 = len(os.listdir(gt_root))
            test_loader = test_dataset(image_root, gt_root, args.img_size)
            DSC = 0.0
            JACARD = 0.0
            preds = []
            gts = []
            for i in range(num1):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()

                res = model(image)  # forward

                # eval Dice
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(os.path.join(save_path, name), res * 255)

                # input = np.where(res >= 0.5, 1, 0)
                input = np.where(res >= 0.5, 1, 0)
                target = np.where(np.array(gt) >= 0.5, 1, 0)

                preds.append(input)
                gts.append(gt)

                N = gt.shape
                smooth = 1
                input_flat = np.reshape(input, (-1))
                target_flat = np.reshape(target, (-1))
                intersection = (input_flat * target_flat)
                union = input_flat + target_flat - intersection

                jacard = ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
                jacard = '{:.4f}'.format(jacard)
                jacard = float(jacard)
                JACARD += jacard

                dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
                dice = '{:.4f}'.format(dice)
                dice = float(dice)
                DSC += dice

            print('*****************************************************')
            print('Dice Score: ' + str(DSC / num1))
            print('Jacard Score: ' + str(JACARD / num1))
            get_metrics_and_print(np.asarray(gts), np.asarray(preds), method=model_name, testset=_data_name, write=True,
                                  save_path=snapshot_path)
            print(_data_name, 'Finish!')
            print('*****************************************************')

        print(f"Testing for {pth_file} (Epoch {epoch}) completed!")