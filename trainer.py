import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume, structure_loss, AvgMeter, clip_gradient
from torchvision import transforms

def trainer_acdc(args, model, snapshot_path):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator([args.img_size, args.img_size])]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                            num_workers=1)
    model.train()
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))
    logging.info("{} test iterations per epoch".format(len(testloader)))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = (1-args.alpha) * loss_ce + args.alpha * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if iter_num % 20 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f, mem: %.0fMB' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_, memory_used))

            if iter_num > 0 and iter_num % 500 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    save_best = os.path.join(snapshot_path, 'epoch_9999.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                    iter_num, performance, mean_hd95))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        save_interval = 10
        if epoch_num > 75 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = (1 - args.alpha) * loss_ce + args.alpha * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if iter_num % 20 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                current_lr = optimizer.param_groups[0]['lr']
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f, mem: %.0fMB' % (
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(), current_lr, memory_used))

        save_interval = 10
        if epoch_num > 70 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        elif epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"


def trainer_polyp(args, model, snapshot_path):
    from datasets.dataloader import get_loader

    dict_plot = {'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'CVC-300': [], 'test': []}
    name = ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300', 'test']
    test_path = args.test_path
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes

    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)
    train_loader = get_loader(image_root, gt_root, batchsize=args.batch_size, trainsize=args.img_size, shuffle=True,
                              augmentation=args.augmentation)
    print("The length of train set is: {}".format(len(train_loader)))
    total_step = len(train_loader)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    best = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        loss_record = AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- forward ----
            P = model(images)

            # ---- compute loss ----
            loss = structure_loss(P, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # ---- recording loss ----
            loss_record.update(loss.data, args.batch_size)

            if i % 20 == 0 or i == total_step:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      ' loss: {:0.4f}], lr: {:0.6f}'.
                      format(epoch_num, args.max_epochs, i, total_step,
                             loss_record.show(), lr_))

        # save model
        save_path = snapshot_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (epoch_num + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path + '/epoch_' + str(epoch_num) + '.pth')
            print('save model to {}'.format(save_path))

        if (epoch_num + 1) % 1 == 0:
            total_dice = 0
            total_images = 0
            for dataset in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300']:
                dataset_dice, n_images = test(model, test_path, dataset, args)
                total_dice += (n_images * dataset_dice)
                total_images += n_images
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch_num, dataset, dataset_dice))
                print(dataset, ': ', dataset_dice)
                dict_plot[dataset].append(dataset_dice)
            meandice = total_dice / total_images
            dict_plot['test'].append(meandice)
            print('Validation dice score: {}'.format(meandice))
            logging.info('Validation dice score: {}'.format(meandice))
            if meandice > best:
                print('##################### Dice score improved from {} to {}'.format(best, meandice))
                logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
                best = meandice
                torch.save(model.state_dict(), save_path + '/' + 'epoch_9999.pth')

    return "Training Finished!"


def test(model, path, dataset, args):
    from datasets.dataloader import test_dataset

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, args.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)

        res = F.upsample(res, size=gt.shape, mode='bilinear',
                         align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # eval Dice
        input = res >= 0.5
        target = np.array(gt >= 0.5)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1
