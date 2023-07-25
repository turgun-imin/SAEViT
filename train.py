import time

import os
import argparse

from ptflops import get_model_complexity_info

from torch.cuda import amp
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.SAEViT import SAEViT
from utils.DiceLoss import DiceLoss

from dataset.custom_dataset import CustomDataset
import dataset.transforms as T

from utils.metrics import evaluator


def train_one_epoch(
        train_loader,
        model,
        criterion1,
        criterion2,
        optimizer,
        lr_scheduler,
        device,
        epoch,
        num_epoch,
        draw_freq
):

    model.train()
    losses_sum = 0.0
    batches = 0.0
    num_correct = 0.0
    num_pixels = 0.0
    dice = 0.0
    loop = tqdm(train_loader, total=len(train_loader), leave=True, ncols=160, colour="WHITE", position=0)
    for image, label in loop:
        image = image.to(device=device)
        label = label.to(device=device)
        optimizer.zero_grad()
        out = model(image)

        celoss = criterion1(out, label.long().squeeze(1))  # CrossEntropyLoss
        dcloss = criterion2(out, label.long(), weight=[0.0, 1.0], softmax=True)  # DiceLoss
        loss = 0.4 * celoss + 0.6 * dcloss

        scaler = amp.GradScaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        with torch.no_grad():
            losses_sum += loss.item()
            batches += 1
            pred = out.cpu().numpy()
            target = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            num_correct += np.sum(pred.flatten() == target.flatten())
            num_pixels += target.size
            dice += ((2 * np.sum(pred * target)) + 1e-6) / (np.sum(pred + target) + 1e-6)
            loop.set_description(f'train epoch[{epoch}/{num_epoch}]')
            loop.set_postfix({'loss': '{0:1.5f}'.format(losses_sum / batches),
                              'oa acc': '{0:1.5f}'.format(num_correct / num_pixels),
                              'dice score': '{0:1.5f}'.format(dice / batches),
                              'lr': '{0:1.8f}'.format(optimizer.param_groups[0]['lr'])})

            # if batches % draw_freq == 0.0:
            #     wind.line([losses_sum / batches], [epoch*len(train_loader)+batches], win='step(100)/train_loss',
            #               opts=dict(title='step(100)/train_loss'), update='append')
            #     wind.line([num_correct / num_pixels], [epoch*len(train_loader)+batches], win='step(100)/train_oa_acc',
            #               opts=dict(title='step(100)/train_oa_acc'), update='append')
            #     wind.line([dice / batches], [epoch*len(train_loader)+batches], win='step(100)/train_dice_score',
            #               opts=dict(title='step(100)/train_dice_score'), update='append')
            #     wind.line([optimizer.param_groups[0]['lr']], [epoch * len(train_loader) + batches], win='step(100)/lr',
            #               opts=dict(title='step(100)/lr'), update='append')
            #     test_out = model(test_image.unsqueeze(0))
            #     test_pred = torch.argmax(test_out, dim=1)
            #     wind.images(test_pred.cpu().numpy().squeeze(0) * 255.0, nrow=1, win='pred', opts=dict(title='testing'))

    avg_loss = losses_sum / batches
    oa_acc = num_correct / num_pixels

    return avg_loss, oa_acc


def val_one_epoch(
        val_loader,
        model,
        criterion1,
        criterion2,
        device,
        epoch,
        num_epoch,
        num_classes,
        draw_freq
):

    model.eval()
    losses_sum = 0.0
    batches = 0.0
    num_correct = 0.0
    num_pixels = 0.0
    dice = 0.0
    metrics = evaluator(num_classes)
    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader), leave=True, ncols=160, colour="WHITE", position=0)
        for image, label in loop:
            image = image.to(device=device)
            label = label.to(device=device)
            out = model(image)

            celoss = criterion1(out, label.long().squeeze(1))  # CrossEntropyLoss
            dcloss = criterion2(out, label.long(), weight=[0.0, 1.0], softmax=True)  # DiceLoss
            loss = 0.4 * celoss + 0.6 * dcloss

            losses_sum += loss.item()
            batches += 1
            pred = out.cpu().numpy()
            target = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            num_correct += np.sum(pred.flatten() == target.flatten())
            num_pixels += target.size
            dice += ((2 * np.sum(pred * target)) + 1e-6) / (np.sum(pred + target) + 1e-6)
            metrics.add_batch(target.flatten(), pred.flatten())
            loop.set_description(f'val epoch[{epoch}/{num_epoch}]')
            loop.set_postfix({'loss': '{0:1.5f}'.format(losses_sum / batches),
                              'oa acc': '{0:1.5f}'.format(num_correct / num_pixels),
                              'dice score': '{0:1.5f}'.format(dice / batches)})
            # if batches % draw_freq == 0.0:
            #     wind.line([losses_sum / batches], [epoch*len(train_loader)+batches], win='step(100)/val_loss',
            #     opts=dict(title='step(100)/val_loss'), update='append')
            #     wind.line([num_correct / num_pixels], [epoch*len(train_loader)+batches], win='step(100)/val_oa_acc',
            #     opts=dict(title='step(100)/val_oa_acc'), update='append')
            #     wind.line([dice / batches], [epoch*len(train_loader)+batches], win='step(100)/val_dice_score',
            #     opts=dict(title='step(100)/val_dice_score'), update='append')

    avg_loss = losses_sum / batches
    oa_acc = num_correct / num_pixels

    return avg_loss, oa_acc, metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_root_path', default=r'')  # Write the root path of the dataset here
    parser.add_argument("--test_image", default='./test_image/test_image_1.png')
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--weight_decay', default=0.0025)
    parser.add_argument('--eps', default=1e-4)
    parser.add_argument('--model_save_path', default='./checkpoint/')
    parser.add_argument('--resume', default=None)
    # parser.add_argument('--resume', default='') # Write pre training weights here
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--logs_save_path', default='/root/tf-logs/logs')
    parser.add_argument('--num_epoch', default=120, type=int)
    parser.add_argument('--draw_freq', default=100, type=int)
    args = parser.parse_args()
    print('\nProgram starting... \n')
    for item in args.__dict__.items():
        key = item[0]
        value = item[1]
        print(f'{key} : {value}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'\nAvailable device : {device}\n')
    torch.cuda.empty_cache()
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    train_dataset = CustomDataset(dataset_root_path=args.dataset_root_path, train=True, transforms=T.ToTensor())
    val_dataset = CustomDataset(dataset_root_path=args.dataset_root_path, train=False, transforms=T.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory)
    print(f'The total number of training dataset image is : {len(train_dataset)}\n'
          f'The total number of validation dataset image is : {len(val_dataset)}\n'
          f'The total number of batches of training dataset image is : {len(train_loader)}\n'
          f'The total number of batches of validation dataset image is : {len(val_loader)}\n')

    model = SAEViT().to(device)

    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = DiceLoss(args.num_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    Total_params = 0.0
    Trainable_params = 0.0
    NonTrainable_params = 0.0
    for param in model.parameters():
        Value_sum = np.prod(param.size())
        Total_params += Value_sum
        if param.requires_grad:
            Trainable_params += Value_sum
        else:
            NonTrainable_params += Value_sum
    macs, params = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False)
    print(f'Computational complexity: {macs:<8}\n'
          # f'Number of parameters:: {params:<8}\n'
          f'Total params: {(Total_params / 1e6):>0.2f} M\n'
          f'Trainable params: {(Trainable_params / 1e6):>0.2f} M\n'
          f'Non-trainable params: {(NonTrainable_params / 1e6):>0.2f} M\n')

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
        print(f"checkpoints save directory has built.\n")

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'\n")
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        print(f"=> loaded checkpoint '{args.resume}'\n")

        # if DataParallel:
        #     self.model.module.load_state_dict(checkpoint['state_dict'])
        # else:
        #     self.model.load_state_dict(checkpoint['state_dict'])

    time_start = time.time()
    ts = time.localtime()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", ts)
    print(f"Program start time : {ts} \n"
          f"-----------------------------------------")
    best_oa = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_iou = 0.0
    no_optim = 0
    if os.path.isfile(args.logs_save_path):
        os.remove(args.logs_save_path)
    writer = SummaryWriter(f'{args.logs_save_path}/')
    # wind = visdom.Visdom()
    # test_image = Image.open(args.test_image).convert('RGB')
    # image_tensor = tf.to_tensor(test_image).to(device=device)
    # wind.image(image_tensor)
    time.sleep(0.01)
    for epoch in range(args.start_epoch, args.num_epoch):
        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            criterion1,
            criterion2,
            optimizer,
            lr_scheduler,
            device,
            epoch,
            args.num_epoch,
            # image_tensor,
            args.draw_freq
        )
        val_loss, val_acc, val_metrics = val_one_epoch(
            val_loader,
            model,
            criterion1,
            criterion2,
            device,
            epoch,
            args.num_epoch,
            args.num_classes,
            args.draw_freq
        )
        time.sleep(0.01)
        val_oa = val_metrics.overall_accuracy()
        val_class_acc = val_metrics.class_pixel_accuracy()
        val_m_acc = val_metrics.mean_pixel_accuracy()
        val_precision = val_metrics.precision()
        val_recall = val_metrics.recall()
        val_f1 = val_metrics.f1()
        val_dice = val_metrics.dice_score()
        val_iou = val_metrics.iou()
        val_miou = val_metrics.mean_iou()
        val_fwiou = val_metrics.fw_iou()
        if val_oa > best_oa:
            best_oa = val_oa
        if val_precision[1] > best_precision:
            best_precision = val_precision[1]
        if val_recall[1] > best_recall:
            best_recall = val_recall[1]
        if val_f1[1] > best_f1:
            best_f1 = val_f1[1]
        if val_iou[1] > best_iou:
            best_iou = val_iou[1]
        print(f'val epoch[{epoch}/{args.num_epoch}]: '
              f'oa={val_oa:>0.5f}/{best_oa:>0.5f}, '
              f'precision={val_precision[1]:>0.5f}/{best_precision:>0.5f}, '
              f'recall={val_recall[1]:>0.5f}/{best_recall:>0.5f}, '
              f'f1={val_f1[1]:>0.5f}/{best_f1:>0.5f}, '
              f'iou={val_iou[1]:>0.5f}/{best_iou:>0.5f}')
        time.sleep(0.01)

        # wind.line([train_loss], [epoch], win='epoch/train_loss', opts=dict(title='epoch/train_loss'), update='append')
        # wind.line([train_acc], [epoch], win='epoch/train_oa_acc', opts=dict(title='epoch/train_oa_acc'), update='append')
        # wind.line([val_loss], [epoch], win='epoch/val_loss', opts=dict(title='epoch/val_loss'), update='append')
        # wind.line([val_acc], [epoch], win='epoch/val_oa_acc', opts=dict(title='epoch/val_oa_acc'), update='append')
        # wind.line([val_precision[1]], [epoch], win='precision', opts=dict(title='precision'), update='append')
        # wind.line([val_recall[1]], [epoch], win='recall', opts=dict(title='recall'), update='append')
        # wind.line([val_f1[1]], [epoch], win='f1', opts=dict(title='f1'), update='append')
        # wind.line([val_iou[1]], [epoch], win='iou', opts=dict(title='iou'), update='append')

        writer.add_scalars("epoch/loss", {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars("epoch/acc", {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('epoch/oa', val_oa, epoch)
        writer.add_scalars("epoch/class_acc",
                           {'non building acc': val_class_acc[0], 'building acc': val_class_acc[1], 'm_acc': val_m_acc}, epoch)
        writer.add_scalars("epoch/iou",
                           {'non building iou': val_iou[0], 'building iou': val_iou[1],
                            'm_iou': val_miou, 'fw_iou': val_fwiou}, epoch)
        writer.add_scalars("epoch/precision", {'non building precision': val_precision[0],
                                               'building precision': val_precision[1]}, epoch)
        writer.add_scalars("epoch/recall", {'non building recall': val_recall[0], 'building recall': val_recall[1]}, epoch)
        writer.add_scalars("epoch/f1", {'non building f1': val_f1[0], 'building f1': val_f1[1]}, epoch)
        writer.add_scalars("epoch/dice", {'non building dice': val_dice[0], 'building dice': val_dice[1]}, epoch)

        model_name = os.path.join(args.model_save_path, 'checkpoint_' + 'epoch_' + str(epoch) + '.pth')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, model_name)

    writer.close()

    time_end = time.time()
    te = time.localtime()
    te = time.strftime("%Y-%m-%d %H:%M:%S", te)
    sp_t = time_end - time_start
    sp_t = time.strftime("%H:%M:%S", time.gmtime(sp_t))
    print(f"---------------------------------------\n"
          f"Program end time: {te} \n"
          f"Program taken time: {sp_t}")
