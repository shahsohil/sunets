import os
import argparse
import numpy as np
import pickle
import random
import math

import torch
from torch.autograd import Variable
from torch.utils import data
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg import get_data_path

def train(args):
    global n_classes

    # Set the seed for reproducing the results
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        cudnn.benchmark = True

    # Set up results folder
    if not os.path.exists('results/saved_val_images'):
        os.makedirs('results/saved_val_images')
    if not os.path.exists('results/saved_train_images'):
        os.makedirs('results/saved_train_images')

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)

    traindata = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    trainloader = data.DataLoader(traindata, batch_size=args.batch_size, num_workers=7, shuffle=True)

    valdata = data_loader(data_path, split="val", is_transform=False, img_size=(args.img_rows, args.img_cols))
    valloader = data.DataLoader(valdata, batch_size=args.batch_size, num_workers=7, shuffle=False)

    n_classes = traindata.n_classes
    n_trainsamples = len(traindata)
    n_iters_per_epoch = np.ceil(n_trainsamples / float(args.batch_size * args.iter_size))

    # Setup Model
    model = torch.nn.DataParallel(get_model(args.arch, n_classes, ignore_index=traindata.ignore_index, output_stride=args.ost))

    if torch.cuda.is_available():
        model.cuda()

    epochs_done=0
    X=[]
    Y=[]
    Y_test=[]
    avg_pixel_acc = 0
    mean_class_acc = 0
    mIoU = 0
    avg_pixel_acc_test = 0
    mean_class_acc_test = 0
    mIoU_test = 0

    if args.model_path:
        model_name=args.model_path.split('.')
        checkpoint_name = model_name[0]+'_optimizer.pkl'
        checkpoint = torch.load(checkpoint_name)
        optm = checkpoint['optimizer']
        model.load_state_dict(checkpoint['state_dict'])
        split_str=model_name[0].split('_')
        epochs_done=int(split_str[-1])
        saved_loss = pickle.load( open( "results/saved_loss.p", "rb" ) )
        saved_accuracy = pickle.load( open( "results/saved_accuracy.p", "rb" ) )
        X=saved_loss["X"][:epochs_done]
        Y=saved_loss["Y"][:epochs_done]
        Y_test=saved_loss["Y_test"][:epochs_done]
        avg_pixel_acc=saved_accuracy["P"][:epochs_done,:]
        mean_class_acc = saved_accuracy["M"][:epochs_done,:]
        mIoU = saved_accuracy["I"][:epochs_done,:]
        avg_pixel_acc_test=saved_accuracy["P_test"][:epochs_done,:]
        mean_class_acc_test = saved_accuracy["M_test"][:epochs_done,:]
        mIoU_test = saved_accuracy["I_test"][:epochs_done,:]

    # Learning rates: For new layers (such as final layer), we set lr to be 10x the learning rate of layers already trained
    bias_10x_params = filter(lambda x: ('bias' in x[0]) and ('final' in x[0]) and ('conv' in x[0]),
                         model.named_parameters())
    bias_10x_params = list(map(lambda x: x[1], bias_10x_params))

    bias_params = filter(lambda x: ('bias' in x[0]) and ('final' not in x[0]),
                         model.named_parameters())
    bias_params = list(map(lambda x: x[1], bias_params))

    nonbias_10x_params = filter(lambda x: (('bias' not in x[0]) or ('bn' in x[0])) and ('final' in x[0]),
                         model.named_parameters())
    nonbias_10x_params = list(map(lambda x: x[1], nonbias_10x_params))

    nonbias_params = filter(lambda x: ('bias' not in x[0]) and ('final' not in x[0]),
                            model.named_parameters())
    nonbias_params = list(map(lambda x: x[1], nonbias_params))

    optimizer = torch.optim.SGD([{'params': bias_params, 'lr': args.l_rate},
                                 {'params': bias_10x_params, 'lr': 20 * args.l_rate},
                                 {'params': nonbias_10x_params, 'lr': 10 * args.l_rate},
                                 {'params': nonbias_params, 'lr': args.l_rate},],
                                lr=args.l_rate, momentum=args.momentum, weight_decay=args.wd,
                                nesterov=(args.optim == 'Nesterov'))
    numgroups = 4

    # Setting up scheduler
    if args.model_path and args.restore:
        # Here we restore all states of optimizer
        optimizer.load_state_dict(optm)
        total_iters = n_iters_per_epoch * args.n_epoch
        lambda1 = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*numgroups, last_epoch=epochs_done*n_iters_per_epoch)
    else:
        # Here we simply restart the training
        if args.T0:
            total_iters = args.T0 * n_iters_per_epoch
        else:
            total_iters = ((args.n_epoch - epochs_done) * n_iters_per_epoch)
        lambda1 = lambda step: 0.5 + 0.5 * math.cos(np.pi * step / total_iters)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1]*numgroups)

    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps, steps_test

    scheduler.step()

    for epoch in range(epochs_done,args.n_epoch):
        # Reset all variables every epoch
        l_avg=0
        totalclasswise_pixel_acc = 0
        totalclasswise_gtpixels = 0
        totalclasswise_predpixels = 0
        l_avg_test=0
        totalclasswise_pixel_acc_test = 0
        totalclasswise_gtpixels_test = 0
        totalclasswise_predpixels_test = 0
        steps=0
        steps_test=0

        trainmodel(model, optimizer, trainloader, epoch, scheduler, traindata)
        valmodel(model, valloader, epoch)

        # save the model every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == args.n_epoch-1:
            torch.save(model, "results/{}_{}_{}.pkl".format(args.arch, args.dataset, epoch+1))
            torch.save({'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},
                       "results/{}_{}_{}_optimizer.pkl".format(args.arch, args.dataset, epoch+1))

        if os.path.isfile("results/saved_loss.p"):
            os.remove("results/saved_loss.p")
        if os.path.isfile("results/saved_accuracy.p"):
            os.remove("results/saved_accuracy.p")

        # saving train and validation loss
        X.append(epoch+1)
        Y.append(l_avg / steps)
        Y_test.append(l_avg_test / steps_test)
        saved_loss={"X":X,"Y":Y,"Y_test":Y_test}
        pickle.dump(saved_loss, open("results/saved_loss.p","wb"))

        # pixel accuracy
        totalclasswise_pixel_acc = totalclasswise_pixel_acc.reshape((-1, n_classes)).astype(np.float32)
        totalclasswise_gtpixels = totalclasswise_gtpixels.reshape((-1, n_classes))
        totalclasswise_predpixels = totalclasswise_predpixels.reshape((-1, n_classes))
        totalclasswise_pixel_acc_test = totalclasswise_pixel_acc_test.reshape((-1, n_classes)).astype(np.float32)
        totalclasswise_gtpixels_test = totalclasswise_gtpixels_test.reshape((-1, n_classes))
        totalclasswise_predpixels_test = totalclasswise_predpixels_test.reshape((-1, n_classes))

        if isinstance(avg_pixel_acc, np.ndarray):
            avg_pixel_acc = np.vstack((avg_pixel_acc, np.sum(totalclasswise_pixel_acc, axis=1) / np.sum(totalclasswise_gtpixels, axis=1)))
            mean_class_acc = np.vstack((mean_class_acc, np.mean(totalclasswise_pixel_acc / totalclasswise_gtpixels, axis=1)))
            mIoU = np.vstack((mIoU, np.mean(totalclasswise_pixel_acc / (totalclasswise_gtpixels + totalclasswise_predpixels - totalclasswise_pixel_acc), axis=1)))

            avg_pixel_acc_test = np.vstack((avg_pixel_acc_test, np.sum(totalclasswise_pixel_acc_test,axis=1) / np.sum(totalclasswise_gtpixels_test, axis=1)))
            mean_class_acc_test = np.vstack((mean_class_acc_test, np.mean(totalclasswise_pixel_acc_test / totalclasswise_gtpixels_test, axis=1)))
            mIoU_test = np.vstack((mIoU_test, np.mean(totalclasswise_pixel_acc_test / (totalclasswise_gtpixels_test + totalclasswise_predpixels_test - totalclasswise_pixel_acc_test), axis=1)))
        else:
            avg_pixel_acc = np.sum(totalclasswise_pixel_acc, axis=1) / np.sum(totalclasswise_gtpixels, axis=1)
            mean_class_acc = np.mean(totalclasswise_pixel_acc / totalclasswise_gtpixels, axis=1)
            mIoU = np.mean(totalclasswise_pixel_acc / (totalclasswise_gtpixels + totalclasswise_predpixels - totalclasswise_pixel_acc), axis=1)

            avg_pixel_acc_test = np.sum(totalclasswise_pixel_acc_test, axis=1) / np.sum(totalclasswise_gtpixels_test, axis=1)
            mean_class_acc_test = np.mean(totalclasswise_pixel_acc_test / totalclasswise_gtpixels_test, axis=1)
            mIoU_test = np.mean(totalclasswise_pixel_acc_test / (totalclasswise_gtpixels_test + totalclasswise_predpixels_test - totalclasswise_pixel_acc_test), axis=1)

        saved_accuracy = {"X": X, "P": avg_pixel_acc, "M": mean_class_acc, "I": mIoU,
                          "P_test": avg_pixel_acc_test, "M_test": mean_class_acc_test, "I_test": mIoU_test}
        pickle.dump(saved_accuracy, open("results/saved_accuracy.p","wb"))


# Incase one want to freeze BN params
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def trainmodel(model, optimizer, trainloader, epoch, scheduler, data):
    global l_avg, totalclasswise_pixel_acc, totalclasswise_gtpixels, totalclasswise_predpixels
    global steps

    model.train()
    if args.freeze:
        model.apply(set_bn_eval)

    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            imagesV = Variable(images.cuda())
            labelsV = Variable(labels.cuda(), requires_grad=False)
        else:
            imagesV = Variable(images)
            labelsV = Variable(labels, requires_grad=False)

        if i % args.iter_size == 0:
            optimizer.zero_grad()

        outputs, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel = \
            model(imagesV, labelsV)

        total_valid_pixel = float(total_valid_pixel.sum(0).data.cpu().numpy())

        totalloss = losses.sum()

        # Because size_average=False
        totalloss = totalloss / float(total_valid_pixel)

        # This is normalize loss when weight updates is done after multiple forward pass
        totalloss = totalloss / float(args.iter_size)

        totalloss.backward()

        if (i+1) % args.iter_size == 0:
            optimizer.step()

        l_avg += (losses.sum().data.cpu().numpy())
        steps += total_valid_pixel
        totalclasswise_pixel_acc += classwise_pixel_acc.sum(0).data.cpu().numpy()
        totalclasswise_gtpixels += classwise_gtpixels.sum(0).data.cpu().numpy()
        totalclasswise_predpixels += classwise_predpixels.sum(0).data.cpu().numpy()

        print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.n_epoch, losses.sum().data[0]))

        if (i+1) % args.iter_size == 0:
            scheduler.step()

        if (i + 1) % args.log_size == 0:
            pickle.dump(images[0].numpy(),
                        open("results/saved_train_images/" + str(epoch) + "_" + str(i) + "_input.p", "wb"))

            pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                        open("results/saved_train_images/" + str(epoch) + "_" + str(i) + "_output.p", "wb"))

            pickle.dump(np.transpose(data.decode_segmap(labels[0].numpy()), [2, 0, 1]),
                        open("results/saved_train_images/" + str(epoch) + "_" + str(i) + "_target.p", "wb"))


def valmodel(model, valloader, epoch):
    global l_avg_test, totalclasswise_pixel_acc_test, totalclasswise_gtpixels_test, totalclasswise_predpixels_test
    global steps_test

    model.eval()

    for i, (imgs_test, lbls_test) in enumerate(valloader):
        if torch.cuda.is_available():
            imgs_testV = Variable(imgs_test.cuda(), volatile=True)
            lbls_testV = Variable(lbls_test.cuda(), volatile=True)
        else:
            imgs_testV = Variable(imgs_test, volatile=True)
            lbls_testV = Variable(lbls_test, volatile=True)

        outputs, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel = \
            model(imgs_testV, lbls_testV)

        total_valid_pixel = float(total_valid_pixel.sum(0).data.cpu().numpy())

        l_avg_test += (losses.sum().data.cpu().numpy())
        steps_test += total_valid_pixel
        totalclasswise_pixel_acc_test += classwise_pixel_acc.sum(0).data.cpu().numpy()
        totalclasswise_gtpixels_test += classwise_gtpixels.sum(0).data.cpu().numpy()
        totalclasswise_predpixels_test += classwise_predpixels.sum(0).data.cpu().numpy()

        if (i + 1) % 50 == 0:
            pickle.dump(imgs_test[0].numpy(),
                        open("results/saved_val_images/" + str(epoch) + "_" + str(i) + "_input.p", "wb"))

            pickle.dump(np.transpose(data.decode_segmap(outputs[0].data.cpu().numpy().argmax(0)), [2, 0, 1]),
                        open("results/saved_val_images/" + str(epoch) + "_" + str(i) + "_output.p", "wb"))

            pickle.dump(np.transpose(data.decode_segmap(lbls_test[0].numpy()), [2, 0, 1]),
                        open("results/saved_val_images/" + str(epoch) + "_" + str(i) + "_target.p", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='sunet7128',
                        help='Architecture to use [\'sunet64, sunet128, sunet7128 etc\']')
    parser.add_argument('--model_path', help='Path to the saved model', type=str)
    parser.add_argument('--dataset', nargs='?', type=str, default='sbd',
                        help='Dataset to use [\'sbd, coco, cityscapes etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=90,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=22,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')
    parser.add_argument('--manualSeed', default=0, type=int,
                        help='manual seed')
    parser.add_argument('--iter_size', type=int, default=1,
                        help='number of batches per weight updates')
    parser.add_argument('--log_size', type=int, default=400,
                        help='iteration period of logging segmented images')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.95,
                        help='Momentum for SGD')
    parser.add_argument('--wd', nargs='?', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optim', nargs='?', type=str, default='SGD',
                        help='Optimizer to use [\'SGD, Nesterov etc\']')
    parser.add_argument('--ost', nargs='?', type=str, default='16',
                        help='Output stride to use [\'32, 16, 8 etc\']')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze BN params')
    parser.add_argument('--restore', action='store_true',
                        help='Restore Optimizer params')
    parser.add_argument('--split', nargs='?', type=str, default='train_aug',
                        help='Sets to use [\'train_aug, train, trainvalrare, trainval_aug, trainval etc\']')

    global args
    args = parser.parse_args()
    train(args)