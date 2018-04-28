from os import listdir
import argparse
import numpy as np
import visdom
import pickle
import time


def main(args):
    vis = visdom.Visdom(port=6008)

    losses = pickle.load( open( "results/saved_loss.p", "rb" ) )
    x=np.squeeze(np.asarray(losses["X"]))
    l=np.squeeze(np.asarray(losses["Y"]))
    ltest=np.squeeze(np.asarray(losses["Y_test"]))
    vis.line(np.vstack((l,ltest)).T, x, env='loss_acc', opts=dict(title="Loss"))

    accuracy = pickle.load( open( "results/saved_accuracy.p", "rb" ) )
    x=np.squeeze(np.asarray(accuracy["X"]))
    P = accuracy["P"]
    M = accuracy["M"]
    I = accuracy["I"]
    Ptest = accuracy["P_test"]
    Mtest = accuracy["M_test"]
    Itest = accuracy["I_test"]
    vis.line(np.vstack((P.T, Ptest.T)).T, x, env='loss_acc', opts=dict(title="Pixel Accuracy"))
    vis.line(np.vstack((M.T, Mtest.T)).T, x, env='mean_acc', opts=dict(title="Mean Accuracy"))
    vis.line(np.vstack((I.T, Itest.T)).T, x, env='IoU', opts=dict(title="Mean IoU"))

    if args.images:
        onlyfiles = [f for f in listdir('./results/saved_val_images')]
        onlyfiles.sort()
        for f in onlyfiles:
            if f.endswith('.p'):
                image = pickle.load( open( "results/saved_val_images/"+f, "rb" ) )
                if len(image.shape) == 4:
                    vis.image(image[0], env='images', opts=dict(title=f))
                else:
                    if image.shape[0] != 3:
                        image = image.transpose((2,0,1))
                    vis.image(image, env='images', opts=dict(title=f))
                time.sleep(0.25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--images', help='Also displays images from validation set', action='store_true')

    args = parser.parse_args()
    main(args)