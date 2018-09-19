import os
import torch
import argparse
import numpy as np
from skimage.transform import resize
from torchvision.transforms import Scale
import torch.nn as nn
from PIL import Image

from torch.autograd import Variable

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg import get_data_path

def test(args):

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    testdata = data_loader(data_path, split=args.split, is_transform=False, img_size=(512, 512))
    n_classes = testdata.n_classes
    eps = 1e-10

    args.coco += 5

    scales = [0.5, 0.75, 1.0, 1.25]
    base_size = min(testdata.img_size)
    crop_size = (args.img_rows, args.img_cols)
    stride = [0, 0]
    stride[0] = int(np.ceil(float(crop_size[0]) * 2/3))
    stride[1] = int(np.ceil(float(crop_size[1]) * 2/3))
    size_transform_img = [Scale(int(base_size*i)) for i in scales]

    # Setup Model
    model = torch.nn.DataParallel(get_model(args.arch, n_classes, ignore_index=testdata.ignore_index, output_stride=args.ost))
    model_name = args.model_path.split('.')
    checkpoint_name = model_name[0] + '_optimizer.pkl'
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    soft = nn.Softmax2d()
    if torch.cuda.is_available():
        model.cuda()
        soft.cuda()

    for f_no, line in enumerate(testdata.files):
        imgr = readfile(args.img_path, line)
        origw, origh = imgr.size

        # Maintain final prediction array for each image
        pred = np.zeros((n_classes, origh, origw), dtype=np.float32)

        # Loop over all scales for single image
        for i in range(len(scales)):
            img = size_transform_img[i](imgr)
            imsw, imsh = img.size

            imwstart, imhstart = 0, 0
            imw, imh = imsw, imsh
            # Zero padding if any size if smaller than crop_size
            if imsw < crop_size[1] or imsh < crop_size[0]:
                padw, padh = max(crop_size[1] - imsw, 0), max(crop_size[0] - imsh, 0)
                imw += padw
                imh += padh
                im = Image.new(img.mode, (imw, imh), tuple(testdata.filler))
                im.paste(img, (int(padw / 2), int(padh / 2)))
                imwstart += int(padw / 2)
                imhstart += int(padh / 2)
                img = im

            # Now tile image - each of crop_size and loop over them
            h_grid = int(np.ceil(float(imh - crop_size[0]) / stride[0])) + 1
            w_grid = int(np.ceil(float(imw - crop_size[1]) / stride[1])) + 1

            # maintain prediction probability for each pixel
            datascale = torch.zeros(n_classes, imh, imw).cuda()
            countscale = torch.zeros(n_classes, imh, imw).cuda()
            for w in range(w_grid):
                for h in range(h_grid):
                    # crop portion from image - crop_size
                    x1, y1 = w * stride[1], h * stride[0]
                    x2, y2 = int(min(x1 + crop_size[1], imw)), int(min(y1 + crop_size[0], imh))
                    x1, y1 = x2 - crop_size[1], y2 - crop_size[0]
                    img_cropped = img.crop((x1, y1, x2, y2))

                    # Input image as well its flipped version
                    img1 = testdata.image_transform(img_cropped)
                    img2 = testdata.image_transform(img_cropped.transpose(Image.FLIP_LEFT_RIGHT))
                    images = torch.stack((img1, img2), dim=0)

                    if torch.cuda.is_available():
                        images = Variable(images.cuda(), volatile=True)
                    else:
                        images = Variable(images, volatile=True)

                    # Output prediction for image and its flip version
                    outputs = model(images)

                    # Sum prediction from image and its flip and then normalize
                    prob = outputs[0] + outputs[1][:, :, getattr(torch.arange(outputs.size(3)-1, -1, -1), 'cuda')().long()]
                    prob = soft(prob.view(-1, *prob.size()))

                    # Place the score in the proper position
                    datascale[:, y1:y2, x1:x2] += prob.data
                    countscale[:, y1:y2, x1:x2] += 1
            # After looping over all tiles of image, normalize the scores and bilinear interpolation to orignal image size
            datascale /= (countscale + eps)
            datascale = datascale[:, imhstart:imhstart+imsh, imwstart:imwstart+imsw]
            datascale = datascale.cpu().numpy()
            datascale = np.transpose(datascale, (1, 2, 0))
            datascale = resize(datascale, (origh, origw), order=1, preserve_range=True, mode='symmetric', clip=False)
            datascale = np.transpose(datascale, (2, 0, 1))

            # Sum up all the scores for all scales
            pred += (datascale / (np.sum(datascale, axis=0) + eps))

        pred = pred / len(scales)
        pred = pred.argmax(0).astype(np.uint32)

        im = Image.fromarray(pred)
        im.save(os.path.join(args.outpath, str(args.coco) + "_" + str(args.split) + "_cls/" + line + ".png"))


def readfile(img_path, img_name):
    img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='sunet7128',
                        help='Architecture to use [\'sunet64, sunet128, sunet7128 etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='sunet7128_sbd.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='sbd',
                        help='Dataset to use [\'sbd, cityscapes etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512,
                        help='Height of the Crop size')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                        help='Width of the Crop size')
    parser.add_argument('--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    parser.add_argument('--coco', nargs='?', type=int, default=0,
                        help='Trained with external data (coco) ?')
    parser.add_argument('--split', type=str, default='val',
                        help='val or test split')
    parser.add_argument('--ost', nargs='?', type=str, default='16',
                        help='Output stride to use [\'32, 16, 8 etc\']')
    args = parser.parse_args()
    test(args)