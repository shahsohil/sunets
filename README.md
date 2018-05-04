# Stacked U-Nets (SUNets) #

## Introduction ##
This is a PyTorch implementation for training classification and semantic segmentation task using Stacked U-Nets models presented in the following paper ([paper](https://arxiv.org/abs/1804.10343)):

Sohil Shah, Pallabi Ghosh, Larry S. Davis and Tom Goldstein. Stacked U-Nets:A No-Frills Approach to Natural Image Segmentation.

If you use this code in your research, please cite our paper.
```
@article{shah2018sunets,
	author    = {Sohil Atul Shah and Pallabi Ghosh and Larry S. Davis and Tom Goldstein},
	title     = {Stacked U-Nets:A No-Frills Approach to Natural Image Segmentation},
	journal   = {arXiv:1804.10343},
	year      = {2018},
}
```

The source code and dataset are published under the MIT license. See [LICENSE](LICENSE) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

## Requirements ##

* Python 2.7
* [Pytorch](http://pytorch.org/) >= v0.2.0
* [visdom](https://github.com/facebookresearch/visdom) >=1.0.1 (for loss and results visualization)
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

## Data ##

* Dataset(s) can be downloaded using the list of URLs provided [here](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets).
* Extract the zip / tar and modify the path appropriately in [config.json](config.json)


## Usage ##

## ImageNet Classification 
**To train the model :**

```
python train_imagenet.py [-h] [--arch ARCH] [-j N] [--epochs N]
                         [--start-epoch N] [-b N] [--lr LR] [--momentum M]
                         [--weight-decay W] [--print-freq N] [--resume PATH]
                         [-e] [--pretrained] [--world-size WORLD_SIZE]
                         [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                         [--id ID] [--tensorboard] [--manualSeed MANUALSEED]
                         DIR

  DIR                   path to dataset
  --arch, -a            model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn | sunet64 | sunet128 | sunet7128 | 
                        (default: sunet7128)
  -j, --workers         number of data loading workers (default: 8)
  --epochs              number of total epochs to run
  --start-epoch         manual epoch number (useful on restarts)
  -b, --batch-size      mini-batch size (default: 256)
  --lr                  initial learning rate
  --momentum            momentum
  --wd                  weight decay (default: 5e-4)
  --print-freq, -p      print frequency (default: 10)
  --resume              path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --id                  identifying number
  --tensorboard         Log progress to TensorBoard
  --manualSeed          manual seed                        
```
For example, one can start training on ImageNet data using
```angular2html
python train_imagenet.py /path/to/imagenet/ -a sunet7128 -b 256 --resume /path/to/checkpoint/ --manualSeed 0 --id $JOBID --tensorboard --lr 0.01 --epochs 100
```

**To view the results:**

Simply run
```angular2html
tensorboard --logdir logs/
```

## Semantic Segmentation 
**To train the model :**

```
python train_seg.py [-h] [--arch [ARCH]] [--model_path MODEL_PATH]
                    [--dataset [DATASET]] [--img_rows [IMG_ROWS]]
                    [--img_cols [IMG_COLS]] [--n_epoch [N_EPOCH]]
                    [--batch_size [BATCH_SIZE]] [--l_rate [L_RATE]]
                    [--manualSeed MANUALSEED] [--iter_size ITER_SIZE]
                    [--log_size LOG_SIZE] [--momentum [MOMENTUM]] [--wd [WD]]
                    [--optim [OPTIM]] [--ost [OST]] [--freeze] [--restore]
                    [--split [SPLIT]]

  --arch                Architecture to use ['sunet64, sunet128, sunet7128 etc']                        
  --model_path          Path to the saved model                        
  --dataset             Dataset to use ['sbd, coco, cityscapes etc']
  --img_rows            Height of the input image                        
  --img_cols            Width of the input image                        
  --n_epoch             # of the epochs
  --batch_size          Batch Size                        
  --l_rate              Learning Rate
  --manualSeed          manual seed                        
  --iter_size           number of batches per weight updates                        
  --log_size            iteration period of logging segmented images
  --momentum            Momentum for SGD                        
  --wd                  Weight decay
  --optim               Optimizer to use ['SGD, Nesterov etc']
  --ost                 Output stride to use ['32, 16, 8 etc']
  --freeze              Freeze BN params
  --restore             Restore Optimizer params
  --split               Sets to use ['train_aug, train, trainvalrare, trainval_aug, trainval etc']                        
```
For example, one can start fine-tuning on pascal VOC2012 data using
```angular2html
python train.py --arch sunet7128 --dataset sbd --batch_size 22 --iter_size 1 --n_epoch 90 --l_rate 0.0002 --momentum 0.95 --wd 1e-4 --optim SGD --img_rows 512 --img_cols 512
```

**To validate the model at multiple scales:**

```
python test_multiscale.py [-h] [--arch [ARCH]] [--model_path [MODEL_PATH]]
                          [--dataset [DATASET]] [--img_rows [IMG_ROWS]]
                          [--img_cols [IMG_COLS]]

  --arch                Architecture to use ['sunet64, sunet128, sunet7128 etc']                        
  --model_path          Path to the saved model               
  --dataset             Dataset to use ['sbd, cityscapes etc']
  --img_rows            Height of the Crop size             
  --img_cols            Width of the Crop size
```
For example, one can validate on pascal VOC2012 validation data using
```angular2html
python test_multiscale.py --arch sunet7128 --dataset sbd --model_path /path/to/checkpoint --img_rows 512 --img_cols 512
```

**To evaluate the model on custom images(s):**

```
python evaluate_pascal.py [-h] [--arch [ARCH]] [--model_path [MODEL_PATH]]
                          [--dataset [DATASET]] [--img_rows [IMG_ROWS]]
                          [--img_cols [IMG_COLS]] [--img_path [IMG_PATH]]
                          [--out_path [OUT_PATH]] [--coco [COCO]]
                          [--split SPLIT]

  --img_path            Path of the input image             
  --out_path            Path of the output segmap. Arranged according to PASCAL server requirements.             
  --coco                Trained with external data (coco) ?
  --split               val or test split
```
For example, one can evaluate on pascal VOC2012 test data using
```angular2html
python test_multiscale.py --arch sunet7128 --dataset sbd --model_path /path/to/checkpoint --img_rows 512 --img_cols 512 --split val --img_path /path/to/images --out_path /path/to/output_folder
```

**To view the results:**

Launch [visdom](https://github.com/facebookresearch/visdom#launch) by running (in a separate terminal window) and run [display.py](display.py).
```
python -m visdom.server
python display.py [--images]
```
The 'images' option will also additionally display few validation images.

## Acknowledgements ##

Parts of the code are inspired by the PyTorch implementation of semantic segmentation models by [@meetshah1995](https://github.com/meetshah1995/pytorch-semseg) and [@zijundeng](https://github.com/zijundeng/pytorch-semantic-segmentation).
