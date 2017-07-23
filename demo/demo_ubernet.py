# load modules
import _init_paths
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import numpy
import re
import sys
import caffe
import rpn
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test   import im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from collections import namedtuple
from PIL import Image
from utils.setupnet import setup_nmx_layers, setup_fusion_layers
from utils.viz import visualize_detection,visualize_soft,replicate,normalize13,visualize_gray
import scipy.io as sio

#local settings -  update your paths accordingly
UBERNET_ROOT  = '/home/iason/ubernet'
MODEL_PATH    = UBERNET_ROOT + '/model/'
CFG_PATH      = MODEL_PATH + 'cfg.yml'
PROTOTXT_PATH = MODEL_PATH + 'test.prototxt'
SNAPSHOT_PATH = MODEL_PATH + 'model.caffemodel'
DEMO_PATH     = UBERNET_ROOT + '/demo/'
IM_NAME       = DEMO_PATH  + 'horse.jpg'
DO_SHOW       = False                 #Generate images or else store to folder
SV_DIR        = DEMO_PATH   + '/res/' #if do_show=False, this is where images are saved
DEV_ID        = 1                     #GPU Device ID

# task-specific functions used to visualize the different network layers
output_info = namedtuple('Output', ['name', 'post_fun'])
names_used = {
    'sigmoid-upscore-fuse-mr-prt': output_info(name='human parts', post_fun=visualize_soft),
    'sigmoid-upscore-fuse-mr-sbd': output_info(name='semantic boundaries', post_fun=visualize_soft),
    'sigmoid-upscore-fuse-mr-seg': output_info(name='semantic segmentation', post_fun=visualize_soft),
    'upscore-fuse-mr-nrm_nrmed': output_info(name='surface normals', post_fun=lambda x: x.transpose(1, 2, 0)/2. + .5),
    'sigmoid-upscore-fuse-mr-sal': output_info(name='saliency', post_fun=replicate),
    'sigmoid-upscore-fuse-mr-edg': output_info(name='boundaries', post_fun=replicate),
}

# CNN code starts here
# load network
cfg_from_file(CFG_PATH)
net             = caffe.Net(str(PROTOTXT_PATH), str(SNAPSHOT_PATH), caffe.TEST)
net             = setup_nmx_layers(net);

caffe.set_mode_gpu()
caffe.set_device(DEV_ID)

# load image & run
im_in   = np.asarray(cv2.imread(IM_NAME))
scores, boxes = im_detect(net,im_in,cfg)

# CNN code ends here
# visualize results

if DO_SHOW:
    plt.figure(figsize=(16,8))
    plt.subplot(2, 4, 1)
    plt.title("Input")
    plt.imshow(im_in)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 2)
    visualize_detection(im_in,boxes,scores,.5,cfg)
    plt.title("Object Detection", fontsize=10)
    plt.xticks([])
    plt.yticks([])

    for i, key in enumerate(names_used, start=1):
        plt.subplot(2, 4, i+2)
        im = names_used[key].post_fun(net.blobs[key].data.squeeze(axis=0))
        plt.title(names_used[key].name.title(), fontsize=10)
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
    plt.show()

else:
    fig = plt.figure(frameon=False)
    ax1 = fig.add_subplot(111,aspect='equal')
    visualize_detection(im_in,boxes,scores,.5,cfg)

    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    fig.savefig(SV_DIR+'im_detection.png')

    for i, key in enumerate(names_used, start=1):
        im = names_used[key].post_fun(net.blobs[key].data.squeeze(axis=0))

        fig = plt.figure(frameon=False)
        ax1 = fig.add_subplot(111,aspect='equal')
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()
        fig.savefig(SV_DIR+'im_'+names_used[key].name+'.png')
        plt.clf()
