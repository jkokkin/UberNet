import math
import numpy as np
from fast_rcnn.nms_wrapper import nms
import os
import matplotlib
import matplotlib.pyplot as plt

voc_classes=['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','sofa','train','tv']

# BGR ordering
def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

def PascalColormap(num_colors=256):
    """The Pascal colormap as taken from the matlab code.
    Args:
      num_colors: the numbers of colours.
    Returns:
      a numpy array of dimension [num_colors, 3].
    """
    cmap = np.zeros((num_colors, 3), dtype=np.float)
    for i in range(num_colors):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap / 255.
colormap = PascalColormap()

def visualize_classes(activations):
    act = activations.argmax(axis=0)
    return colormap[act]

def visualize_soft(activations):
    shape = activations.shape;
    total  = np.zeros((shape[1],shape[2],3))
    for i in range(activations.shape[0]):
        #print "i: ",i
        strength = activations[i]
        color    = colormap[i]
        for c in range(0,3):
            #print "c: ",c
            total[:,:,c] = total[:,:,c]+  strength*color[c]
    return total


def visualize_gray(activations,slice=1):
    shape = activations.shape;
    #print shape
    total  = np.zeros((shape[1],shape[2],3))
    strength = activations[slice]
    for c in range(0,3):
          total[:,:,c] = total[:,:,c]+  strength
    return total

def visualize_detection(im,boxes,scores,thresh,cfg):
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    for j in xrange(1, scores.shape[1]):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        dets = cls_dets[keep, :]
        for i in xrange(np.minimum(10, dets.shape[0])):
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=colormap[j], linewidth=5)
                )
                left    = bbox[0];
                bottom  = bbox[1];
                plt.text(left,bottom,voc_classes[j-1],horizontalalignment='left',verticalalignment='top',fontsize=15,bbox={'facecolor':colormap[j], 'alpha':0.95, 'pad':10})

def flip(img):
    flipped = img[::-1, :, :].copy()
    return {
        "flipped": flipped
    }

def replicate(activations):
    shape = activations.shape;
    total  = np.zeros((shape[1],shape[2],3))
    total[:,:,0] = activations[0];
    total[:,:,1] = activations[0];
    total[:,:,2] = activations[0];
    return total

def normalize13(activations):
    shape = activations.shape;
    total  = np.zeros((shape[1],shape[2],3))
    for d in xrange(0,2):
        din = activations[d];
        mx = din.max(axis=1); mx = mx.max(axis=0);
        mn = din.min(axis=1); mn = mn.min(axis=0);

        total[:,:,d] = (din - mn)/mx;
    return total

def normalize46(activations):
    shape = activations.shape;
    total  = np.zeros((shape[1],shape[2],3))
    for d in xrange(3,5):
        din = activations[d];
        mx = din.max(axis=1); mx = mx.max(axis=0);
        mn = din.min(axis=1); mn = mn.min(axis=0);
        total[:,:,d-3] = (din - mn)/mx;
    return total
