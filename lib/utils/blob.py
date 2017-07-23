# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import math
from fast_rcnn.config import cfg

DEBUG = False
def im_list_to_blob(ims,type=np.float32,domain=[]):
    """
    Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape  = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)  
    
    if len(max_shape)==2:
        ndims = 1
    else:
        ndims = max_shape[2]
                
    blob = np.zeros((num_images, max_shape[0], max_shape[1], ndims),dtype=type)
    for i in xrange(num_images):
        im = ims[i]
        if (ndims==1):
            blob[i, 0:im.shape[0], 0:im.shape[1], 0] = im
        else:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
                    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means,target_size, max_size,max_area,fix_scale):
    """Mean subtract and scale an image for use in a blob."""
    im  = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape_prev = im.shape
    if fix_scale:
        im_scale = [float(1),float(1)]
    else:
        dsize_prelim,   im_scale_prelim   = _conform_to_max(im.shape,       target_size,max_size,max_area)
        dsize_,         im_scale_         = _conform_to_max(dsize_prelim,   target_size,max_size,max_area,1)
        im_scale = [im_scale_prelim[0]*im_scale_[0],im_scale_prelim[1]*im_scale_[1]]
        # stretching in horizontal [0] (x) and vertical [1] (y) direction
        im = cv2.resize(im, dsize=(dsize_[1],dsize_[0]),interpolation=cv2.INTER_LINEAR)
        # but cv2 requires height/width -> permute [0,1]

    if DEBUG:
        print "fix_scale: ",fix_scale
        print "max_size:  ",max_size, " step: ",cfg['SIZE_STEP']
        print "dsize prelim:     ",dsize_prelim
        print "dsize final :     ",dsize_
        print "imscale prelim:   ",im_scale_prelim
        print "imscale step 2:   ",im_scale_
        print "imscale composite:",[im_scale[0],im_scale[1]]
        print "imshape prev:     ",im_shape_prev
        
        print "ratio: ",im.shape[0]/dsize_[0]," , ",im.shape[1]/dsize_[1]
        print "imshp-after: ",im.shape

    return im, im_scale

def prep_sg_for_blob(sg,  target_size, max_size,max_area,hole):
    """ scale a segmentation for use in a blob."""
    sg              = sg.astype(np.uint8, copy=False)
    #dsize_,sg_scale = _conform_to_max(sg.shape,target_size,max_size,max_area)
    dsize_          = _predict_transform(sg.shape,hole);
    sg              = cv2.resize(sg, dsize=(dsize_[1],dsize_[0]), interpolation=cv2.INTER_NEAREST)
    #if DEBUG:
        #print "dize: ",dsize_," transform: ",transform_, "sg.shape: ", sg.shape
    return sg

def _conform_to_max(im_shape,target_size,max_size,max_area,im_scale_in = -1):
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if (im_scale_in==-1):
        im_scale_scalar = float(target_size) / float(im_size_min)
    else:
        im_scale_scalar = 1.0

    # Prevent the biggest axis from being more than MAX_SIZE
    fixed_area = False
    if np.ceil(im_scale_scalar * im_size_max) > max_size:
         if DEBUG:
            print "fixing max size"
         im_scale_scalar = float(max_size) / float(im_size_max)

    cnst        = cfg['SIZE_STEP']
    im_size_y   = im_shape[0]
    im_size_x   = im_shape[1]                    
    anticipated_x = cnst*(math.ceil((im_size_x*im_scale_scalar -1.0)/cnst)) + 1;
    anticipated_y = cnst*(math.ceil((im_size_y*im_scale_scalar -1.0)/cnst)) + 1;
    
    if DEBUG:
        print "anticipated_x: ",anticipated_x,"anticipated_y: ",anticipated_y,"anticipated_area: ",anticipated_x*anticipated_y
    if (anticipated_y*anticipated_x) > max_area:
         if DEBUG:
            print "fixing area"
         fixed_area = True
         #print max_area,im_scale_scalar*im_scale_scalar * im_shape[0]* im_shape[1]
         im_scale_scalar = min(im_scale_scalar,math.sqrt(float(max_area) / float(im_shape[0]* im_shape[1])))
    
    if DEBUG:
        print "im scale scalar: ",im_scale_scalar, " max_size: ",float(max_size), " area: ",float(im_shape[0]* im_shape[1])," max_area: ",max_area," im_shape: ", im_shape[0], im_shape[1]
                 
    if fixed_area:
        im_size_x_  = cnst*(math.floor((im_size_x*im_scale_scalar -1.0)/cnst)) + 1;
        im_size_y_  = cnst*(math.floor((im_size_y*im_scale_scalar -1.0)/cnst)) + 1;
    else:
        im_size_x_  = cnst*(math.ceil((im_size_x*im_scale_scalar -1.0)/cnst)) + 1;
        im_size_y_  = cnst*(math.ceil((im_size_y*im_scale_scalar -1.0)/cnst)) + 1;
    dsize_      = (int(im_size_x_),int(im_size_y_))
    im_scale    = [float(im_size_x_)/float(im_size_x),float(im_size_y_)/float(im_size_y)]
    return dsize_,im_scale
            
def _predict_transform(sg_shape,hole):
    newx  = sg_shape[0]
    newy  = sg_shape[1]
    if DEBUG:
        print "newx,newy: ",newx,newy
    if (hole>1):
        newy    = (((newy-1)/4) +1)/2 + 1
        newx    = (((newx-1)/4) +1)/2 + 1
    else:
        newy    = (((newy-1)/8) +1)/2 + 1
        newx    = (((newx-1)/8) +1)/2 + 1
    if DEBUG:
        print "newx, newy after: ",newx,newy
    sgsize_ = (int(newx), int(newy))
    return sgsize_

