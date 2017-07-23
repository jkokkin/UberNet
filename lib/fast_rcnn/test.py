# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""
#<Iasonas>
# my own test.py
# no longer commenting line by line (too much to comment)
# modified quite a bit - can now store all other task outputs
# as images/matfiles

from fast_rcnn.config import get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob,_conform_to_max,prep_im_for_blob
import os
import scipy.io as sio
import code
import sys
import scipy
from utils.viz import visualize_classes,visualize_gray
import matplotlib
import matplotlib.pyplot as plt

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois,cfg):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    im_list, im_scale_factors =  prep_im_for_blob(im,  cfg['PIXEL_MEANS'],
                                                cfg.TRAIN['SCALES'][0], cfg.TEST['MAX_SIZE'],cfg.TEST['MAX_AREA'],False)
    processed_ims = []
    processed_ims.append(im_list)
    blobs['data']  = im_list_to_blob(processed_ims)
    return blobs, im_scale_factors

def im_detect(net, im,cfg, boxes=None,doDet=True):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes,cfg)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if (doDet and (cfg.DEDUP_BOXES > 0 and not cfg.TRAIN['HAS_RPN'])):
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    im_blob = blobs['data']
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    forward_kwargs            = {'data': blobs['data'].astype(np.float32, copy=False)}
    if doDet:
        blobs['im_info'] = np.array(
                [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
                dtype=np.float32)
        # reshape network inputs
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)
    #print "doDet: ",doDet
    if (not doDet):
        scores =2;
        boxes = 1;
        return scores, boxes;

    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5]
    if len(im_scales)==1:
        boxes = boxes/im_scales[0]
    else:
        boxes[:,0] = boxes[:,0]/im_scales[0]
        boxes[:,1] = boxes[:,1]/im_scales[1]
        boxes[:,2] = boxes[:,2]/im_scales[0]
        boxes[:,3] = boxes[:,3]/im_scales[1]

    scores = net.blobs['cls_prob_0_0'].data

    if cfg.TRAIN['BBOX_REG']:
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg['DEDUP_BOXES'] > 0 and not cfg.TRAIN['HAS_RPN']:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def _vis_im(im_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    im = im_blob[0, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    plt.imshow(im)

def test_net(net, imdb,cfg,max_per_image=100, thresh=0.05, vis=False,task = 'det',nmhead=''):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection threshold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]


    output_dir = get_output_dir(imdb, net)
    #print output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    do_det   = task=='det';
    det_file = os.path.join(output_dir, 'detections.pkl')
    if (do_det and os.path.isfile(det_file)):
        all_boxes =  cPickle.load(open(det_file,'r'));
    else:
        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}
        if not cfg.TRAIN['HAS_RPN']:
            roidb = imdb.roidb

        #only_seg = cfg.TRAIN['USE_SEG'] & (not cfg.TRAIN['USE_DET'])
        do_seg   = task=='seg'
        do_edg   = task=='edg'
        do_nrm   = task=='nrm'
        do_sbd   = task=='sbd'
        do_prt   = task=='prt'
        do_sal   = task=='sal'

        print "num_images = ",num_images

        for i in xrange(num_images):
            next_file = os.path.join(output_dir,imdb.image_index[i] + '.mat')
            next_file_png = os.path.join(output_dir,imdb.image_index[i] + '.png')

            if os.path.exists(next_file) or os.path.exists(next_file_png):
                continue

            im = cv2.imread(imdb.image_path_at(i))
            #shape = im.shape
            #print im.shape
            _t['im_detect'].tic()
            scores, boxes = im_detect(net, im,cfg, None,do_det)
            _t['im_detect'].toc()
            _t['misc'].tic()

            # skip j = 0, because it's the background class
            if do_det:
                for j in xrange(1, imdb.num_classes):
                    inds = np.where(scores[:, j] > 0.05)[0]
                    cls_scores = scores[inds, j]
                    if cfg.TEST.AGNOSTIC:
                        cls_boxes = boxes[inds, 4:8]
                    else:
                        cls_boxes = boxes[inds, j*4:(j+1)*4]

                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    if vis:
                        vis_detections(im, imdb.classes[j], cls_dets)
                    all_boxes[j][i] = cls_dets

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in xrange(1, imdb.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in xrange(1, imdb.num_classes):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]
                _t['misc'].toc()

                print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                      .format(i + 1, num_images, _t['im_detect'].average_time,
                              _t['misc'].average_time)

            #print "here!!"
            if do_edg:
               if nmhead[0:4]!='sigm':
                   nmhead = 'sigmoid-'+nmhead

            #from IPython import embed; embed()

            if (do_seg or do_prt):
                data = net.blobs[nmhead].data.copy();
                data= data.squeeze(axis=0)
                posteriors = data.argmax(axis=0)
                posteriors = posteriors.astype('uint8').copy();
                from PIL import Image
                im = Image.fromarray(posteriors)
                im.save(next_file_png)

            if (do_sal or do_edg):
                #print "nmhead: ",nmhead
                data = net.blobs[nmhead].data.copy();
                if nmhead=='mtn_result1':
                    slice = 0;
                else:
                    slice = 1;
                data= data.squeeze(axis=0)
                posteriors = 255.0*visualize_gray(data,slice);
                posteriors = posteriors.astype('uint8').copy();
                from PIL import Image
                im = Image.fromarray(posteriors)
                im.save(next_file_png)

            if (do_nrm):
                data  =net.blobs[nmhead].data.copy();
                data = -128.*data.copy().squeeze(axis=0) + 128.
                data = np.transpose(data,(1,2,0))
                matplotlib.image.imsave(next_file_png,data.astype('uint8').copy())

            if do_sbd:
                d = {'res':net.blobs[nmhead].data.copy()};
                scipy.io.savemat(next_file,d, oned_as='column')

            # end of for-loop over images

        if do_det:
            with open(det_file, 'wb') as f:
                cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if do_det:
        print 'Evaluating detections'
        imdb.evaluate_detections(all_boxes, output_dir)
