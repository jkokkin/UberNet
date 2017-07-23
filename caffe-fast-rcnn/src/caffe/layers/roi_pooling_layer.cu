#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/fast_rcnn_layers.hpp"
//Iasonas' roi-pooling layer
// no longer commenting line by line (too much to comment)
//Changes w.r.t. standard:
// (i) allows using atrous convolution in conv5_3 (skips every second feature)
// (ii) experimental code on  pooling with different regions (Gidaris + Komodakis) - not used in ubernet, but not removed due to lazyness

using std::max;
using std::min;
namespace caffe {

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int bb_type, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data,bool is_rich_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    Dtype RoiManip[4],RoiInterior[4];
    for (int i=0; i<4; i++)
    {
    RoiManip[i]    = bottom_rois[i+1];
    RoiInterior[i] = bottom_rois[i+1];
    }
    bool HasHole = false;
    Dtype cen_w = Dtype(bottom_rois[1] + bottom_rois[3])/2.0;
    Dtype scl_w = Dtype(bottom_rois[3] - bottom_rois[1])/2.0;
    Dtype cen_h = Dtype(bottom_rois[2] + bottom_rois[4])/2.0;
    Dtype scl_h = Dtype(bottom_rois[4] - bottom_rois[2])/2.0;
    enum {START_W,START_H,END_W,END_H};
    enum {ORIG,LEFT,RIGHT,UP,DOWN,CENTER,HOLEI,HOLEII,HOLEIII,HOLEIV};
    Dtype scale_in = Dtype(1.0); Dtype scale_out = Dtype(1.0);

    switch (bb_type){
      case ORIG:
        break;
      case LEFT:
        RoiManip[END_W]   = cen_w;
        break;
      case RIGHT:
        RoiManip[START_W] = cen_w;
        break;
      case UP:
        RoiManip[END_H]   = cen_h;
        break;
      case DOWN:
        RoiManip[START_H] = cen_h;
        break;
      case CENTER:
        scale_out = 0.5;
        break;
      case HOLEI:
        scale_out = 0.8;
        scale_in  = 0.3;
        HasHole = true;
        break;
      case HOLEII:
        scale_out = 1.0;
        scale_in  = 0.5;
        HasHole = true;
        break;
      case HOLEIII:
        scale_out = 1.5;
        scale_in  = 0.8;
        HasHole = true;
        break;
      case HOLEIV:
        scale_out = 1.8;
        scale_in  = 1.0;
        HasHole = true;
        break;
      }
      if (scale_out!=Dtype(1.0))
      {
        RoiManip[START_W] = cen_w - scale_out*scl_w;
        RoiManip[END_W]   = cen_w + scale_out*scl_w;
        RoiManip[START_H] = cen_h - scale_out*scl_h;
        RoiManip[END_H]   = cen_h + scale_out*scl_h;
      }
      if (HasHole)
      {
        RoiInterior[START_W] = cen_w - scale_in*scl_w;
        RoiInterior[END_W]   = cen_w + scale_in*scl_w;
        RoiInterior[START_H] = cen_h - scale_in*scl_h;
        RoiInterior[END_H]   = cen_h + scale_in*scl_h;
      }

    int hole_start_w = round(RoiInterior[0] * spatial_scale);
    int hole_start_h = round(RoiInterior[1] * spatial_scale);
    int hole_end_w = round(RoiInterior[2] * spatial_scale);
    int hole_end_h = round(RoiInterior[3] * spatial_scale);

    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(RoiManip[0] * spatial_scale);
    int roi_start_h = round(RoiManip[1] * spatial_scale);
    int roi_end_w = round(RoiManip[2] * spatial_scale);
    int roi_end_h = round(RoiManip[3] * spatial_scale);


    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;

    int step = 1 + (int) is_rich_;
    for (int h = hstart; h < hend; h+=step) {
      for (int w = wstart; w < wend; w+=step) {
        if ((HasHole)&&(h>hole_start_h)&&(h<hole_end_h)&&(w>hole_start_w)&&(w<hole_end_w))
            continue;
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();

  int count = top[0]->count();
//  LOG(INFO)<<bb_type_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, bb_type_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data,is_rich_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,const int bb_type,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }
     Dtype RoiManip[4];
     Dtype RoiInterior[4];
    for (int i=0; i<4; i++)
    {
    RoiManip[i]    = offset_bottom_rois[i+1];
    RoiInterior[i] = offset_bottom_rois[i+1];
    }
    bool HasHole = false;
    Dtype cen_w = Dtype(offset_bottom_rois[1] + offset_bottom_rois[3])/2.0;
    Dtype scl_w = Dtype(offset_bottom_rois[3] - offset_bottom_rois[1])/2.0;
    Dtype cen_h = Dtype(offset_bottom_rois[2] + offset_bottom_rois[4])/2.0;
    Dtype scl_h = Dtype(offset_bottom_rois[4] - offset_bottom_rois[2])/2.0;
    enum {START_W,START_H,END_W,END_H};
    enum {ORIG,LEFT,RIGHT,UP,DOWN,CENTER,HOLEI,HOLEII,HOLEIII,HOLEIV};
    Dtype scale_in = 1.0; Dtype scale_out = 1.0;

    switch (bb_type){
      case ORIG:
        break;
      case LEFT:
        RoiManip[END_W]   = cen_w;
        break;
      case RIGHT:
        RoiManip[START_W] = cen_w;
        break;
      case UP:
        RoiManip[END_H]   = cen_h;
        break;
      case DOWN:
        RoiManip[START_H] = cen_h;
        break;
      case CENTER:
        scale_out = 0.7;
        break;
      case HOLEI:
        scale_out = 0.8;
        scale_in  = 0.3;
        HasHole = true;
        break;
      case HOLEII:
        scale_out = 1.0;
        scale_in  = 0.5;
        HasHole = true;
        break;
      case HOLEIII:
        scale_out = 1.5;
        scale_in  = 0.7;
        HasHole = true;
        break;
      case HOLEIV:
        scale_out = 1.8;
        scale_in  = 1.0;
        HasHole = true;
        break;
      }
      if (scale_out!=1.0)
      {
        RoiManip[START_W] = cen_w - scale_out*scl_w;
        RoiManip[END_W]   = cen_w + scale_out*scl_w;
        RoiManip[START_H] = cen_h - scale_out*scl_h;
        RoiManip[END_H]   = cen_h + scale_out*scl_h;
      }
      if (HasHole)
      {
        RoiInterior[START_W] = cen_w - scale_in*scl_w;
        RoiInterior[END_W]   = cen_w + scale_in*scl_w;
        RoiInterior[START_H] = cen_h - scale_in*scl_h;
        RoiInterior[END_H]   = cen_h + scale_in*scl_h;
      }

    int roi_start_w = round(RoiManip[0] * spatial_scale);
    int roi_start_h = round(RoiManip[1] * spatial_scale);
    int roi_end_w   = round(RoiManip[2] * spatial_scale);
    int roi_end_h   = round(RoiManip[3] * spatial_scale);


      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }
      if (HasHole)
      {
        // Skip if  (h, w) is in the region's hole
        int hole_start_w = round(RoiInterior[0] * spatial_scale);
        int hole_start_h = round(RoiInterior[1] * spatial_scale);
        int hole_end_w = round(RoiInterior[2] * spatial_scale);
        int hole_end_h = round(RoiInterior[3] * spatial_scale);
        if (h>hole_start_h && h<hole_end_h && w>hole_start_w && w<hole_end_w )
            continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_,bb_type_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);

}  // namespace caffe
