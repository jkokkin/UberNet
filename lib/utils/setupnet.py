import math
import numpy as np
def setup_nmx_layers(net):
    # sets layers used for non-maximum suppression in boundaries (nothing is trained - replicating nmx from edgeboxes)

    to_fix = ['sm1_', 'sm3_', 'cn1_', 'cn3_']
    for it in range(0, len(net.params.items())):
        name_layer = net.params.items()[it][0]
        prefix     = name_layer[0:4]
        for fx in range(0,len(to_fix)):
            if prefix==to_fix[fx]:
                sigma = float(prefix[2])/1.5;
                sv     = int(math.ceil(3.3*sigma));
                st     = 2*sv+ 1;
                x, y   = np.meshgrid(range(-sv,sv+1), range(-sv,sv+1), sparse=False, indexing='xy')
                f      = np.exp( -( x*x + y*y)/(2*sigma*sigma))/(2*np.pi*sigma*sigma);

                weights_cal = np.zeros((1,2,st,st))
                weights_cal[0,1,:,:] = f
                if prefix[0:2]=='cn':
                    weights_cal[0,0,:,:] = f
                net.params[name_layer][0].data[:,:,:,:] = weights_cal
            if prefix=='deri':
                #print name_layer
                fxx = np.array([[0,0,0],[-1,2,-1],[0,0,0]])
                fyy = fxx.transpose();
                fxy = np.array([[0,0,0],[1,-1,0],[-1,1,0]])/2.0;
                weights_cal = np.zeros((3,1,3,3));
                weights_cal[0,0,:,:]    = fxx;
                weights_cal[1,0,:,:]    = -fxy;
                weights_cal[2,0,:,:]    = fyy;
                net.params[name_layer][0].data[:,:,:,:] = weights_cal

    return net
    """
      case 'CLBR'
            [grh,grv] = meshgrid([-3:3],[-3:3]);
            dh = 1./(1+exp(3.5*(abs(grh)-3)));
            dv = 1./(1+exp(3.5*(abs(grv)-3)));
            ds = single(min(dh,dv));
            for k=1:21
                for k_ = [1:21]
                    idxs = (k_-1)*49 + [1:49];
                    if k==k_
                        if k==1
                            weights_cal(k,idxs) = 1./ds(:);
                        else
                            weights_cal(k,idxs) = ds(:);
                        end
                        %else
                        %weights_cal(k,idxs) = -ds(:);
                    end
                end
            end
            doit = 1;
    """

def setup_fusion_layers(net):
    print "Setting up fusion layers"
    for it in range(0,len(net.params.items())):
        name_layer = net.params.items()[it][0]
        has_upscore = name_layer.find('upscore-fuse-')>=0
        has_inprod  = name_layer.find('inprod-fuse-')>=0
        if (has_upscore or has_inprod):
            no_deri     = name_layer.find('deri',0,4)<0   # should not be
            no_split    = name_layer.find('split')<0
            no_seg      = name_layer.find('upscore-fuse-res-seg1')<0
            shape       = net.params[name_layer][0].data.shape;
            no_empty    = len(shape)>0
            if (no_deri and no_split and no_seg and no_empty):
                print "This layer gets updated: ",name_layer
                n_ch    = shape[0];
                n_in    = shape[1]/n_ch;
                vals    = np.zeros(shape);
                eye     = np.eye(n_ch)/n_in;
                #print "# channels: ", n_ch, " # n_in: ",n_in
                #print "eye: ",eye
                for k in range(0,n_in):
                    vals[:,range(n_ch*k,n_ch*(k + 1)),0,0] = eye;

                #print "vals: " ,vals
                net.params[name_layer][0].data[:,:,:,:] = vals;

    return net

import numpy as np
import caffe

def init_network(net_target,pretrained_model,do_shortcut):

    print "initializing network"
    is_vgg = "VGG"  in pretrained_model;
    is_rnet = (not is_vgg);

    #else:

    root_dir        = '/mnt/vol/gfsai-east/ai-group/users/rbg/data/imagenet_models/'
    tasks           = ['seg', 'det'];

    if is_rnet:
        source_weights  = root_dir + 'ResNet-101-model.caffemodel';
        source_proto    = root_dir + 'ResNet-101-deploy.prototxt';
    else:
        source_weights  = root_dir + 'VGG16.v2.caffemodel';
        source_proto    = root_dir + 'VGG_ILSVRC_16_layers_deploy.prototxt';


    if False:
        if is_vgg==False:
            #for it  in range(0,10):
            #t14917192    print "Hey!!!"
            net_target     = caffe.Net(str(source_proto),str(source_weights),caffe.TEST)
            net_target = setup_nmx_layers(net_target);
            net_target = setup_fusion_layers(net_target);
            return net_target

    net_source      = caffe.Net(str(source_proto),str(source_weights),caffe.TEST)
    instantiated   = np.zeros(len(net_target.params.items()));

    renamer = {'cls_score':'fc8',
               'fc8_coco':'score_seg_6_0_hole0',
               'fc8_voc':'score_seg_6_0_hole0',
               'fc8_voc12':'score_seg_6_0_hole0',
               'fc1_seg_c0':'score_seg_6_0_hole6',
               'fc1_seg_c1':'score_seg_6_0_hole12',
               'fc1_seg_c2':'score_seg_6_0_hole18',
               'fc1_seg_c3':'score_seg_6_0_hole24'};

    #task  = 'edg'
    stack           = '0';
    instantiated    = {'':''}
    for task in tasks:
        for it in range(0,len(net_source.params.items())):
            ## take care of all possible ways in which name in reference net may be distorted in target net
            name_source = net_source.params.items()[it][0]
            name_0      = name_source
            if name_0 in renamer:
                name_0 = renamer[name_0]
            if name_0.find('fc8_voc',0,7)>=0:
                name_0 = 'score_seg_6_0_hole0' + name_0[9:];
            cands = [name_0,name_0+'_'+task.lower(),name_0+'_'+task.lower()+'_'+stack]
            if task != 'det':
                cands.append(name_0+'_conv');

            if (name_0=='fc8'):
                continue;
            apps = ['_RS3','_RS4','_RS2','_RS0','_PW','']
            if is_rnet:
                apps.append('det');
            for k in range(0,8):
                apps.append('_{}'.format(k));

            for app in apps:
                for cand in cands:
                    cand_composite = cand + app
                    if cand_composite in net_target.params:
                        if (cand_composite in instantiated) == False:
                            instantiated[cand_composite] = task
                            #print '{} ==> {}'.format(name_source,cand_composite)
                            #target_blob = net_target.params[cand_composite]

                            # do it for the bias and the linear terms
                            for pr in range(0,len(net_source.params[name_source])):
                                #print name_source
                                params = net_source.params[name_source][pr].data

                                ## implement Large-FOV initialization for VGG network
                                if (is_vgg and (name_source=='fc6' or name_source=='fc7') and (task!='det')):
                                    #print '{} ==> {}'.format(name_source,cand_composite)

                                    # perform VGG-decimation
                                    # output dimensions: instead of 1:4096, use 1:4:4096
                                    range_out = range(0,1024*4,4)
                                    # fc6 input dimensions: instead of 1:7, use 1:4:7
                                    range_in  = range(0,7,3)

                                    # decimate the fc6/fc7 weights
                                    if pr==0:
                                        if (name_source=='fc6'):
                                            params = params.reshape((4096,512,7,7))
                                            params = ((params[range_out,:,:,:])[:,:,range_in,:])[:,:,:,range_in]
                                        else:
                                            params = (params[:,:,range_out,:])[:,:,:,range_out]
                                            # this scaling factor was there in the LFOV init parameters
                                            params = params.reshape((1024,1024,1,1))*4.0
                                    else:
                                        # bias term
                                        params = params[:,:,:,range_out]

                                net_target.params[cand_composite][pr].data[:] = params

    #
    for cand_composite in net_target.params:
        if (cand_composite in instantiated) == False:
            print 'could not instantiate: {}'.format(cand_composite)

    #blahblahblah
    net_target = setup_nmx_layers(net_target);
    net_target = setup_fusion_layers(net_target);
    return net_target
