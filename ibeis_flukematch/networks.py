import lasagne.layers as ll
from lasagne.nonlinearities import linear, softmax, sigmoid, rectify
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam, nesterov_momentum
from lasagne.init import Orthogonal, Constant
from lasagne.regularization import l2, regularize_network_params
import theano.tensor as T
import theano
import sys



#----------- NOTCH TIPS EXTRACTOR

def build_kpextractor64():
    inp = ll.InputLayer(shape=(None, 1, 64, 64), name='input')
    # we're going to build something like what Daniel Nouri made for Facial Keypoint detection for a base reference
    # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 32 x 32
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 16 x 16
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 8 x 8
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    # larger max pool to reduce parameters in the FC layer
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2) # now down to 4x4
    bn4 = ll.BatchNormLayer(mp4)
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2) # down to 2x2
    bn5 = ll.BatchNormLayer(mp5)
    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(bn5, num_units=256, nonlinearity=rectify)
    bn6 = ll.BatchNormLayer(fc1)
    #dp1 = ll.DropoutLayer(bn1, p=0.5)
    fc2 = ll.DenseLayer(bn6, num_units=64, nonlinearity=rectify)
    #dp2 = ll.DropoutLayer(fc2, p=0.5)
    bn7 = ll.BatchNormLayer(fc2)
    out = ll.DenseLayer(bn7, num_units=6, nonlinearity=linear)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs

def build_kpextractor128():
    inp = ll.InputLayer(shape=(None, 1, 128, 128), name='input')
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 64 x 64
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 32 x 32
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 16 x 16
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2) # now down to 8 x 8
    bn4 = ll.BatchNormLayer(mp4)
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2) # down to 4 x 4
    bn5 = ll.BatchNormLayer(mp5)

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2) # down to 4 x 4
    bn6 = ll.BatchNormLayer(mp6)

    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(bn6, num_units=256, nonlinearity=rectify)
    bn1_fc = ll.BatchNormLayer(fc1)
    #dp1 = ll.DropoutLayer(bn1, p=0.5)
    fc2 = ll.DenseLayer(bn1_fc, num_units=64, nonlinearity=rectify)
    #dp2 = ll.DropoutLayer(fc2, p=0.5)
    bn2_fc = ll.BatchNormLayer(fc2)
    out = ll.DenseLayer(bn2_fc, num_units=6, nonlinearity=linear)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs


def build_kpextractor128_decoupled():
    inp = ll.InputLayer(shape=(None, 1, 128, 128), name='input')
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2, name='mp1') # now down to 64 x 64
    bn1 = ll.BatchNormLayer(mp1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp2') # now down to 32 x 32
    bn2 = ll.BatchNormLayer(mp2, name='bn2')
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2, name='mp3') # now down to 16 x 16
    bn3 = ll.BatchNormLayer(mp3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp4') # now down to 8 x 8
    bn4 = ll.BatchNormLayer(mp4, name='bn4')
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2, name='mp5') # down to 4 x 4
    bn5 = ll.BatchNormLayer(mp5, name='bn5')

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp6') # down to 4 x 4
    bn6 = ll.BatchNormLayer(mp6, name='bn6')
    dp0 = ll.DropoutLayer(bn6, p=0.5, name='dp0')

    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(dp0, num_units=256, nonlinearity=rectify, name='fc1')
    bn1_fc = ll.BatchNormLayer(fc1, name='bn1_fc')
    dp1 = ll.DropoutLayer(bn1_fc, p=0.5, name='dp1')
    # so what we're going to do here instead is break this into three separate layers (each 32 units)
    # then each of these layers goes into a separate out, and out_rs will be a merge and then reshape
    fc2_left = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify, name='fc2_left')
    fc2_right = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify, name='fc2_right')
    fc2_notch = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify, name='fc2_notch')

    out_left = ll.DenseLayer(fc2_left, num_units=2, nonlinearity=linear, name='out_left')
    out_right = ll.DenseLayer(fc2_right, num_units=2, nonlinearity=linear, name='out_right')
    out_notch = ll.DenseLayer(fc2_notch, num_units=2, nonlinearity=linear, name='out_notch')

    out = ll.ConcatLayer([out_left, out_right, out_notch], axis=1, name='out')
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2), name='out_rs')

    return out_rs

#----------- TRAILING EDGE SCORER

class Softmax4D(ll.Layer):
    def get_output_for(self, input, **kwargs):
        si = input.reshape((input.shape[0], input.shape[1], -1))
        shp = (si.shape[0], 1, si.shape[2])
        exp = T.exp(si - si.max(axis=1).reshape(shp))
        softmax_expression = (exp / (exp.sum(axis=1).reshape(shp) + 1e-7) ).reshape(input.shape)
        return softmax_expression

def crossentropy_flat(pred, true):
    # basically we have a distribution output that's in the shape batch, prob, h, w
    # it doesn't look like we can apply the nnet categorical cross entropy easily on a tensor4
    # so we'll have to flatten it out to a tensor2, which is a pain in the ass but easily done

    pred2 = pred.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    true2 = true.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)

    return T.nnet.categorical_crossentropy(pred2, true2)


def build_segmenter_simple():
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(7,7), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2')
    conv3 = ll.Conv2DLayer(conv2, num_filters=128, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=64, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4')
    conv5 = ll.Conv2DLayer(conv4, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv5')
    conv6 = ll.Conv2DLayer(conv5, num_filters=16, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv6')

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(conv6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return [softmax]

def build_segmenter_upsample():
    # downsample down to a small region, then upsample all the way back up
    # Note: w/o any learning on the upsampler, we're limited in how far we can downsample
    # there will always be an error signal unless the loss fn is run on downsampled targets...
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    mp1 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    conv4 = ll.Conv2DLayer(conv3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    mp2 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    conv6 = ll.Conv2DLayer(conv5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    mp3 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    conv8 = ll.Conv2DLayer(conv7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    # f 68 s 8
    # now start the upsample
    up = ll.Upscale2DLayer(conv8, 8, name='upsample_8x')
    conv_f = ll.Conv2DLayer(up, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear, name='conv_final')
    softmax = Softmax4D(conv_f, name='4dsoftmax')
    return [softmax]


def build_segmenter_jet():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    mp1 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    conv4 = ll.Conv2DLayer(conv3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    mp2 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    conv6 = ll.Conv2DLayer(conv5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    mp3 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    conv8 = ll.Conv2DLayer(conv7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)
    conv_f8 = ll.Conv2DLayer(conv8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')
    up8 = ll.Upscale2DLayer(softmax_8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX 8 AND PRED ON CONV 6
    softmax_4up = ll.Upscale2DLayer(softmax_8, 2, name='upsample_4x_pre') # 4x downsample
    conv_f6 = ll.Conv2DLayer(conv6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f6, name='4dsoftmax_4x') # 4x downsample
    softmax_4_merge = ll.ElemwiseSumLayer([softmax_4, softmax_4up], coeffs=0.5, name='softmax_4_merge')

    up4 = ll.Upscale2DLayer(softmax_4_merge, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_4_MERGE AND CONV 4
    softmax_2up = ll.Upscale2DLayer(softmax_4_merge, 2, name='upsample_2x_pre') # 2x downsample
    conv_f4 = ll.Conv2DLayer(conv4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f4, name='4dsoftmax_2x')
    softmax_2_merge = ll.ElemwiseSumLayer([softmax_2, softmax_2up], coeffs=0.5, name='softmax_2_merge')

    up2 = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample

    return [up8, up4, up2]

def build_segmenter_jet_2():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    # this jet will have another conv layer in the final upsample
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    mp1 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    conv4 = ll.Conv2DLayer(conv3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    mp2 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    conv6 = ll.Conv2DLayer(conv5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    mp3 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    conv8 = ll.Conv2DLayer(conv7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)
    conv_f8 = ll.Conv2DLayer(conv8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')
    up8 = ll.Upscale2DLayer(softmax_8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX 8 AND PRED ON CONV 6
    softmax_4up = ll.Upscale2DLayer(softmax_8, 2, name='upsample_4x_pre') # 4x downsample
    conv_f6 = ll.Conv2DLayer(conv6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f6, name='4dsoftmax_4x') # 4x downsample
    softmax_4_merge = ll.ElemwiseSumLayer([softmax_4, softmax_4up], coeffs=0.5, name='softmax_4_merge')

    up4 = ll.Upscale2DLayer(softmax_4_merge, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_4_MERGE AND CONV 4
    softmax_2up = ll.Upscale2DLayer(softmax_4_merge, 2, name='upsample_2x_pre') # 2x downsample
    conv_f4 = ll.Conv2DLayer(conv4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f4, name='4dsoftmax_2x')
    softmax_2_merge = ll.ElemwiseSumLayer([softmax_2, softmax_2up], coeffs=0.5, name='softmax_2_merge')

    up2 = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_2_MERGE AND CONV 2
    softmax_1up = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_1x_pre') # 1x downsample (i.e. no downsample)
    conv_f2 = ll.Conv2DLayer(conv2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_1xpred')

    softmax_1 = Softmax4D(conv_f2, name='4dsoftmax_1x')
    softmax_1_merge = ll.ElemwiseSumLayer([softmax_1, softmax_1up], coeffs=0.5, name='softmax_1_merge')

    # this is where up1 would go but that doesn't make any sense
    return [up8, up4, up2, softmax_1_merge]

def build_segmenter_jet_preconv():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    # this jet will have another conv layer in the final upsample
    # difference here is that instead of combining softmax layers in the jet, we'll upsample before the conv_f* layer
    # this will certainly make the model slower, but should give us better predictions...
    # The awkward part here is combining the intermediate conv layers when they have different filter shapes
    # We could:
    #   concat them
    #   have intermediate conv layers that bring them to the shape needed then merge them
    # in the interests of speed we'll just concat them, though we'll have a ton of filters at the end
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    bn1 = ll.BatchNormLayer(conv1, name='bn1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    bn2 = ll.BatchNormLayer(conv2, name='bn2')
    mp1 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    bn3 = ll.BatchNormLayer(conv3, name='bn3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    bn4 = ll.BatchNormLayer(conv4, name='bn4')
    mp2 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    bn5 = ll.BatchNormLayer(conv5, name='bn5')
    conv6 = ll.Conv2DLayer(conv5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    bn6 = ll.BatchNormLayer(conv6, name='bn6')
    mp3 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    bn7 = ll.BatchNormLayer(conv7, name='bn7')
    conv8 = ll.Conv2DLayer(conv7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    bn8 = ll.BatchNormLayer(conv8, name='bn8')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)

    up8 = ll.Upscale2DLayer(bn8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample
    conv_f8 = ll.Conv2DLayer(up8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')

    ## COMBINE BY UPSAMPLING CONV 8 AND CONV 6
    conv_8_up2 = ll.Upscale2DLayer(bn8, 2, name='upsample_c8_2') # 4x downsample
    concat_c8_c6 = ll.ConcatLayer([conv_8_up2, bn6], axis=1, name='concat_c8_c6')
    up4 = ll.Upscale2DLayer(concat_c8_c6, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample
    conv_f4 = ll.Conv2DLayer(up4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f4, name='4dsoftmax_4x') # 4x downsample

    ## COMBINE BY UPSAMPLING CONCAT_86 AND CONV 4
    concat_86_up2 = ll.Upscale2DLayer(concat_c8_c6, 2, name='upsample_concat_86_2') # 2x downsample
    concat_ct86_c4 = ll.ConcatLayer([concat_86_up2, bn4], axis=1, name='concat_ct86_c4')

    up2 = ll.Upscale2DLayer(concat_ct86_c4, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample
    conv_f2 = ll.Conv2DLayer(up2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f2, name='4dsoftmax_2x')


    ## COMBINE BY UPSAMPLING CONCAT_864 AND CONV 2
    concat_864_up2 = ll.Upscale2DLayer(concat_ct86_c4, 2, name='upsample_concat_86_2') # no downsample
    concat_864_c2 = ll.ConcatLayer([concat_864_up2, bn2], axis=1, name='concat_ct864_c2')
    conv_f1 = ll.Conv2DLayer(concat_864_c2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_1xpred')

    softmax_1 = Softmax4D(conv_f1, name='4dsoftmax_1x')

    # this is where up1 would go but that doesn't make any sense
    return [softmax_8, softmax_4, softmax_2, softmax_1]



def build_segmenter_simple_absurd_res():
    sys.setrecursionlimit(1500)
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    n_layers = 64 # should get a 128 x 128 receptive field
    layers = [inp]
    for i in range(n_layers):
        # every 2 layers, add a skip connection
        layers.append(ll.Conv2DLayer(layers[-1], num_filters=8, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear, name='conv%d' % (i+1)))
        layers.append(ll.BatchNormLayer(layers[-1], name='bn%i' % (i+1)))
        if (i % 2 == 0) and (i != 0):
            layers.append(ll.ElemwiseSumLayer([layers[-1], # prev layer
                                              layers[-6],] # 3 actual layers per block, skip the previous block
                                              ))
        layers.append(ll.NonlinearityLayer(layers[-1], nonlinearity=rectify))

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(layers[-1], num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return [softmax]


