import numpy as np
import caffe
import sys


# noinspection PyShadowingNames,PyStringFormat
def fuzzyprune():
    global prototxt, caffemodel, layers, num_states, interval, num_prune
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '============  Surgery: threshold   ============'

    if num_prune == 'auto':
        num_prune = []
        for idx, layer in enumerate(layers):
            num_prune.append(np.prod(net.params[layer][0].data.shape, dtype=np.float32))
        num_prune = np.sqrt(num_prune / np.sum(num_prune) / range(1, len(num_prune)+1)) * num_states
        num_prune = num_prune.tolist()
        print num_prune

    states = []
    masks = []
    for idx, layer in enumerate(layers):
        W = net.params[layer][0].data
        normW = np.abs(W) / np.max(np.abs(W), keepdims=True)

        normW[normW == 1] = num_states
        for i in range(num_states):
            normW[np.logical_and(normW >= (i * interval), normW < ((i + 1) * interval))] = i + 1
        states.append(normW)
        if isinstance(num_prune, list):
            mask = (normW > num_prune[idx])
        else:
            mask = (normW > num_prune)
        masks.append(mask)
        print 'non-zero Mask percentage %i = %0.5f' % (idx, np.mean(mask))
        W = W * mask
        # print 'non-zero W percentage = %0.5f ' % (np.count_nonzero(W.flatten()) / float(np.prod(W.shape)))
        net.params[layer][0].data[...] = W
        # print 'First check mask shape: ' + str(net.params[layer][0].mask.shape)
        net.params[layer][0].mask[...] = mask
        # print 'Second check mask shape: ' + str(net.params[layer][0].mask.shape)
    total_mask = sum([np.prod(m.shape, dtype=np.float32) for m in masks])
    num_mask = sum([m.sum() for m in masks])
    print 'Overall pruned: %f' % (num_mask / total_mask)

    output_model = caffemodel.split('.')[0] + '_' + suffix + ".caffemodel"
    net.save(output_model)
    return states, masks

caffe.set_mode_gpu()

num_states = 100
interval = 1. / num_states
num_prune = [1, 3, 3, 4, 5, 20, 20, 15]

model_folder = './models/alexnet_finetune_artist'
prototxt = model_folder + '/train_val.prototxt'
caffemodel = model_folder + '/caffe_alexnet_artist_train.caffemodel'
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8_artist']
suffix = 'fp'

results = fuzzyprune()
print 'Fuzzy pruning finished!'
