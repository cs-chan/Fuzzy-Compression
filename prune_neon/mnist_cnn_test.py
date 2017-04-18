import logging
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine
from neon.transforms import Softmax, CrossEntropyMulti, Rectlin, Misclassification
from neon.models import Model
from neon.data import ArrayIterator, MNIST
from neon.callbacks.callbacks import Callbacks


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = NeonArgparser(__doc__)
args = parser.parse_args()
be = gen_backend(backend='gpu',
                 batch_size=128,
                 datatype=np.float32)

# setup a dataset iterator
mnist = MNIST(path='../dataset/mnist')
(X_train, y_train), (X_test, y_test), nclass = mnist.load_data()
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

# define model
nfilters = [20, 50, 500]
init_w = Gaussian(scale=0.01)
relu = Rectlin()
common_params = dict(init=init_w, activation=relu)
layers = [
    Conv((5, 5, nfilters[0]), bias=Constant(0.1), padding=0, **common_params),
    Pooling(2, strides=2, padding=0),
    Conv((5, 5, nfilters[1]), bias=Constant(0.1), padding=0, **common_params),
    Pooling(2, strides=2, padding=0),
    Affine(nout=nfilters[2], bias=Constant(0.1), **common_params),
    Affine(nout=10, bias=Constant(0.1), activation=Softmax(), init=Gaussian(scale=0.01))
]
model = Model(layers=layers)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.initialize(train_set, cost)
model.load_params('models/mnist/mnist_cnn.pkl', load_states=False)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, metric=Misclassification(), **args.callback_args)
print('Accuracy = %.2f%%' % (100. - model.eval(valid_set, metric=Misclassification()) * 100))
