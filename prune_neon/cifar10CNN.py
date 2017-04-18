import logging
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Dropout
from neon.optimizers.optimizer import MultiOptimizer, GradientDescentMomentum
from neon.transforms import Softmax, CrossEntropyMulti, Rectlin, Misclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks
from callbacks.callbacks import TrainByStageCallback


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = NeonArgparser(__doc__)
args = parser.parse_args()
args.backend = 'gpu'
args.datatype = np.float16
be = gen_backend(backend=args.backend,
                 batch_size=128,
                 rng_seed=123,
                 device_id=args.device_id,
                 datatype=args.datatype)

options = dict(repo_dir='../dataset/Cifar10',
               inner_size=32,
               subset_pct=100,
               dtype=args.datatype)
train_set = ImageLoader(set_name='train', scale_range=32, shuffle=True, **options)
valid_set = ImageLoader(set_name='validation', scale_range=32, do_transforms=False, **options)

# define model
nfilters = [96, 192, 256]
init_w = Gaussian(scale=0.01)
relu = Rectlin()
common_params = dict(init=init_w, activation=relu)
convp1 = dict(padding=1, **common_params)
layers = [
    Conv((3, 3, nfilters[0]), bias=Constant(0.1), **convp1),
    Conv((3, 3, nfilters[0]), bias=Constant(0.1), **convp1),
    Pooling(3, strides=2, padding=1),   # 32 -> 16
    Dropout(keep=0.7),
    Conv((3, 3, nfilters[1]), bias=Constant(0.1), **convp1),
    Conv((3, 3, nfilters[1]), bias=Constant(0.1), **convp1),
    Conv((3, 3, nfilters[1]), bias=Constant(0.1), **convp1),
    Pooling(3, strides=2, padding=1),   # 16 -> 8
    Dropout(keep=0.8),
    Conv((3, 3, nfilters[2]), bias=Constant(0.1), **convp1),
    Conv((3, 3, nfilters[2]), bias=Constant(0.1), **convp1),
    Conv((3, 3, nfilters[2]), bias=Constant(0.1), **convp1),
    Pooling(3, strides=2, padding=1),   # 8 -> 4
    Dropout(keep=0.7),
    Affine(nout=10, bias=Constant(0.1), activation=Softmax(), init=Gaussian(scale=0.01))
]
model = Model(layers=layers)

# define optimizer
opt_w = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9, wdecay=0.0005)
opt_b = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
opt = MultiOptimizer({'default': opt_w, 'Bias': opt_b}, name='multiopt')

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, metric=Misclassification(), **args.callback_args)
callbacks.add_callback(TrainByStageCallback(model, valid_set, Misclassification(), max_patience=10))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
logger.info('Training ...')
model.fit(train_set, optimizer=opt, num_epochs=250, cost=cost, callbacks=callbacks)
print('Accuracy = %.2f%%' % (100. - model.eval(valid_set, metric=Misclassification()) * 100))

