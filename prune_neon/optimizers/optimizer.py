# import numpy as np
from neon.optimizers.optimizer import Optimizer, get_param_list


class RMSPropNesterov(Optimizer):
    """
    RMSProp with Nesterov Momentum
    """
    # TODO: max norm constraint
    def __init__(self, stochastic_round=False, momentum=0.5, decay_rate=0.90, learning_rate=1e-4, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, name="rmspropNAG"):
        super(RMSPropNesterov, self).__init__(name=name)

        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.stochastic_round = stochastic_round

    def optimize(self, layer_list, epoch):
        lrate, epsilon, decay = (self.learning_rate, self.epsilon, self.decay_rate)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:

            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))
                states.append(self.be.zeros_like(grad))

            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            # update state
            state = states[0]
            state[:] = decay * state + self.be.square(grad) * (1.0 - decay)

            # update velocity
            velocity = states[1]
            temp_velocity = - (scale_factor * grad * lrate) / self.be.sqrt(state + epsilon)
            velocity[:] = self.momentum * velocity + temp_velocity

            param[:] = param + self.momentum * velocity + temp_velocity


class MaxNormConstraint(Optimizer):
    """
    Max Norm Constraint
    """
    def __init__(self, optimizer, max_col_norm=0.9, max_kern_norm=1.9, name='maxnorm'):
        super(MaxNormConstraint, self).__init__(name=name)
        self.optimizer = optimizer
        self.max_col_norm = max_col_norm
        self.max_kern_norm = max_kern_norm
        self.learning_rate = self.optimizer.learning_rate

    def __setattr__(self, key, value):
        if key == 'learning_rate':
            self.optimizer.learning_rate = value
        super(MaxNormConstraint, self).__setattr__(key, value)

    def optimize(self, layer_list, epoch):
        self.optimizer.optimize(layer_list, epoch=epoch)
        param_list = get_param_list(layer_list)

        for idx, ((param, grad), states) in enumerate(param_list):
            if layer_list[idx].name in ['BiasLayer']:
                continue
            if layer_list[idx].name in ['ConvolutionLayer']:
                max_norm = self.max_kern_norm
                axes = 0
            else:
                max_norm = self.max_col_norm
                axes = 1

            norm = self.be.empty_like(param)
            norm[:] = self.be.sqrt(self.be.sum(self.be.square(param), axis=axes))
            target_norm = self.be.clip(norm, 0., max_norm)
            param[:] = param * target_norm / (norm + 1e-7)
