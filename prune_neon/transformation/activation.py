from neon.transforms.transform import Transform


class LeakyRect(Transform):
    """
    Leaky Relu
    """
    def __init__(self, name='leakyrelu', slope=100):
        super(LeakyRect, self).__init__(name)
        self.slope = 1. / slope

    def __call__(self, x):
        return self.be.maximum(x, x * self.slope)

    def bprop(self, x):
        return self.be.greater(x, 0) * (1.0 - self.slope) + self.slope


class Explin(Transform):
    """
    ELU activation function
    """
    def __init__(self, alpha=1.0, name='elu'):
        super(Explin, self).__init__(name)
        self.alpha = alpha

    def __call__(self, x):
        return self.be.maximum(x, 0) + self.alpha * (self.be.exp(self.be.minimum(x, 0)) - 1)

    def bprop(self, x):
        return self.be.greater(x, 0) + self.be.minimum(x, 0) + self.alpha * self.be.less(x, 0)
