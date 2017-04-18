from neon.transforms.cost import Cost


class MulticlsSVMLoss(Cost):
    def __init__(self, delta=1.):
        self.delta = delta

    def __call__(self, y, t):
        T = self.be.empty_like(y)
        T[:] = self.be.max(y * t, axis=0)
        # T = self.be.array(self.be.max(y * t, axis=0).asnumpyarray(), y.shape[0], axis=0)
        margin = self.be.square(self.be.maximum(0, y - T + self.delta)) * 0.5
        return self.be.sum(margin) / self.be.bsz

    def bprop(self, y, t):
        T = self.be.empty_like(y)
        T[:] = self.be.max(y * t, axis=0)
        return self.be.maximum(0, y - T + self.delta) / self.be.bsz


class L1SVMLoss(Cost):
    def __init__(self, C=10):
        self.C = C

    def __call__(self, y, t):
        return self.C * self.be.sum(self.be.square(self.be.maximum(0, 1 - y * (t * 2 - 1)))) * 0.5 / y.shape[0]

    def bprop(self, y, t):
        return - self.C * (t * 2 - 1) * self.be.maximum(0, 1 - y * (t * 2 - 1)) / self.be.bsz / y.shape[0]
