from neon.callbacks.callbacks import Callback
import numpy as np
import logging
import cPickle


logger = logging.getLogger(__name__)


class TrainByStageCallback(Callback):
    """
    0. Show model structure
    1. Calculate and display classification error.
    2. Save classification eror
    3. Store best stage
    4. Train by stages
    5. Early Stopping
    6. Save everything on train end
    """

    def __init__(self, model, eval_set, metric, filename='model.pkl', max_patience=10, epoch_freq=1, max_stages=3):
        super(TrainByStageCallback, self).__init__(epoch_freq=epoch_freq)
        self.filename = filename
        self.model = model
        self.eval_set = eval_set
        self.metric = metric

        self.max_stages = max_stages
        self.best_epoch = 0
        self.best_stats = None
        self.best_model = None
        self.patience = 0
        self.max_patience = max_patience
        self.stage = 1
        self.stats_list = []
        self.cost_list = []

    def on_epoch_end(self, callback_data, model, epoch):
        if (epoch + 1) % self.epoch_freq == 0:
            # Calculate and display classification error
            self.eval_set.reset()
            stats = self.model.eval(self.eval_set, metric=self.metric)
            logger.info('Epoch: %d, Classification error: %.2f%%' % (epoch, stats.flatten() * 100))

            # Save classification error
            self.stats_list.extend(stats)
            self.cost_list.extend(self.model.total_cost)

            # Store best stage
            if self.best_stats > stats or self.best_stats is None:
                self.best_epoch = epoch
                self.best_stats = stats
                self.best_model = self.model.serialize()
                self.patience = 0
                cPickle.dump(self.best_model, open('temp_model.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
            else:
                self.patience += 1

            # Train by stages + early stopping
            if self.max_patience <= self.patience:
                if self.stage < self.max_stages:
                    self.stage += 1
                    self.patience = 0
                    if self.model.optimizer.name == 'multiopt':
                        for optimizer in self.model.optimizer.optimizer_mapping:
                            self.model.optimizer.optimizer_mapping[optimizer].learning_rate /= 10
                        logger.info('Recuding learning rate to %f' %
                                    self.model.optimizer.optimizer_mapping['default'].learning_rate)
                    else:
                        self.model.optimizer.learning_rate /= 10
                        logger.info('Recuding learning rate to %f' % self.model.optimizer.learning_rate)
                    # model.deserialize(self.best_model)
                else:
                    self.model.finished = True

    def on_train_begin(self, callback_data, model, epochs):
        logger.info("Model:\n%s", self.model)

    def on_train_end(self,  callback_data, model):
        pdict = dict()
        pdict['model'] = self.best_model
        pdict['misclas'] = self.stats_list
        pdict['cost'] = self.cost_list
        cPickle.dump(pdict, open(self.filename, 'wb'), cPickle.HIGHEST_PROTOCOL)
        logger.info('Best result at epoch %i: %.2f%%' % (self.best_epoch, self.best_stats * 100.))
        model.deserialize(self.best_model)


class MetricCallback(Callback):
    def __init__(self, model, eval_set, metric, epoch_freq=1):
        super(MetricCallback, self).__init__(epoch_freq=epoch_freq)
        self.model = model
        self.eval_set = eval_set
        self.metric = metric

    def on_train_begin(self, callback_data, model, epochs):
        logger.info("Model:\n%s", self.model)

    def on_epoch_end(self, callback_data, model, epoch):
        self.eval_set.reset()
        stats = self.model.eval(self.eval_set, metric=self.metric)
        logger.info('Epoch: %d, Classification error: %.2f%%' % (epoch, stats.flatten() * 100))


class FuzzyPruneCallback(Callback):
    def __init__(self, num_states=10, op='max', start_prune_epoch=5, epoch_freq=1, filename='fprune.pkl',
                 num_prune='auto', one_time=True, model=None):
        super(FuzzyPruneCallback, self).__init__(epoch_freq=epoch_freq)
        self.one_time = one_time
        self.prune = False
        self.start_prune_epoch = start_prune_epoch
        self.num_states = num_states
        self.interval = 1. / num_states
        if isinstance(num_prune, list):
            self.num_prune = num_prune
        elif num_prune == 'auto':
            assert model is not None
            self.num_prune = []
            for idx, l in enumerate(model.layers_to_optimize):
                if l.__class__.__name__ in ['Convolution', 'Linear']:
                    self.num_prune.append(np.prod(l.W.shape, dtype=np.float32))
            self.num_prune = np.sqrt(self.num_prune / np.sum(self.num_prune) / range(1, len(self.num_prune)+1)) * num_states
            print self.num_prune
        else:
            self.num_prune = [num_prune]
        assert op in ['max', 'sum']
        self.op = op
        self.states = []
        self.masks = []
        self.filename = filename

    def reset(self):
        self.states = []
        self.masks = []

    def on_minibatch_end(self, callback_data, model, epoch, minibatch):
        offset = 0
        if len(self.masks) > 0:
            for i, l in enumerate(model.layers_to_optimize):
                if l.__class__.__name__ in ['Convolution', 'Linear']:
                    l.set_params({'params': {'W': l.W.asnumpyarray() * self.masks[i - offset]}})
                    # l.W = l.W * self.masks[i - offset]
                else:
                    offset += 1

    # noinspection PyStringFormat
    def on_train_begin(self, callback_data, model, epochs):
        if self.one_time:
            offset = 0
            for idx, l in enumerate(model.layers_to_optimize):
                if l.__class__.__name__ in ['Convolution', 'Linear']:
                    W = l.W.get()
                    if self.op is 'max':
                        normW = np.abs(W) / np.max(np.abs(W), keepdims=True)
                    elif self.op is 'sum':
                        normW = np.abs(W) / np.sum(np.abs(W), keepdims=True)
                    else:
                        raise Exception('Operation ''%s'' not implemented!' % self.op)
                    normW[normW == 1] = self.num_states
                    for i in range(self.num_states):
                        normW[np.logical_and(normW >= (i * self.interval), normW < ((i + 1) * self.interval))] = i + 1
                    self.states.append(normW)
                    if len(self.num_prune) > 1:
                        masks = (normW > self.num_prune[idx - offset])
                    else:
                        masks = (normW > self.num_prune[0])
                    self.masks.append(masks)
                    l.set_params({'params': {'W': W * masks}})
                    # l.W = l.W * masks
                    print 'Mask %i: %f' % (idx - offset, np.mean(masks))
                else:
                    offset += 1
            num_mask = sum([mask.sum() for mask in self.masks])
            shape_mask = sum([np.prod(mask.shape, dtype=np.float32) for mask in self.masks])
            print 'Overall pruned: %f' % (num_mask / shape_mask)

    # noinspection PyStringFormat
    def on_epoch_end(self, callback_data, model, epoch):
        if self.one_time:
            return
        if (not self.prune) and (epoch + 1) >= self.start_prune_epoch:
            self.prune = True
        if self.prune and (epoch + 1) % self.epoch_freq == 0:
            self.reset()
            offset = 0
            for idx, l in enumerate(model.layers_to_optimize):
                if l.__class__.__name__ in ['Convolution', 'Linear']:
                    W = l.W.asnumpyarray()
                    if self.op is 'max':
                        normW = np.abs(W) / np.max(np.abs(W), keepdims=True)
                    elif self.op is 'sum':
                        normW = np.abs(W) / np.sum(np.abs(W), keepdims=True)
                    else:
                        raise Exception('Operation ''%s'' not implemented!' % self.op)
                    normW[normW == 1] = self.num_states
                    for i in range(self.num_states):
                        normW[np.logical_and(normW >= (i * self.interval), normW < ((i + 1) * self.interval))] = i + 1
                    self.states.append(normW)
                    if len(self.num_prune) > 1:
                        masks = (normW > self.num_prune[idx - offset])
                    else:
                        masks = (normW > self.num_prune[0])
                    self.masks.append(masks)
                    l.set_params({'params': {'W': W * masks}})
                    print 'Mask %i: %f' % (idx - offset, np.mean(masks))
                else:
                    offset += 1

    def on_train_end(self, callback_data, model):
        cPickle.dump({'normW': self.states, 'masks': self.masks}, open(self.filename, 'wb'), cPickle.HIGHEST_PROTOCOL)
