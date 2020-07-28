import numpy as np
import tensorflow as tf


class CosineAnnealingLearningRateSchedule(tf.keras.callbacks.Callback):

    def __init__(self, n_epochs, n_cycles, max_lr):
        super().__init__()
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.max_lr = max_lr
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, max_lr):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return max_lr / 2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.max_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)


class SnapshotEnsemble(tf.keras.callbacks.Callback):
    """
    Implementation of Snapshot Ensembles

    Code taken from: https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/

    As described in: https://arxiv.org/pdf/1704.00109.pdf
    Cosine annealing: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, family_name, n_epochs=30, n_cycles=30, max_lr=0.001):
        super().__init__()
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.max_lr = max_lr
        self.lrs = []
        self.family_name = family_name

    def cosine_annealing(self, epoch, n_epochs, n_cycles, max_lr):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return max_lr / 2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        if not logs:
            logs = {}
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.max_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        epochs_per_cycle = np.floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            run_name = self.family_name + '__{}/best_weights.h5'.format(int((epoch) / epochs_per_cycle))
            self.model.save(run_name)
