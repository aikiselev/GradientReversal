from keras.callbacks import Callback

from evaluation import compute_cvm, compute_ks, roc_auc_truncated
from misc import predict_probs


class ShowMetrics(Callback):
    def __init__(self, model, Xa, ya, wa, Xc, mc, X, y, verbose=False):
        super().__init__()
        self._model = model
        self.Xa = Xa
        self.ya = ya
        self.wa = wa
        self.Xc = Xc
        self.mc = mc
        self.X = X
        self.y = y
        self.history_ks = []
        self.history_cvm = []
        self.history_auc = []
        self.verbose = verbose

    def get_history(self):
        return self.history_ks, self.history_cvm, self.history_auc

    def calculate_metrics(self, Xa, ya, wa, Xc, mc, X, y):
        pa = self._model.predict_probs(Xa)
        ks = compute_ks(pa[ya == 0], pa[ya == 1], wa[ya == 0], wa[ya == 1])

        pc = self._model.predict_probs(Xc)
        cvm = compute_cvm(pc, mc)

        p = self._model.predict_probs(X)
        auc = roc_auc_truncated(y[:, 1], p)
        return ks, cvm, auc

    def update_statistics(self):
        ks, cvm, auc = self.calculate_metrics(self.Xa, self.ya, self.wa,
                                              self.Xc, self.mc, self.X, self.y)
        if self.verbose:
            print("KS: {} : 0.09 / CvM: {} : 0.002 / AUC: {}".format(ks, cvm, auc))
        self.history_ks.append(ks)
        self.history_cvm.append(cvm)
        self.history_auc.append(auc)

    def on_epoch_end(self, epoch, logs={}):
        self.update_statistics()

    def on_train_end(self, logs={}):
        self.update_statistics()
