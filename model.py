import theano.gof
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model, Sequential

from evaluation import *


class ReverseGradient(theano.gof.Op):
    # Custom Theano operaion for gradient reversal layer
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


class GradientReversal(Layer):
    # Gradient reversal layer, described in http://arxiv.org/pdf/1409.7495.pdf
    def __init__(self, l, **kwargs):
        super().__init__(**kwargs)
        self.op = ReverseGradient(l)

    def call(self, x, mask=None):
        return self.op(x)


def build_sequential_model(hidden_layers, name=""):
    m = Sequential(name=name)
    for layer in hidden_layers:
        m.add(layer)
    return m


def create_clf(feature_extractor, label_classifier, domain_classifier, lam):
    model_input = Input(shape=(feature_extractor.input_shape[1],))
    features = feature_extractor(model_input)

    label_class = label_classifier(features)

    grl = GradientReversal(lam, input_shape=(domain_classifier.input_shape[1],))
    domain_class = domain_classifier(grl(features))

    m = Model(input=[model_input], output=[label_class, domain_class])
    m.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1, 1], metrics=['accuracy', 'accuracy'], optimizer='rmsprop')
    return m


def show_metrics(model, Xa, ya, wa, Xc, mc, X, y):
    pa = predict_probs_model(model, Xa)
    ks = compute_ks(pa[ya == 0], pa[ya == 1], wa[ya == 0], wa[ya == 1])

    pc = predict_probs_model(model, Xc)
    cvm = compute_cvm(pc, mc)

    p = predict_probs_model(model, X)
    auc = roc_auc_truncated(y[:, 1], p)
    print("KS: {} : 0.09 / CvM: {} : 0.002 / AUC: {}".format(ks, cvm, auc))
    return ks, cvm, auc


class ShowMetrics(Callback):
    def __init__(self, model, Xa, ya, wa, Xc, mc, X, y):
        self.model = model
        self.Xa = Xa
        self.ya = ya
        self.wa = wa
        self.Xc = Xc
        self.mc = mc
        self.X = X
        self.y = y
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        self.history += show_metrics(self.model, self.Xa, self.ya, self.wa,
                                     self.Xc, self.mc, self.X, self.y)

    def on_train_end(self, logs={}):
        self.history += show_metrics(self.model, self.Xa, self.ya, self.wa,
                                     self.Xc, self.mc, self.X, self.y)


def fit_model(model, X, y, domain_prediction, Xa, ya, wa, Xc, mc, X_original, y_original,
              validation_split=0., epoch_count=75, batch_size=64, verbose=0, callbacks=[]):
    model.fit([X], [y, domain_prediction],
              nb_epoch=epoch_count, batch_size=batch_size,
              validation_split=validation_split, verbose=verbose,
              callbacks=callbacks)
    f = model.get_layer('feature_extractor')
    l = model.get_layer('label_classifier')
    d = model.get_layer('domain_classifier')
    return f, l, d


def predict_model(model, X):
    return model.predict([X], batch_size=256, verbose=0)[0]


def predict_probs_model(model, X):
    return model.predict([X], batch_size=256, verbose=0)[0][:, 1]
