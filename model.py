import theano.gof
from keras.callbacks import Callback
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Activation, Dropout, MaskedLayer
from keras.models import Graph
from keras.utils import np_utils

from evaluation import *


class ReverseGradient(theano.gof.Op):
    # custom Theano operaion for gradient reversal layer

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


class GradientReversal(MaskedLayer):
    def __init__(self, l, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.op = ReverseGradient(l)  # TODO make lambda a parameter

    def get_output(self, train=False):
        return self.op(self.get_input(train))


def build_model(input_size, prediction_layers, domain_layers, connection_position,
                l, domain_loss_weight):
    model = Graph()
    model.add_input(name='input', input_shape=(input_size,))

    counts = [0, 0, 0]
    last_node_name = 'input'

    def connect(last_node_name, node):
        if node['type'] == 'Dense':
            layer_name = 'dense{}'.format(counts[0])
            model.add_node(Dense(node['size']), name=layer_name, input=last_node_name)
            counts[0] += 1
        if node['type'] == 'Dropout':
            layer_name = 'drop{}'.format(counts[1])
            model.add_node(Dropout(node['p']), name=layer_name, input=last_node_name)
            counts[1] += 1
        if node['type'] == 'PReLU':
            layer_name = 'prelu{}'.format(counts[2])
            model.add_node(PReLU(), name=layer_name, input=last_node_name)
            counts[2] += 1
        return layer_name

    for node in prediction_layers:
        last_node_name = connect(last_node_name, node)
    model.add_node(Activation('softmax'), name='softmax', input=last_node_name)
    model.add_output(name='output', input='softmax')

    model.add_node(GradientReversal(l), name='grl', input='dense{}'.format(connection_position))
    last_node_name = 'grl'

    for node in domain_layers:
        last_node_name = connect(last_node_name, node)
    model.add_node(Activation('softmax'), name='softmax_domain', input=last_node_name)
    model.add_output(name='domain', input='softmax_domain')

    model.compile(loss={'output': 'categorical_crossentropy',
                        'domain': 'categorical_crossentropy'},
                  loss_weights={
                      'output': 1,
                      'domain': domain_loss_weight  # lambda parameter
                  },
                  optimizer='rmsprop')
    return model


def dense(size):
    return {'type': 'Dense', 'size': size}
def dropout(p):
    return {'type': 'Dropout', 'p': p}
def prelu():
    return {'type': 'PReLU'}


def show_metrics(model, Xa, ya, wa, Xc, mc):
    pa = predict_model(model, Xa)
    ks = compute_ks(pa[ya == 0], pa[ya == 1], wa[ya == 0], wa[ya == 1])

    pc = predict_model(model, Xc)
    cvm = compute_cvm(pc, mc)
    print "KS: {} : 0.09 / CvM: {} : 0.002".format(ks, cvm)
    return ks, cvm


class ShowMetrics(Callback):
    def __init__(self, model, Xa, ya, wa, Xc, mc):
        self.model = model
        self.Xa = Xa
        self.ya = ya
        self.wa = wa
        self.Xc = Xc
        self.mc = mc
        self.ks = 0
        self.cvm = 0

    def on_epoch_end(self, epoch, logs={}):
        self.ks, self.cvm = show_metrics(self.model, self.Xa, self.ya, self.wa, self.Xc, self.mc)

    def on_train_end(self, logs={}):
        self.ks, self.cvm = show_metrics(self.model, self.Xa, self.ya, self.wa, self.Xc, self.mc)


def fit_model(model, X, y, Xa, ya, wa, Xc, mc, validation_split=0.,
              epoch_count=75, batch_size=64, verbose=0):
    y = np_utils.to_categorical(y)
    domain_prediction = y
    model.fit({'input': X, 'output': y, 'domain': domain_prediction},
              nb_epoch=epoch_count, batch_size=batch_size,
              validation_split=validation_split, verbose=verbose,
              callbacks=[ShowMetrics(model, Xa, ya, wa, Xc, mc)])
    return model


def predict_model(model, X):
    return model.predict({'input': X}, batch_size=256, verbose=0)['output'][:, 1]
