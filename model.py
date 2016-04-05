import numpy as np
import theano.gof
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


def create_model(input_size, l=0.5, domain_loss_weight=0.2):

    model = build_model(input_size,
                        [dense(input_size), dropout(0.2),
                         dense(300), prelu(), dropout(0.1),
                         dense(200),
                         dense(2)],
                        [dense(150), prelu(), dropout(0.05),
                         dense(100), dropout(0.01),
                         dense(2)], 0, l, domain_loss_weight=domain_loss_weight)
    return model


def fit_model_channels(model, X, y, Xa, ya, wa, Xc, mc, epoch_count_train=50, epoch_count_adapt=5):
    y = np_utils.to_categorical(y)

    for _ in xrange(30):
        model.fit({'input': X,
                   'output': y,
                   'domain': np.zeros_like(y)},
                  nb_epoch=1, verbose=0,
                  validation_split=0)
        show_metrics(model, Xa, ya, wa, Xc, mc)
        agreement_prediction = model.predict({'input': Xa})['output']
        model.fit({'input': Xa,
                   'output': agreement_prediction,
                   'domain': np.ones_like(agreement_prediction)},
                  nb_epoch=1, verbose=0)
        show_metrics(model, Xa, ya, wa, Xc, mc)
    return model


def show_metrics(model, Xa, ya, wa, Xc, mc):
    pa = predict_model(model, Xa)
    ks = compute_ks(pa[ya == 0], pa[ya == 1], wa[ya == 0], wa[ya == 1])

    pc = predict_model(model, Xc)
    cvm = compute_cvm(pc, mc)
    print "KS: {} : 0.09 / CvM: {} : 0.002".format(ks, cvm)
    return ks, cvm


def fit_model_mc(model, X, y, Xa, ya, wa, Xc, mc, validation_split=0.,
                 epoch_count=75, batch_size=64):
    #
    # domain_mc = np.where(y == 1, 0, 1)
    # domain_real = np.where(y == 1, 1, 0)
    # domain_prediction = np.dstack((domain_real, domain_mc))[0]

    y = np_utils.to_categorical(y)
    domain_prediction = y
    for _ in xrange(epoch_count):
        model.fit({'input': X,
               'output': y,
               'domain': domain_prediction},
                  nb_epoch=1, batch_size=batch_size,
                  validation_split=validation_split, verbose=2)
        show_metrics(model, Xa, ya, wa, Xc, mc)
    return model


def predict_model(model, X):
    return model.predict({'input': X}, batch_size=256, verbose=0)['output'][:, 1]
