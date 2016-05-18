import theano.gof
from keras.engine.topology import Layer


class ReverseGradient(theano.gof.Op):
    # Custom Theano operaion for gradient reversal layer
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage, params=None):
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
