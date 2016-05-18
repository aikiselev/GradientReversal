from keras.layers import Input
from keras.models import Model, Sequential

from grl_layer import GradientReversal
from misc import SetterProperty


class GRL_classifier():
    def __init__(self, _feature_extractor, _label_classifier, _domain_classifier, _lam):
        self.feature_extractor = _feature_extractor
        self.label_classifier = _label_classifier
        self.domain_classifier = _domain_classifier
        self.model = None
        self.__dict__['lam'] = _lam
        self.build()

    def build(self):
        model_input = Input(shape=(self.feature_extractor.input_shape[1],))
        features = self.feature_extractor(model_input)

        label_class = self.label_classifier(features)

        grl = GradientReversal(self.lam, input_shape=(self.domain_classifier.input_shape[1],))
        domain_class = self.domain_classifier(grl(features))

        self.model = Model(input=[model_input], output=[label_class, domain_class])
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                           loss_weights=[1, 1], metrics=['accuracy', 'accuracy'],
                           optimizer='rmsprop')

    def rebuild(self):
        self.feature_extractor = self.model.get_layer('feature_extractor')
        self.label_classifier = self.model.get_layer('label_classifier')
        self.domain_classifier = self.model.get_layer('domain_classifier')
        self.build()

    def fit(self, x, y, domain_prediction, validation_split=0., epoch_count=75,
            batch_size=64, verbose=0, callbacks=[]):
        self.model.fit([x], [y, domain_prediction], nb_epoch=epoch_count, batch_size=batch_size,
                       validation_split=validation_split, verbose=verbose, callbacks=callbacks)

    @SetterProperty
    def lam(self, value):
        self.__dict__['lam'] = value
        self.rebuild()

    def predict(self, x):
        return self.model.predict([x], batch_size=256, verbose=0)[0]

    def predict_probs(self, x):
        return self.model.predict([x], batch_size=256, verbose=0)[0][:, 1]


def build_sequential_model(hidden_layers, name=""):
    m = Sequential(name=name)
    for layer in hidden_layers:
        m.add(layer)
    return m
