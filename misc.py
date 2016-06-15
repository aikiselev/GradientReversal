class SetterProperty(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)


def predict(m, x):
    if m.classifier_type == 'classic':
        return m.predict([x], batch_size=256, verbose=0)
    else:
        return m.predict([x], batch_size=256, verbose=0)[0]


def predict_probs(m, x):
    if m.classifier_type == 'classic':
        return m.predict([x], batch_size=256, verbose=0)[:, 1]
    else:
        return m.predict([x], batch_size=256, verbose=0)[0][:, 1]
