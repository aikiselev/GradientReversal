#!/usr/bin/env python2
from random import seed

import numpy as np

import data
import model

np.random.seed(1337)  # TODO repeatability
seed(1337)

X, y, _, _, _ = data.load("~/Documents/flavours-of-physics-start/input/training.csv",
                          shuffle=True)
Xa, ya, wa, _, _ = data.load("~/Documents/flavours-of-physics-start/input/check_agreement.csv",
                             shuffle=False, weight=True)
Xc, _, _, mc, _ = data.load("~/Documents/flavours-of-physics-start/input/check_correlation.csv",
                            shuffle=False, mass=True, test=True)
print "Loaded training data"

X, scaler = data.preprocess_data(X)
Xa, _ = data.preprocess_data(Xa)
Xc, _ = data.preprocess_data(Xc)

print "Preprocessed data"

X_test, _, _, _, ids = data.load("~/Documents/flavours-of-physics-start/input/test.csv", test=True,
                                 ids=True)
X_test = scaler.transform(X_test)

print "Loaded test data"

print "Creating models:"
n_models = 1
n_epochs = 50
probs = None
for i in range(n_models):
    # l = choice(xrange(1, 50))
    l = 1
    print "Model {}; l = {}".format(i, l)
    m = model.create_model(X.shape[1], l, 1)
    # model.fit_model_channels(m, X, y, Xa, ya, wa, Xc, mc)
    model.fit_model_mc(m, X, y, Xa, ya, wa, Xc, mc,
                       epoch_count=n_epochs, batch_size=64, validation_split=0.05)
    p = model.predict_model(m, np.array(X_test))
    probs = p if probs is None else p + probs
probs /= n_models

# print "Fitted models"
# random_classifier = np.random.rand(len(probs))
# q = 0.98
# combined_probs = q * (probs ** 30) + (1 - q) * random_classifier

# prediction = data.decorrelate(prediction, 1)

data.save_submission(ids, probs, "grl_prediction.csv")

print "Made predictions"


