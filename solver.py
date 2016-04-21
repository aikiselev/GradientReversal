import numpy as np

import data
import model
from model import build_sequential_model, create_clf

#####
X, y, _, _, _ = data.load("~/Documents/flavours-of-physics-start/input/training.csv",
                          shuffle=True)
Xa, ya, wa, _, _ = data.load("~/Documents/flavours-of-physics-start/input/check_agreement.csv",
                             shuffle=False, weight=True)
Xa_train, ya_train, _, _, _ = data.load(
    "~/Documents/flavours-of-physics-start/input/check_agreement.csv",
    shuffle=True)
Xc, _, _, mc, _ = data.load("~/Documents/flavours-of-physics-start/input/check_correlation.csv",
                            shuffle=False, mass=True, test=True)
X, scaler = data.preprocess_data(X)
Xa, _ = data.preprocess_data(Xa, scaler)
Xa_train, _ = data.preprocess_data(Xa_train, scaler)
Xc, _ = data.preprocess_data(Xc, scaler)
X_test, _, _, _, ids = data.load("~/Documents/flavours-of-physics-start/input/test.csv", test=True,
                                 ids=True)
X_test, _ = data.preprocess_data(X_test, scaler)
print("Done")
#####
from keras.layers import PReLU, Dropout, Dense


def feature_extractor(input_size, output_size):
    return build_sequential_model([Dense(150, input_dim=input_size),
                                   PReLU(), Dropout(0.4), Dense(140),
                                   PReLU(), Dropout(0.35), Dense(130),
                                   PReLU(), Dropout(0.3), Dense(120),
                                   PReLU(), Dropout(0.27), Dense(output_size)],
                                  name="feature_extractor")


def label_classifier(input_size, name="label_classifier"):
    return build_sequential_model([Dense(100, input_dim=input_size),
                                   PReLU(), Dropout(0.3), Dense(80),
                                   PReLU(), Dropout(0.25), Dense(70),
                                   PReLU(), Dropout(0.2), Dense(2, activation='softmax')],
                                  name=name)


def domain_classifier(input_size, l):
    clf = label_classifier(input_size, name="domain_classifier")
    return attach_grl(clf, l)


#####
n_models = 1
n_epochs = 5
probs = None
for i in range(n_models):
    from keras.utils import np_utils

    domain_weight = 1
    np.random.seed(42)  # repeatability

    #     print("Model {}; l = {}; domain_weight = {}".format(i, lam, domain_weight))
    n_extracted_features = 60
    f = feature_extractor(X.shape[1], n_extracted_features)
    l = label_classifier(n_extracted_features)
    d = label_classifier(n_extracted_features, name="domain_classifier")

    transfering_ratio = 0.7  # 0.1
    # Learning on train

    m = create_clf(f, l, d, 0)
    f, l, d = model.fit_model(m, X, np_utils.to_categorical(y), np_utils.to_categorical(y), Xa, ya,
                              wa, Xc, mc, X, y,
                              epoch_count=int((1 - transfering_ratio) * n_epochs),
                              batch_size=128, validation_split=0.05, verbose=2, show_metrics=True)
    # Transfering to check_agreement
    ya_output = model.predict_model(m, np.array(Xa_train))
    steps = 20
    for step in range(steps):
        lam = np.linspace(0.2, 10, steps)[step]  # np.random.choice(np.linspace(1, 10, 10))
        print('lambda = ', lam)
        m = create_clf(f, l, d, lam)
        f, l, d = model.fit_model(m, Xa_train, ya_output, np_utils.to_categorical(ya_train), Xa, ya,
                                  wa, Xc, mc, X, y,
                                  epoch_count=int(transfering_ratio * n_epochs / steps),
                                  batch_size=512, validation_split=0.5, verbose=2,
                                  show_metrics=True)

    # Output
    p = model.predict_probs_model(m, np.array(X_test))
    probs = p if probs is None else p + probs
probs /= n_models
#####
data.save_submission(ids, probs, "grl_prediction.csv")
