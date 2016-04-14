import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_features(df):
    # significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    df['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']) ** 2
    df['NEW_IP_dira'] = df['IP'] * df['dira']
    # Stepan Obraztsov's magic features
    df['NEW_FD_SUMP'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])
    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    # some more magic features
    df['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']
    df['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione',
                               'isolationf']].min(
        axis=1)
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT'] = df['FlightDistance'] / df['LifeTime']
    return df


def load(data_file, tail=None, weight=False, mass=False, shuffle=False, ids=False, test=False):
    data = pd.read_csv(data_file)
    # data = add_features(data) TODO add features
    if tail is not None:
        data = data[-tail:]

    # shuffle
    if shuffle:
        data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)

    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    # 'IP']  # , 'isolationc', 'SPDhits', 'IPSig']
    features = list(f for f in data.columns if f not in filter_out)
    X = data[features].values
    y = data['signal'].values if not test else None
    w = data['weight'].values if weight else None
    m = data['mass'].values if mass else None
    ids = data['id'].values if ids else None
    return X, y, w, m, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
    scaler.partial_fit(X)
    X = scaler.transform(X)
    return X, scaler


def save_submission(X_ids, prediction, file_name):
    submission = pd.DataFrame({"id": X_ids, "prediction": prediction})
    submission.to_csv(file_name, index=False)


def decorrelate(probs, n_models_to_combine):
    corr = np.corrcoef(probs)
    cr = corr.sum(axis=0)
    idx = np.argsort(cr)   # sorted correlations
    ids = idx[:n_models_to_combine]  # take first least corrlated
    ids.sort()             # h5py requires increasing index
    p = probs[list(ids)].mean(axis=0)
    return p