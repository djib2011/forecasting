from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import tensorflow as tf

import metrics

np.seterr(all='ignore')

def get_last_N(series, N=18):
    ser_N = series.dropna().iloc[-N:].values
    if len(ser_N) < N:
        pad = [ser_N[0]] * (18 - len(ser_N))
        ser_N = np.r_[pad, ser_N]
    return ser_N


def get_predictions(model, X):
    preds = []

    for i in range(len(X) // 256):
        x = X[i * 256:(i + 1) * 256]

        mn, mx = x.min(axis=1).reshape(-1, 1), x.max(axis=1).reshape(-1, 1)
        x_sc = (x - mn) / (mx - mn)
        pred = model(x_sc[..., np.newaxis])
        preds.append(pred[..., 0] * (mx - mn) + mn)

    x = X[(i + 1) * 256:]
    mn, mx = x.min(axis=1).reshape(-1, 1), x.max(axis=1).reshape(-1, 1)

    x_sc = (x - mn) / (mx - mn)
    pred = model(x_sc[..., np.newaxis])
    preds.append(pred[..., 0] * (mx - mn) + mn)

    return np.vstack(preds)


def create_results_df(results, ensemble=False):
    new_keys = [k for k in results['smape'].keys() if not k.isdigit()]
    columns = ['input_len', 'output_len', 'loss', 'bottleneck_size',
               'bottleneck_activation', 'LSTM_type', 'num']
    if ensemble:
        columns.pop()

    df = pd.DataFrame([k.split('__') for k in new_keys],
                      columns=columns)

    for column in ('input_len', 'output_len', 'loss',
                   'bottleneck_size', 'bottleneck_activation',
                   'LSTM_type'):
        df[column] = df[column].apply(lambda x: x.split('_')[1])

    df['smape'] = [results['smape'][k][0] if results['smape'][k] else np.nan for k in new_keys]
    df['mase*'] = [results['mase'][k][0] if results['mase'][k] else np.nan for k in new_keys]

    return df


def evaluate_models(trials, x, y):

    trial_names = [t.name for t in trials]

    results = {'smape': {k: [] for k in trial_names},
               'mase': {k: [] for k in trial_names}}

    # Evaluate all models
    for trial in tqdm(trials):
        model_dir = str(trial / 'best_weights.h5')

        smape = metrics.build_smape(overlap=6)
        mase_estimate = metrics.build_mase(overlap=6)
        owa_estimate = metrics.build_owa(overlap=6)
        reconstruction_loss = metrics.build_reconstruction_loss(overlap=6)

        model = tf.keras.models.load_model(model_dir, custom_objects={'SMAPE': smape,
                                                                      'MASE_estimate': mase_estimate,
                                                                      'OWA_estimate': owa_estimate,
                                                                      'reconstruction_loss': reconstruction_loss})

        preds = get_predictions(model, x)

        tf.keras.backend.clear_session()

        results['smape'][trial.name].append(np.nanmean(metrics.SMAPE(y, preds[:, 6:])))
        results['mase'][trial.name].append(np.nanmean(metrics.MASE(x, y, preds[:, 6:])))

    return results

# Read test data
train_path = Path('data/Yearly-train.csv')
test_path = Path('data/Yearly-test.csv')

train = pd.read_csv(train_path).drop('V1', axis=1)
test = pd.read_csv(test_path).drop('V1', axis=1)

# Read experiments
p = Path('results').absolute()

trials = list(p.glob('*'))
trial_names = [t.name for t in trials]


num_inputs = np.unique([t[4:6] for t in trial_names if not t.isdigit()])

for inp in num_inputs:
    X_test = np.array([get_last_N(ser[1], N=int(inp)) for ser in train.iterrows()])
    y_test = test.values

    results = evaluate_models(trials, X_test, y_test)

    try:
        pd.concat([df, create_results_df(results)])
    except NameError:
        df = create_results_df(results)

# df.to_csv('reports/result_df.csv', index=False)
