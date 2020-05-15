from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import tensorflow as tf
import sys
import os

import metrics

target_dir = Path('reports')

df = None
ensemble = False

if (target_dir / 'result_df.csv').exists():
    df = pd.read_csv(str(target_dir / 'result_df.csv'))

if len(sys.argv) > 1:
    if sys.argv[1] == '--ensemble':
        ensemble=True

if len(sys.argv) > 2:
    if sys.argv[2] == '--fresh':
        df = None

np.seterr(all='ignore')


def check_for_errors(trials, fix=True):
    if fix:
        errors = [trial for trial in trials if not (trial / 'best_weights.h5').exists()]
        for error in errors:
            os.rmdir(str(error))
    return [trial for trial in trials if (trial / 'best_weights.h5').exists()]


def get_last_N(series, N=18):
    ser_N = series.dropna().iloc[-N:].values
    if len(ser_N) < N:
        pad = [ser_N[0]] * (N - len(ser_N))
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

        results['smape'][trial.name].append(np.nanmean(metrics.SMAPE(y, preds[:, -6:])))
        results['mase'][trial.name].append(np.nanmean(metrics.MASE(x, y, preds[:, -6:])))

    return results


def evaluate_model_ensembles(families, x, y):

    family_names = [f.name for f in families]

    results = {'smape': {k: [] for k in family_names + [f.name + str(num) for f in families for num in range(10)]},
               'mase': {k: [] for k in family_names + [f.name + str(num) for f in families for num in range(10)]}}

    print('entered')
    # Evaluate all models
    for family in families:

        print('family', str(family))

        family_preds = []

        for num in range(10):

            print('num:', num)

            trial = str(family) + str(num)
            model_dir = trial + '/best_weights.h5'

            smape = metrics.build_smape(overlap=6)
            mase_estimate = metrics.build_mase(overlap=6)
            owa_estimate = metrics.build_owa(overlap=6)
            reconstruction_loss = metrics.build_reconstruction_loss(overlap=6)

            model = tf.keras.models.load_model(model_dir, custom_objects={'SMAPE': smape,
                                                                          'MASE_estimate': mase_estimate,
                                                                          'OWA_estimate': owa_estimate,
                                                                          'reconstruction_loss': reconstruction_loss})

            preds = get_predictions(model, x)
            family_preds.append(preds)

            tf.keras.backend.clear_session()

            results['smape'][Path(trial).name].append(np.nanmean(metrics.SMAPE(y, preds[:, -6:])))
            results['mase'][Path(trial).name].append(np.nanmean(metrics.MASE(x, y, preds[:, -6:])))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name].append(np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:])))
        results['mase'][family.name].append(np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:])))

        return results


# Read test data
train_path = Path('data/Yearly-train.csv')
test_path = Path('data/Yearly-test.csv')

train = pd.read_csv(train_path).drop('V1', axis=1)
test = pd.read_csv(test_path).drop('V1', axis=1)

# Read experiments
p = Path('results').absolute()

if (target_dir / 'tracked.pkl').exists():
    with open(str(target_dir / 'tracked.pkl'), 'rb') as f:
        tracked_trials = pkl.load(f)
else:
    tracked_trials = []

trials = [t for t in p.glob('*') if t not in tracked_trials]
trials = check_for_errors(trials, fix=False)

num_inputs = np.unique([t.name[4:6] for t in trials if not t.name.isdigit()])

tracked_trials.extend(trials)

for inp in num_inputs:
    X_test = np.array([get_last_N(ser[1], N=int(inp)) for ser in train.iterrows()])
    y_test = test.values

    curr_trial_list = [t for t in trials if t.name[4:6] == inp]

    families = list(set([Path(str(c)[:-1]) for c in curr_trial_list]))

    results = evaluate_model_ensembles(families, X_test, y_test)

    if isinstance(df, pd.DataFrame):
        df = pd.concat([df, create_results_df(results)])
    else:
        df = create_results_df(results)

df.to_csv(str(target_dir / 'result_df.csv'), index=False)

with open(str(target_dir / 'tracked.pkl'), 'wb') as f:
    pkl.dump(tracked_trials, f)
