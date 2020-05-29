from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import tensorflow as tf
import sys
import argparse
import os

import metrics

def check_for_errors(trials, fix=True):
    if fix:
        errors = [trial for trial in trials if not (trial / 'best_weights.h5').exists()]
        for error in errors:
            os.rmdir(str(error))
    return [trial for trial in trials if (trial / 'best_weights.h5').exists()]


def check_families_for_errors(families, num_models):
    return [(family, num) for family, num in zip(families, num_models) if all([(Path(str(family) + '__' + str(n)) / 'best_weights.h5').exists() for n in range(num)])]


def filter_tracked(candidate_names, tracked_dict):
    candidate_families = list(set([Path('__'.join(str(t).split('__')[:-1])) for t in candidate_names]))
    candidate_models = [len(list(p.glob(f.name + '*'))) for f in candidate_families]

    untracked_families, untracked_nums = [], []

    for i in range(len(candidate_families)):
        if candidate_families[i] in tracked_dict.keys():
            if candidate_models[i] == tracked_dict[candidate_families[i]]:
                continue
        untracked_families.append(candidate_families[i])
        untracked_nums.append(candidate_models[i])

    filtered = check_families_for_errors(untracked_families, untracked_nums)
    if filtered:
        untracked_families, untracked_nums = zip(*filtered)
        return untracked_families, untracked_nums
    else:
        return [], []


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

    df['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in new_keys]
    df['mase*'] = [results['mase'][k] if results['mase'][k] else np.nan for k in new_keys]

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

    results = {'smape': {}, 'mase': {}}

    # Evaluate all models
    for family, num_models in tqdm(families):

        family_preds = []

        for num in range(num_models):

            trial = str(family) + '__' + str(num)
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

            results['smape'][Path(trial).name] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
            results['mase'][Path(trial).name] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
        results['mase'][family.name] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

    return results


def run(num_inputs, families, num_models, train_set, test_set, df):
    for inp in num_inputs:
        X_test = np.array([get_last_N(ser[1], N=int(inp)) for ser in train_set.iterrows()])
        y_test = test_set.values

        curr_family_list = [(f, m) for f, m in zip(families, num_models) if f.name[4:6] == inp]

        results = evaluate_model_ensembles(curr_family_list, X_test, y_test)

        if isinstance(df, pd.DataFrame):
            df = pd.concat([df, create_results_df(results)])
        else:
            df = create_results_df(results)

    return df


if __name__ == '__main__':
    target_dir = Path('reports')

    df = None

    if (target_dir / 'result_df.csv').exists():
        df = pd.read_csv(str(target_dir / 'result_df.csv'))

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fresh', action='store_true', help='Ignore tracked experiments; Evaluate all models from '
                                                                 'scratch.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print which experiments will be evaluated (i.e. don\'t run actual evaluations.')
    args = parser.parse_args()

    if args.fresh:
        df = None

    np.seterr(all='ignore')

    # Read test data
    train_path = Path('data/Yearly-train.csv')
    test_path = Path('data/Yearly-test.csv')

    train = pd.read_csv(train_path).drop('V1', axis=1)
    test = pd.read_csv(test_path).drop('V1', axis=1)

    # Read experiments
    p = Path('results').absolute()

    if (target_dir / 'tracked.pkl').exists() and not args.fresh:
        with open(str(target_dir / 'tracked.pkl'), 'rb') as f:
            tracked = pkl.load(f)
    else:
        tracked = {}

    trials = list(p.glob('*'))
    trials = check_for_errors(trials, fix=False)
    trials = [t for t in trials if not t.name.startswith('line')]

    families, num_models = filter_tracked(trials, tracked)

    num_inputs = np.unique([f.name[4:6] for f in families if not f.name.isdigit()])

    if args.debug:
        print('Individual tracked trials:    ', sum(tracked.values()))
        print('Tracked trial families:       ', len(tracked))
        print('Individual trials found:      ', len(trials))
        print('Individual untracked trials:  ', sum(num_models))
        print('Untracked trial families:     ', len(families))
        print()
        for f, m in zip(families, num_models):
            print('{:>3} @ {}'.format(m, f))

    else:
        df = run(num_inputs, families, num_models, train, test, df)

        df.to_csv(str(target_dir / 'result_df.csv'), index=False)

        tracked.update(dict(zip(families, num_models)))

        with open(str(target_dir / 'tracked.pkl'), 'wb') as f:
            pkl.dump((tracked), f)

        print('Done!')
