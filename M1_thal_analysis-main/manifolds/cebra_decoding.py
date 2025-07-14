import numpy as np
import matplotlib.pyplot as plt
import cebra
import cebra.datasets
import sklearn.neighbors
import sklearn.metrics
import pickle, sys, zipfile, os
import time as tm
from manifolds.process_cebra import process
import matplotlib.gridspec as gridspec

def split_data(neural, auxiliary, train_to_test_ratio):
    split_idx = int(len(neural)* train_to_test_ratio)
    neural_train = neural[:split_idx]
    neural_test = neural[split_idx:]
    label_train = auxiliary[:split_idx]
    label_test = auxiliary[split_idx:]
    return neural_train, neural_test, label_train, label_test

# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
def predict_labels(embedding_train, embedding_test, label_train, n_neighb=36):
   # stage_decoder = cebra.KNNDecoder(n_neighbors=n_neighb, metric="cosine")
   stage_decoder = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighb, metric="cosine")
   stage_decoder.fit(embedding_train, label_train[:,0])
   stage_pred = stage_decoder.predict(embedding_test) # stage_pred[label_test[:,0]==stage_pred] # 2720 1831

   jpos_decoder = cebra.KNNDecoder(n_neighbors=n_neighb, metric="cosine")
   jpos_decoder.fit(embedding_train, label_train[:,1:3])
   jpos_pred = jpos_decoder.predict(embedding_test)
   
   prediction = np.hstack([stage_pred.reshape(-1,1), jpos_pred])
   return prediction

def decode(train_modes, neural_train, neural_test, label_train, isShuffled=False, train_to_test_ratio=None):

    pred_labels = {}
    models = {}
    for train_mode in train_modes:

        model, embedding_train = process(gen_dir, train_mode, neural_train, label_train, isShuffled, train_to_test_ratio)

        embedding_train = model.transform(neural_train)
        embedding_test = model.transform(neural_test)

        predicted = predict_labels(embedding_train, embedding_test, label_train)
        pred_labels[train_mode] = predicted
        models[train_mode] = model
    return models, pred_labels


def plot_decoding_performance(models, predicted_labels, models_shuffled, predicted_labels_shuffled, observed_test_labels, training_modes):

    def get_stg_accuracy(predicted, observed):
        correct = len(predicted[predicted[:,0] == observed[:,0]])
        return correct / len(predicted)
    
    def get_median_error(predicted, observed):
        return np.median(np.abs(predicted[:,1:3] - observed[:,1:3]))
    
    def plot_joy_pos_error(ax, predicted, predicted_shuffled, observed, title, ylabel, is_joy_pos=True):
        # Define the mapping dictionary
        label_mapping = {
            'stg': 'Task stage',
            'joy': 'Joystick position',
            'stg-joy': 'Stage + position',
            'thal': 'Thal.',
            'joy-thal': 'Thal. + position'
        }

        errs, errs_sh = [], []
        for tr_mode in training_modes:
            err_func = get_median_error if is_joy_pos else get_stg_accuracy
            err = err_func(predicted[tr_mode], observed)
            err_sh = err_func(predicted_shuffled[tr_mode], observed)
            errs.append(err)
            errs_sh.append(err_sh)

        ax.set_title(title)
        bar_width = 0.25  # Width of each bar
        indices = np.arange(len(errs))  # The x locations for the groups

        ax.bar(indices, errs, bar_width, label='Actual', color='deepskyblue')
        ax.bar(indices + bar_width, errs_sh, bar_width, label='Shuffled', color='gray')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(indices + bar_width / 2)
        # Use the mapping dictionary to set the x-tick labels
        ax.set_xticklabels([label_mapping.get(mode, mode) for mode in training_modes], rotation=45)
        
        ax.set_ylabel(ylabel)
        ax.legend()

    def plot_joystick_trajectory(ax, predicted, observed, pred_shuffled, train_mode):
        predicted = predicted[train_mode]
        pred_shuffled = pred_shuffled[train_mode]

        dt = 0.002 * 4
        time = np.arange(0, len(predicted) * dt, dt)
        sl = slice(500, 2500)
        pred = predicted[sl, 1]
        pred_sh = pred_shuffled[sl, 1]
        klength = 8 # or 16
        kernel = np.ones(klength) / klength
        pred = np.convolve(pred, kernel, 'same')
        pred_sh = np.convolve(pred_sh, kernel, 'same')

        ax.plot(time[sl], pred_sh, label='shuffled', linewidth=0.5, color='gray')
        ax.plot(time[sl], observed[sl, 1], label='original')
        ax.plot(time[sl], pred, label='decoded')
        ax.legend(title=f'Train mode: {train_mode}')
        ax.set_xlabel('time [s]'), ax.set_ylabel('joystick x')

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[2, 3])

    ax_loss = plt.subplot(gs[0, 0])  # Upper left
    ax_reconstr = plt.subplot(gs[0, 1])  # Upper right
    ax_decode_joy = plt.subplot(gs[1, 0])  # Lower left
    ax_decode_stg = plt.subplot(gs[1, 1])  # Lower right


    from manifolds.process_cebra import plot_losses
    tr_modes_sh = [f'{mode}_sh' for mode in training_modes]
    plot_losses(list(models.values()) + list(models_shuffled.values()),
                labels=training_modes + tr_modes_sh,
                ax=ax_loss)
    # plot_reconstruction(models, ax=ax_reconstr)

    plot_joystick_trajectory(ax_reconstr, predicted_labels, observed_test_labels, predicted_labels_shuffled, plot_traj_train_mode)

    plot_joy_pos_error(ax_decode_joy, predicted_labels, predicted_labels_shuffled, observed_test_labels, 'Joystick pos. decoding', 'Median err.')
    plot_joy_pos_error(ax_decode_stg, predicted_labels, predicted_labels_shuffled, observed_test_labels, 'Task stage decoding', 'Accuracy', is_joy_pos=False)

    plt.savefig(f'{gen_dir}/decoding_performance_traj_{plot_traj_train_mode}.png')


if __name__ == "__main__":
    from pathlib import Path

    sess_id = '230517_2759_1606VAL'

    from utils import MetaParams

    pmp = MetaParams('neural_data', {
        'region': ('m1', 'region to process as neural data', False),
        'rec_type': ('spikes', 'type of data to process (spikes or lfp)', False),
        'smooth': (True, 'smooth spikes'),
        # 'use_thal_as': ('th-orig', 'preprocessing method for thalamus as auxiliary data', False),
    })
    preprocess_label = pmp.full_label()
    output_base_dir = Path('output') / 'manifolds' / 'cebra' / preprocess_label

    from manifolds.prepr import load_preprocessed_joint_data
    try:
        neural_data, auxiliary, session_labels = load_preprocessed_joint_data(output_base_dir / 'all_data.pkl')
    except FileNotFoundError:
        print(f"Data not found at {output_base_dir}. Run manifolds/process_cebra.py first.")
        exit()

    import __main__
    mp = MetaParams('decode', {
        'rs': (True, 'treat pre- and post- resting as the same'),
        'iter': (10000, 'cebra max iterations'),
        'ndim': (3, 'cebra output dimension'),
        # 'hybr': (False, 'use isHybrid for model'),
        'cond': ('delta', 'conditional for contrastive learning'),
        # 'lrate': (1e-4, 'learning rate'),
        # 'temp_auto': (True, 'learn cebra temperature'),
    })
    __main__.metaParams = mp

    gen_label = mp.full_label()
    gen_dir = output_base_dir / sess_id / gen_label

    os.makedirs(gen_dir, exist_ok=True)
    
    # save current script
    import sys, zipfile
    with zipfile.ZipFile(f'{gen_dir}/source.zip', 'w') as zip_file:
        zip_file.write(sys.argv[0])
        # zip_file.write('utils.py')

    print(f'Start decoding: {gen_dir}')

    sess_idx = session_labels.index(sess_id)
    neural = neural_data[sess_idx]
    aux = auxiliary[sess_idx]

    if mp.val('rs'):
        aux[aux[:,0] == 3,0] = 0 # treat pre- and post- resting as the same

    train_to_test_ratio = 0.8

    neural_train, neural_test, label_train, label_test = split_data(neural, aux, train_to_test_ratio)

    training_modes = ['stg', 'joy', 'stg-joy'] # ['time', 'stg', 'joy', 'stg-joy', 'thal', 'joy-thal']
    plot_traj_train_mode = 'stg-joy'
    models, predictions = decode(training_modes, neural_train, neural_test, label_train, False, train_to_test_ratio)
    
    # use shuffled labels now: 
    models_sh, predictions_sh = decode(training_modes, neural_train, neural_test, label_train, True, train_to_test_ratio)

    plot_decoding_performance(models, predictions, models_sh, predictions_sh, label_test, training_modes)


