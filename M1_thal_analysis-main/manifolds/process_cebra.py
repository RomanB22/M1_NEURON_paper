from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cebra
import time as tm


def create_model(train_mode, isHybrid=False):
    import __main__
    mp = __main__.metaParams

    cond = 'time' if train_mode == 'time' else mp.val('cond', 'time_delta')

    return cebra.CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=mp.val('lrate', 3e-4),
                        temperature_mode='auto' if mp.val('temp_auto', False) else 'constant',
                        temperature=1,
                        output_dimension=mp.val('ndim'),
                        max_iterations=mp.val('iter'),
                        distance='cosine',
                        conditional=cond,
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        delta=0.1,
                        hybrid=isHybrid)


def load_or_train_model(dir, train_mode, neural_data, labels, isShuffled=False, suffix=None):

    # if train_mode is 'time' (no aux data), shuffling make no sense, so just load the non-shuffled model
    if train_mode == 'time' and isShuffled:
        isShuffled = False

    dir_path = Path(dir) / 'models'
    sh = '_shuffled' if isShuffled else ''
    if suffix is None:
        suffix = ''
    file_path = dir_path / f"model_{train_mode}{suffix}{sh}.pt"
    try:
        model = cebra.CEBRA.load(file_path)
        print(f"Loaded model from {file_path}")
        loaded = True
    except FileNotFoundError:
        print(f"No trained model found for {file_path}.\nTraining the model...")
        loaded = False

        # depending on training mode, select the correct labels
        if train_mode == 'time':
            label = None
        elif train_mode == 'stg':
            label = labels[:,0]
        elif train_mode == 'joy':
            label = labels[:,1:3]
        elif train_mode == 'stg-joy':
            label = labels[:,0:3]
        elif train_mode == 'thal':
            label = labels[:,3:]
        elif train_mode == 'joy-thal':
            label = labels[:,1:]

        model = create_model(train_mode)
        if isShuffled and (train_mode != 'time'):
            label = np.random.permutation(label) # shuffle along the first dimension
        start_time = tm.time()

        # fit the model
        if label is not None: # hypothesis-driven mode
            model.fit(neural_data, label)
        else: # discovery-driven mode
            model.fit(neural_data)

        end_time = tm.time()
        print(f"Model training time: {end_time - start_time} seconds")

        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        model.save(file_path)

    return model, loaded

def plot_embedding(embedding, neural_data, aux):

    # Visualization
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] # R -> G -> B
    if len(np.unique(aux[:,0])) == 4:
        colors += [(1, 1, 0)] # R -> G -> B -> Y
    cmap_name = 'cebra'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

    # cebra_spikes = cebra_model_A.transform(neural_data.T)
    # fig1 = plt.figure(figsize=(15,6))

    numplots = 1 #len(neural_data)
    fig, axs = plt.subplots(1, numplots, figsize=(5 * numplots, 6), subplot_kw={'projection': '3d'})

    # ax1.scatter(cebra_spikes[:, 0], cebra_spikes[:, 1], cebra_spikes[:, 2], c='gray', marker='.')
    ax = cebra.plot_embedding(embedding, embedding_labels=aux[:,0], ax=axs, title="Embedding", cmap=cmap)
    return ax

def plot_losses(models, labels=None, colors=None, ax=None, save_as=None):
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = [f'Model {i}' for i in range(len(models))]
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, model in enumerate(models):
        is_shuffled = '_sh' in labels[i]
        color_idx = i if not is_shuffled else i % int(len(models)/2) # use the same color for shuffled and non-shuffled of same train mode
        ax = cebra.plot_loss(model, ax=ax, label=None if is_shuffled else labels[i], color=colors[color_idx], alpha=0.75, linestyle=':' if is_shuffled else '-')
    plt.legend()

    if save_as is not None:
        plt.savefig(save_as)

    return ax


def plot_sessions_data(neural_data, auxiliary, session_labels, output_dir):
    # Calculate mean activity
    mean_activity = [nd.mean() for nd in neural_data]

    # Calculate number of neurons
    num_neurons = [nd.shape[1] for nd in neural_data]

    # Create a figure and axis with adjusted size
    fig, ax1 = plt.subplots(figsize=(10, 8))  # Increase height to 8

    # Plot mean activity
    ax1.scatter(np.arange(len(mean_activity)), mean_activity, marker='o', linestyle='-', color='b', label='Mean Activity')
    ax1.set_title('Mean Activity and Number of Neurons Across Sessions')
    ax1.set_xlabel('Session Index')
    ax1.set_ylabel('Mean Activity')
    ax1.grid(True)

    # Set session labels as x-ticks
    ax1.set_xticks(np.arange(len(session_labels)))
    ax1.set_xticklabels(session_labels, rotation=90, fontsize='small')

    # Adjust the layout to make space for x-tick labels
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust bottom margin

    # Calculate mean and standard deviation
    mean = np.mean(mean_activity)
    std = np.std(mean_activity)

    # Add horizontal lines for mean ± std
    ax1.axhline(y=mean + std, color='b', linestyle='--', label='mean + std')
    ax1.axhline(y=mean - std, color='b', linestyle='--', label='mean - std')

    # Create a second y-axis for the number of neurons
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(num_neurons)), num_neurons, color='none', edgecolor='g', label='Number of Neurons')
    ax2.set_ylabel('Number of Neurons')

    # Calculate mean and standard deviation for number of neurons
    mean_neurons = np.mean(num_neurons)
    std_neurons = np.std(num_neurons)

    # Add horizontal lines for mean ± std for number of neurons
    ax2.axhline(y=mean_neurons + std_neurons, color='g', linestyle='--', label='mean + std (neurons)')
    ax2.axhline(y=mean_neurons - std_neurons, color='g', linestyle='--', label='mean - std (neurons)')
    med = np.median(num_neurons)
    ax2.axhline(y=med, color='g', linestyle='-', label=f'median ({med})')

    # Find points within ±1 std of mean
    mask = (mean_activity >= mean - std) & (mean_activity <= mean + std) & (num_neurons >= mean_neurons - std_neurons) & (num_neurons <= mean_neurons + std_neurons)
    ax1.plot(np.where(mask)[0], np.array(mean_activity)[mask], 
             'ro', markersize=12, fillstyle='none', label='Within ±1 std')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'{output_dir}/sessions_data.png')


def process(gen_dir, train_mode, neural_data, labels, isShuffled=False, train_split_ratio=None):

    suffix = f'_tr{train_split_ratio}' if train_split_ratio is not None else None

    model, was_loaded = load_or_train_model(gen_dir, train_mode, neural_data, labels, isShuffled, suffix)

    embedding = model.transform(neural_data)

    if not was_loaded: # otherwise it's already plotted
        ax = plot_embedding(embedding, None, labels)
        sh = '_sh' if isShuffled else ''
        plt.savefig(f'{gen_dir}/models/embedding_{train_mode}{suffix}{sh}.png')

    return model, embedding


def check_consistency(model, neural_data, auxiliary):
    embedding = model.transform(neural_data)
    cebra.sklearn.metrics.consistency_score(embedding, embedding_labels=auxiliary[:,0])


if __name__ == '__main__':
    import __main__

    selected_sessions = '230517_2759_1606VAL'
    output_base_dir = Path('output/manifolds/cebra/')

    from utils import MetaParams
    pmp = MetaParams('neural_data', {
        'region': ('m1', 'region to process as neural data', False),
        'rec_type': ('spikes', 'type of data to process (spikes or lfp)', False),
        'smooth': (True, 'smooth spikes'),
        # To use with training mode 'thal':
        # 'use_thal_as': ('th-cebra', 'preprocessing method for thalamus as auxiliary data', False), # e.g. 'th-cebra' or 'th-pca3' (PCA to 3 dimensions)
    })
    __main__.preprMetaParams = pmp
    preprocess_label = pmp.full_label()

    output_base_dir = output_base_dir / preprocess_label

    # Try to load preprocessed data
    prepr_data_fname = output_base_dir / 'all_data.pkl'

    from manifolds.prepr import load_preprocessed_joint_data, preprocess_joint_data
    if prepr_data_fname.exists():
        neural_data, auxiliary, session_labels = load_preprocessed_joint_data(prepr_data_fname)
        print(f"Loaded data from {prepr_data_fname}")
    else:
        print(f"Data not found at {prepr_data_fname}. Preprocessing...")
        # Root folder containing sessions as subfolders
        raw_data_dir = Path('./data')
        all_data = preprocess_joint_data(raw_data_dir, prepr_data_fname, pmp.val('smooth', False))

        neural_data = all_data['neural']
        auxiliary = all_data['auxiliary'] 
        session_labels = all_data['session_labels']

    plot_sessions_data(neural_data, auxiliary, session_labels, output_base_dir)

    mp = MetaParams('decode', {
        'iter': (10000, 'cebra max iterations'),
        'ndim': (3, 'cebra output dimension'),
        # 'hybr': (False, 'use isHybrid for model'),
        'cond': ('delta', 'conditional for contrastive learning'),
    })
    __main__.metaParams = mp
    gen_label = mp.full_label()

    sess_output_dir = output_base_dir / selected_sessions
    gen_dir = sess_output_dir / gen_label

    # build example embedding
    if type(selected_sessions) not in [list, tuple]:
        selected_sessions = [selected_sessions]

    for sess_id in selected_sessions:
        sess_idx = session_labels.index(sess_id)
        neural = neural_data[sess_idx]
        aux = auxiliary[sess_idx]
        process(gen_dir, 'stg-joy', neural, aux, isShuffled=False)

