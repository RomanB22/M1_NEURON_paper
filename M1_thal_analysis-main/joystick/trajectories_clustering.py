import numpy as np
import matplotlib.pyplot as plt
import utils

def process_trajectories(multitrial, time_pre, t_range, subsample_to=None, smoothing_window=None, plot=False, quantity='coordinate', save_with_name='processed_trajects.npy', ballistic=None):

    assert quantity in ['coordinate', 'velocity'], "`quantity` param should be either 'coordinate' or 'velocity'"

    nn = 0 # only one task for now
    n_trials = multitrial[nn]['Ntrials']

    if t_range is not None:
        fixed_duration = t_range[1] - t_range[0]
        durations = np.ones(n_trials, dtype=int) * fixed_duration
    else:
        # TODO: varying range
        pass

    if subsample_to is not None:
        target_length = subsample_to
    else:
        target_length = durations.max()
    z = np.zeros((n_trials, target_length * 2)) # * 2 because traject will be flattened
    z_pre = np.zeros((n_trials, fixed_duration * 2)) # to store trajects before applying any transformations

    from scipy.interpolate import interp1d

    for trialId in range(n_trials):
        this_trial = multitrial[nn]['Trial'+str(trialId+1)]
        origTraject = np.array(this_trial['Trajectory'])

        start = time_pre + t_range[0]
        stop = time_pre + t_range[1]
        preprocessed_traject = origTraject[:, start : stop]
        z_pre[trialId] = preprocessed_traject.flatten() # save before any transformations

        if quantity in ['velocity']:
            finalTraject = np.diff(preprocessed_traject)
        else:
            finalTraject = preprocessed_traject.copy()
        len_before_smoothing = finalTraject.shape[1]

        if smoothing_window is not None:
            kernel = np.ones(smoothing_window) / smoothing_window
            cv1 = np.convolve(finalTraject[0], kernel, 'valid') # x
            cv2 = np.convolve(finalTraject[1], kernel, 'valid') # y
            finalTraject = np.array([cv1, cv2])
        else:
            kernel = None
        
        length = finalTraject.shape[1]

        if length < target_length:
            # interpolate shorter trajectories to be of uniform length (currently not used)
            old_indices = np.arange(length)
            new_indices = np.linspace(0, length - 1, target_length)
            interpolator = interp1d(old_indices, finalTraject, kind='linear')
            finalTraject = interpolator(new_indices)
        elif length > target_length:
            finalTraject, subsampled_indices = utils.subsample(finalTraject, target_length / length)

        plot_subsampling_example = False
        if plot_subsampling_example:
            plot_smoothing_subsampling_example(quantity, preprocessed_traject, finalTraject, len_before_smoothing, length, target_length, smoothing_window)


        start_stim, end_stim = this_trial['time_stim']
        reward = this_trial['time_reward']

        z[trialId] = finalTraject.flatten()

        if plot:
            if quantity in ['velocity']:
                final = preprocessed_traject
            else:
                final = finalTraject
            
            if reward:
                reward_rel = reward - start_stim # turn absolute to relative
            if ballistic:
                plotTrajectory(origTraject, final, t_range[0], t_range[1], reward_rel, time_pre, trialId, ballistic=ballistic[trialId], smoothing_kernel=kernel)

    if save_with_name:
        np.save(output_folder + save_with_name, z)
        np.save(output_folder + 'preprocessed_trajects.npy', z_pre)
    return z


def plotTrajectory(origTraj, finalTraj, start, end, reward_time, time_pre, trialId, ballistic=None, smoothing_kernel=None):

    fig = plt.figure(layout='constrained', figsize=[12.8, 9.6], dpi=200)
    axs = fig.subplot_mosaic([['trajOrig', 'trajFinal'], ['timeseriesXandY', 'timeseriesXandY'], ['velocity', 'velocity']])

    ax = axs['trajOrig']
    ax.plot(origTraj[0], origTraj[1])
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axs['trajFinal']
    # spine=list(ax.spines.values())[0] # set_linestyle # set_linewidth # set_color
    ax.plot(finalTraj[0], finalTraj[1])
    ax.set_xlabel('x'); ax.set_ylabel('y')

    from matplotlib import transforms
    import matplotlib.patches as mpatches
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', transform=trans, linewidth=2, linestyle='dashed')
    ax.add_patch(rect)

    ax = axs['timeseriesXandY']
    ax.plot(origTraj[0], label='x')
    ax.plot(origTraj[1], label='y')

    ax.axvline(x=time_pre + start, color='k', linestyle='--')
    if reward_time:
        ax.axvline(x=time_pre + reward_time, color='g', linestyle='--')
    ax.axvline(x=time_pre + end, color='k', linestyle='--')

    ax.legend()

    ax = axs['velocity']
    velocity = np.diff(origTraj)
    ax.plot(velocity[0], color='black', label='dx', linewidth=0.75)
    ax.plot(velocity[1], color='gray', label='dy')

    absVelocity = np.sqrt(velocity[0]**2 + velocity[1]**2)
    if smoothing_kernel is not None:
        absVelocity = np.convolve(absVelocity, smoothing_kernel, 'same')

    thresh = 0.008
    ax.plot(absVelocity, label='abs')
    ax.legend()
    # ax.axvline(x=time_pre, color='k', linestyle='--')
    if reward_time:
        ax.axvline(x=time_pre + reward_time, color='g', linestyle='--')
    # ax.axvline(x=time_pre + duration, color='k', linestyle='--')

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    rect = mpatches.Rectangle(xy=(time_pre + start, 0), width=end-start, height=1, transform=trans, edgecolor='black', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(rect)



    # ax.vlines([time_pre, time_post], ymin=0, ymax=100, linestyles=['--', '--'], colors=['r', 'r'])
    ax.set_xlabel('time')
    ax.set_ylabel('velocity')
    # ax.set_title('Abs velocity')

    if ballistic: # bad 4, 28 (just outlier)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
        ax.hlines(0.008, time_pre + ballistic[0], time_pre + ballistic[1], color=color)

    # plt.show()
    plt.savefig(output_folder + f'trajects/{trialId}.png')
    plt.close(fig)


def trajectPCA(z, n_comps=2, num_clusters=3):

    label = utils.gen_label(n_comps, num_clusters)

    from utils import doPCA
    z_, components, explained_variance = doPCA(z, n_comps)

    plt.figure()
    num_items = min(30, len(explained_variance))
    plt.bar(range(num_items), explained_variance[:num_items] * 100)
    plt.xlabel('prinicipal component')
    plt.ylabel('explained variance, %')
    plt.savefig(output_folder + "scree.png")


    from utils import cluster
    clust_labels, clust_centers, score, score_by_cluster = cluster(z_, num_clusters)

    if n_comps > 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(clust_centers[:,0], clust_centers[:,1], clust_centers[:,2], color='r', marker='x')

        for clust in range(num_clusters):
            zc = z_[clust_labels == clust].real
            ax.scatter(zc[:,0], zc[:,1], zc[:,2], marker='o') # , c=[cols[clust]] * len(zc), cmap=twil
        plt.savefig(output_folder + f'scatter3d_{label}.png')
        # plt.show()
    else:
        plt.figure()
        if n_comps > 1:
            plt.scatter(clust_centers[:,0], clust_centers[:,1], color='r', marker='x')

        for clust in range(num_clusters):
            zc = z_[clust_labels == clust].real
            plt.scatter(zc[:,0], zc[:,1], marker='o')
        plt.savefig(output_folder + f'scatter{label}.png')
        # plt.show()

    return clust_labels, clust_centers, score, score_by_cluster, explained_variance


def plot_clustered_trajectories(z, n_comps, n_clusters, clust_labels):

    label = utils.gen_label(n_comps, n_clusters)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # overlay clustered trajectories
    for clust in range(n_clusters):
        plt.figure()
        trajs_for_cluster = z[clust_labels == clust]
        for i, traj in enumerate(trajs_for_cluster):
            trajXY = traj.reshape(2, -1).real
            plt.plot(trajXY[0], trajXY[1], color=colors[clust], linewidth=0.75)
        # plt.show()
        plt.savefig(output_folder + f'{label}_{clust}.png')

    # # reconstruct trajectories
    # z_[:, ndims:] = 0
    # inv = np.linalg.inv(eigVects)
    # z_reconstr = np.matmul(z_, inv)

    # # overlay clustered re-constructed trajectories
    # for clust in range(num_clusters):
    #     plt.figure()
    #     trajs_for_cluster = z_reconstr[clust_labels == clust]
    #     for i, traj in enumerate(trajs_for_cluster):
    #         trajXY = traj.reshape(2, -1).real
    #         plt.plot(trajXY[0], trajXY[1], color=colors[clust], linewidth=0.75)
    #     plt.savefig(output_folder + f'clust_{clust}_reconstr.png')
        

def plot_smoothing_subsampling_example(quantity, preprocessed_traject, finalTraject, len_before_smoothing, length, target_length, smoothing_window):
    # example of smoothing and subsampling
    shift = len_before_smoothing - length

    if quantity == 'coordinate':
        x=np.arange(int(shift/2), len_before_smoothing - (int(shift/2) + 1))
        plt.figure()
        plt.plot(preprocessed_traject[0])
        plt.plot(x, cv1)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
        plt.scatter(subsampled_indices + shift/2, finalTraject[0], marker='|', color=color, s=120)

    else: # velocity
        plt.figure()
        velo = np.diff(preprocessed_traject)
        absVelo = np.sqrt(velo[0]**2 + velo[1]**2)
        plt.plot(absVelo)
        len_before_smoothing = absVelo.shape[0]
        kernel = np.ones(smoothing_window) / smoothing_window
        cv1 = np.convolve(absVelo, kernel, 'valid')
        x=np.arange(int(shift/2), len_before_smoothing - (int(shift/2) + 1))
        plt.plot(x, cv1)
        cvv = cv1.reshape(1, 230)
        absvelosubs, subsampled_indices = subsample(cvv, target_length / length)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
        plt.scatter(subsampled_indices + shift/2, absvelosubs[0], marker='|', color=color, s=120)


