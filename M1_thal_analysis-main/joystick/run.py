import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import trajectories_clustering as tc

quantity = 'coordinate' # 'coordinate' or 'velocity'
t_start = 25
duration = 250
subsample_to = 35
smoothing_window = 20

n_comps = 3
n_clusters = 4

outliers = [40, 45, 52]

output_folder = utils.make_output_folder(t_start, duration, quantity, smoothing_window, subsample_to, outliers)
tc.output_folder = output_folder

try:
    # try loadiing already processed trajects
    z = np.load(output_folder + 'processed_trajects.npy')
except:
    # load source data and process
    import load, joystick.trajectory_analysis as ta
    multitrial, time_pre, time_post = load.load_and_parse()

    ballistic = ta.detect_ballistic_phase(multitrial, time_pre, smoothing_window)

    z = tc.process_trajectories(multitrial, time_pre, t_range=[t_start, t_start + duration], quantity=quantity, subsample_to=subsample_to, smoothing_window=smoothing_window, plot=True, ballistic=ballistic)

inds = np.arange(len(z))
z_pre = np.load(output_folder + 'preprocessed_trajects.npy')

if outliers is not None:
    filter = np.ones(len(inds), dtype=bool)
    filter[outliers] = False

    z = z[filter]
    z_pre = z_pre[filter]
    inds = inds[filter]

result = tc.trajectPCA(z, n_comps=n_comps, num_clusters=n_clusters)
# unpack:
clust_labels, clust_centers, score, score_by_cluster, explained_variance = result


tc.plot_clustered_trajectories(z_pre, n_comps, n_clusters, clust_labels)

# save report to file:
utils.print_and_save_report(output_folder, n_comps, n_clusters, inds, clust_labels, clust_centers, score, score_by_cluster, explained_variance)



