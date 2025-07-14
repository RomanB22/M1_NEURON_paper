import numpy as np
import os

def subsample(data, decimation_factor):
    # assuming data is in shape (n_trajects, traject_len)

    assert (0 < decimation_factor <= 1), "Decimation factor should be in range (0, 1]"

    if decimation_factor == 1:
        return data

    orig_len = data.shape[1]
    new_len = int(orig_len * decimation_factor)

    indices = np.arange(new_len) * (orig_len / new_len)
    indices = np.round(indices).astype(int)

    return data[:, indices], indices


def doPCA(z, n_components):

    z = z.T # so that each column is a data point
    cov = np.cov(z)

    eigVals, eigVects = np.linalg.eigh(cov)
    eigVals = np.flip(eigVals)
    eigVects = np.flip(eigVects, axis=1)
    explained_variance = eigVals / eigVals.sum()

    # keep only first `n_components` eigen vectors (which are prinicipal components)
    components = eigVects[:,:n_components] # they are colums in eigVects matrix

    # change of basis into principal components space for all signals
    z_ = np.matmul(components.T, z) # for orthogonal matrix, transpose is the same as inverse
    z_ = z_.T
    # Now z_ has shape (number_of_signals, n_components), so each row is a signal projected to n_components-dimensional space where pricncipal components form the ortogonal basis

    # # RECONSTRUCT
    # inv = np.linalg.inv(eigVects)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(70):
    #     try:
    #         z0 = z_[0]
    #         z0[69-i:] = 0
    #         reconstructed = np.matmul(z0, inv).reshape(2,-1)
    #         plt.plot(reconstructed[0], reconstructed[1])
    #         print(f'done {i}')
    #     except:
    #         pass

    return z_, components, explained_variance.real


def cluster(points, num_clusters):

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    # Assuming that each row represents an n-dimensional point

    kmeans = KMeans(n_clusters=num_clusters, n_init=1000, random_state=123)

    kmeans.fit(points)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Silhouette coefficients - mean and by clusters
    score = silhouette_score(points, cluster_labels)
    coefs = silhouette_samples(points, cluster_labels)
    coefs_by_cluster = []
    for clust in range(num_clusters):
        cbc = coefs[cluster_labels == clust]
        coefs_by_cluster.append(cbc.mean())
    coefs_by_cluster = np.array(coefs_by_cluster)

    return cluster_labels, cluster_centers, score, coefs_by_cluster


def gen_label(n_comps, n_clusters):
    return f"{n_comps}comps_{n_clusters}clust"


def make_output_folder(t_start, duration, quantity, smoothing_window, subsample_to, outliers=None):

    if outliers:
        outliers_str = '-'.join([str(o) for o in outliers])
        outliers_str = f'_outl{outliers_str}'
    else:
        outliers_str = ''
    output_folder = f'output/traject_clustering_start{t_start}_dur{duration}_{quantity}_smooting{smoothing_window}_subs{subsample_to}{outliers_str}/'

    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(output_folder + 'trajects/'): os.mkdir(output_folder + 'trajects/')
    return output_folder

def print_and_save_report(output_folder, n_comps, n_clusters, elements_inids, clust_labels, clust_centers, score, score_by_cluster, explained_variance):

    label = gen_label(n_comps, n_clusters)
    # save report to file:
    with open(output_folder + f'report_{label}.txt', 'w') as file:
        expl = np.round(explained_variance[:n_comps].sum() * 100, 2)

        report = [f"Generation with {n_comps} principal components and {n_clusters} clusters.\nComponents explain {expl}% of variance\n"]


        report.append(f'Labels: {list(zip(elements_inids, clust_labels))}\n')
        report.append(f'Centers: {clust_centers}\n')
        report.append(f'Score: {score}\n')
        report.append(f'Scores by clusters: {score_by_cluster}\n')

        file.writelines(report)

        for r in report:
            print(r)
