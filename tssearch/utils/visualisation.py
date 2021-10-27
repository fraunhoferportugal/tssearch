import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import novainstrumentation as ni

from matplotlib.collections import LineCollection


# Visualisation
def plot_alignment(ref_signal, estimated_signal, path, **kwargs):
    """
    This functions plots the resulted alignment of two sequences given the path
    calculated by the Dynamic Time Warping algorithm.

    :param ref_signal: (array-like)
                     The reference sequence.
    :param estimated_signal: (array-like)
                     The estimated sequence.
    :param path: (array-like)
                     A 2D array congaing the path resulted from the algorithm
    :param \**kwargs:
        See below:

        * *offset* (``double``) --
            The offset used to move the reference signal to an upper position for
            visualization purposes.
            (default: ``2``)

        * *linewidths* (``list``) --
            A list containing the linewidth for the reference, estimated and connection
            plots, respectively.
            (default: ``[3, 3, 0.5]``)

        * *step* (``int``) --
            The step for
          (default: ``2``)

        * *colors* (``list``) --
          A list containing the colors for the reference, estimated and connection
          plots, respectively.
          (default: ``[sns.color_palette()[0], sns.color_palette()[1], 'k']``)
    """

    step = kwargs.get("step", 2)
    hoffset = kwargs.get("hoffset", 0)
    voffset = kwargs.get("offset", 2)
    linewidths = kwargs.get("linewidths", [3, 3, 0.5])
    colors = kwargs.get("colors", [sns.color_palette()[0], sns.color_palette()[1], "k"])

    copy_ref = np.copy(ref_signal)  # This prevents unexpected changes in the reference signal after the duplicate
    copy_ref += voffset * np.max(ref_signal)  # Set an offset for visualization

    xref = np.arange(len(copy_ref)) + hoffset
    # Actual plot occurs here
    plt.plot(xref, copy_ref, color=sns.color_palette()[0], lw=linewidths[0], label="reference")
    plt.plot(estimated_signal, color=sns.color_palette()[1], lw=linewidths[1], label="estimate")
    plt.legend(fontsize=17)

    [
        plt.plot(
            [[path[0][i] + hoffset], [path[1][i]]],
            [copy_ref[path[0][i]], estimated_signal[path[1][i]]],
            color=colors[2],
            lw=linewidths[2],
        )
        for i in range(len(path[0]))[::step]
    ]

    ref_pks = ni.peakdelta(copy_ref, 0.5)
    est_pks = ni.peakdelta(estimated_signal, 0.5)

    if len(ref_pks[0]) > 0 and len(ref_pks[0]) == len(est_pks[0]):
        for i in range(len(ref_pks[0])):
            plt.plot([ref_pks[0][i, 0], est_pks[0][i, 0]], [ref_pks[0][i, 1], est_pks[0][i, 1]], "red")

    if len(ref_pks[1]) > 0 and len(ref_pks[1]) == len(est_pks[1]):
        for i in range(len(ref_pks[1])):
            plt.plot([ref_pks[1][i, 0], est_pks[1][i, 0]], [ref_pks[1][i, 1], est_pks[1][i, 1]], "red")


def plot_costmatrix(matrix, path):
    """
    This functions overlays the optimal warping path and the cost matrices
    :param matrix: (ndarray-like)
                The cost matrix (local cost or accumulated)
    :param path:   (ndarray-like)
                The optimal warping path
    :return: (void)
                Plots the optimal warping path with an overlay of the cost matrix.
    """
    plt.imshow(matrix.T, cmap="viridis", origin="lower", interpolation="None")
    plt.colorbar()
    plt.plot(path[0], path[1], "w.-")
    plt.xlim((-0.5, matrix.shape[0] - 0.5))
    plt.ylim((-0.5, matrix.shape[1] - 0.5))


def plot_search_distance_result(res, sequence, ts=None, cmap_name="viridis"):

    if ts is None:
        ts = np.arange(len(sequence))
    # set distance scale
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))

    if len(np.shape(sequence)) > 1:
        sequence_shape = np.shape(sequence)[1]
    else:
        sequence_shape = 1

    all_axs = []
    for k in res.keys():
        max_dist = np.max(res[k]["path_dist"])
        min_dist = np.min(res[k]["path_dist"])
        if max_dist == min_dist:
            max_dist += 1
            min_dist -= 1
        delta_dist = max_dist - min_dist

        fig, axs = plt.subplots(sequence_shape + 1, 1, figsize=(15, 5))
        axs[0].set_title(k)
        for i in range(sequence_shape):
            plot_seq = sequence if sequence_shape == 1 else sequence[:, i]
            axs[i].plot(ts, plot_seq, "lightgray")
            for s, e, d in zip(res[k]["start"], res[k]["end"], res[k]["path_dist"]):
                d_idx = int((d - min_dist) * cmap.N / delta_dist) - 1
                axs[i].plot(ts[np.arange(s, e)], plot_seq[s:e], c=colors[d_idx])
            if i < sequence_shape - 2:
                axs[i].sharex(axs[i + 1])
                axs[i].set_xticks([])
                # [axs[i-1].sharex(axs[i]) for i in range(1, sequence_shape)]
        axs[sequence_shape].set_xlabel("Distance")
        axs[sequence_shape].imshow([colors], extent=[min_dist, max_dist, 0, 0.02 * delta_dist])
        axs[sequence_shape].set_yticks([])

        all_axs += [axs]

    return all_axs


def plot_weight_query(x, query, weight, cmap="viridis", axs=None, fig=None):

    points = np.array([x, query]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if axs is None:
        fig = plt.figure()
        axs = plt.subplot()

    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, weight.max()))
    lc.set_array(weight)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)

    cbar = fig.colorbar(line, ax=axs)
    cbar.set_label("weight")

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(query.min() - 1, query.max() + 1)
    plt.show()
