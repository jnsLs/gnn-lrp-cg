import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from scipy.spatial.distance import cdist


###############################################################################
# visualization utils
###############################################################################


def shrink(rx, ry, factor=11):
    rx = np.array(rx)
    ry = np.array(ry)

    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    last_node = rx.shape[0] - 1
    concat_list_x = [np.linspace(rx[0], rx[0], 5)]
    concat_list_y = [np.linspace(ry[0], ry[0], 5)]
    for j in range(last_node):
        concat_list_x.append(np.linspace(rx[j], rx[j + 1], 5))
        concat_list_y.append(np.linspace(ry[j], ry[j + 1], 5))
    concat_list_x.append(np.linspace(rx[last_node], rx[last_node], 5))
    concat_list_y.append(np.linspace(ry[last_node], ry[last_node], 5))

    rx = np.concatenate(concat_list_x)
    ry = np.concatenate(concat_list_y)

    filt = np.exp(-np.linspace(-2, 2, factor) ** 2)
    filt = filt / filt.sum()

    rx = np.convolve(rx, filt, mode="valid")
    ry = np.convolve(ry, filt, mode="valid")

    return rx, ry


###############################################################################
# 2D visualization
###############################################################################
def relevance_vis_2d(
    ax,
    relevances,
    atomic_numbers,
    pos,
    graph,
    cmap=plt.cm.get_cmap("viridis"),
    relevance_scaling=2.0,
    scaling_type="lin",
    shrinking_factor=1,
):
    ####################################################################################################################
    # Utils
    ####################################################################################################################
    if graph is None:
        dists = cdist(pos, pos)
        graph = (dists < 1.6).astype(float)
        graph -= np.eye(graph.shape[0])

    def _iterate_over_all_walks(ax):
        # visualization settings
        selfloopwidth = 0.32
        linewidth = 13.0
        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(relevances):
            # get walk color
            if cmap is not None:
                color = cmap(relevance)
                alpha = 0.7
            else:
                color = "b" if relevance < 0 else "r"
                # scale relevance value
                if scaling_type == "lin":
                    relevance_val = abs(relevance) * relevance_scaling
                elif scaling_type == "root":
                    relevance_val = (abs(relevance) * relevance_scaling) ** 0.7
                elif scaling_type == "log":
                    relevance_val = math.log10(abs(relevance * relevance_scaling) + 1.0)
                else:
                    raise NotImplementedError
                # clip alpha
                if abs(relevance_val) > 1.0:
                    alpha = 1.0
                    print("clipped relevance")
                else:
                    alpha = abs(relevance_val)
            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            # plot self loops
            for i in range(len(rx) - 1):
                if rx[i] == rx[i + 1] and ry[i] == ry[i + 1]:
                    rx_tmp = rx[i] + selfloopwidth * np.cos(
                        np.linspace(0, 2 * np.pi, 16)
                    )
                    ry_tmp = ry[i] + selfloopwidth * np.sin(
                        np.linspace(0, 2 * np.pi, 16)
                    )
                    ax.plot(
                        rx_tmp,
                        ry_tmp,
                        color=color,
                        alpha=alpha,
                        lw=linewidth,
                        zorder=1.0,
                    )
            # plot walks
            rx, ry = shrink(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.0)
        return ax

    ####################################################################################################################
    # Main function code
    ####################################################################################################################

    # plot walks
    ax = _iterate_over_all_walks(ax)
    # prepare molecular graph
    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    names = [atom_names_dict[Z.item()] for i, Z in enumerate(atomic_numbers)]
    G = nx.from_numpy_array(graph)
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", node_size=1200)
    collection.set_zorder(2.0)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=2,
        style="dotted",
        node_size=300,
    )
    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    nx.draw_networkx_labels(
        G, pos_labels, {i: name for i, name in enumerate(names)}, font_size=40
    )
    return ax


def scaling(relevance, log_scaling):
    # define opacity_scaling (logarithmic)
    if log_scaling:
        if math.log10(abs(relevance) + 1) > 1:
            opacity_scaling = 1
        else:
            opacity_scaling = math.log10(abs(relevance) + 1)
    # linear scaling
    else:
        if abs(relevance) > 1:
            opacity_scaling = 1
        else:
            opacity_scaling = abs(relevance)
            # return
    return opacity_scaling
