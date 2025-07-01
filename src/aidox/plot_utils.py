
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from collections.abc import Callable
from sklearn.preprocessing import MinMaxScaler
from .constants import FONT


matplotlib.rc('font', **FONT)


def plot_loss(
        it: int,
        output: str,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        anchor_point: dict = {},
        features: list = [],
        dummies: list = [],
        npoints: int = 50,
        n_exp_points: int = 5,
        plot_hist: bool = False,
        plot_scatter: bool = False,
        eval_function: Callable = None,
        loss_function: Callable = None,
        doe_points: dict = None,
        constr_dict: dict = {},
        fig=None,
        axs=None,
        c: np.ndarray = np.array([1, 1, 1, 1])):
    d = len(features)
    if fig is None and axs is None:
        cols = 3
        rows = math.ceil(d / cols)
        fig, axs = plt.subplots(
            nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows)
        )
    X = df_x.values
    y = df_y.values
    id_out = list(df_y.columns).index(output)
    if doe_points is not None:
        exp_df = pd.DataFrame(doe_points).T
        x_exp = exp_df[df_x.columns].values
        y_exp = exp_df[df_y.columns].values

    inputs = list(df_x.columns) # pipe.feature_names_in_.tolist()

    anchor_idx = {inputs.index(k): v for k, v in anchor_point.items()}
    dummies_idx = [list(inputs).index(k) for k in list(set(dummies) & set(inputs))]
    fig.suptitle(f'Loss_steps_{output}')
    for idx, ax in enumerate(axs.flatten()):
        if idx < d:
            i_idx = inputs.index(features[idx])
            # eval min max for i feature
            if idx in dummies_idx:
                x_col_val_i = np.unique(X[:, i_idx])
                npoints = x_col_val_i.shape[0]
            else:
                min_i, max_i = X[:, i_idx].min(), X[:, i_idx].max()
                npoints_i = npoints
                x_col_val_i = np.linspace(min_i, max_i, npoints_i)
            anchor_upd = dict(ChainMap(*[x[2] for x in constr_dict]))
            x = build_slice_x(X=X, dummies_idx=dummies_idx, npoints=npoints)
            # set anchor points
            for anc_, fix_val in anchor_idx.items():
                x[:, anc_] = fix_val
            for anc_, fix_val in anchor_upd.items():
                x[:, inputs.index(anc_)] = fix_val

            x[:, i_idx] = x_col_val_i

            z, std = eval_function(pd.DataFrame(x, columns=inputs))

            to_plot = loss_function(z, std)

            ax.plot(x[:, i_idx], to_plot, label=f'loss_{str(it)}', color=c)
            ax.set_ylabel('loss')
            ax.set_xlabel(features[idx])
            ax.set_xlabel(features[idx])

            if plot_hist:
                ax2 = ax.twinx()
                ax2.hist(X[:, i_idx], alpha=0.3, color='orange', label='input distribution [counts]')
                ax2.set_yticks([])
            if plot_scatter:
                drop_cols = list(set([i_idx]) | set(dummies_idx))
                ref_anc = np.tile(np.delete(x, drop_cols, 1)[0], (X.shape[0], 1)).astype(float)
                exp_point = np.delete(X, drop_cols, 1).astype(float)
                ord_ = np.argsort(np.linalg.norm(ref_anc - exp_point, axis=1))[:n_exp_points]
                ax3 = ax.twinx()
                ax3.scatter(X[ord_, i_idx], y[ord_, id_out], alpha=0.3, color='k',
                            label='experiments')
                ax3.set_ylabel(output)

                ax3.scatter(x_exp[:, i_idx],
                            y_exp[:, id_out], alpha=0.5, color='r',
                            label='doe')
        else:
            ax.axis('off')

    fig.tight_layout()
    handles = [ax.get_legend_handles_labels()[0] for ax in axs.flatten()]
    handles_flat = [item for sublist in handles for item in sublist]
    labels = [ax.get_legend_handles_labels()[1] for ax in axs.flatten()]
    unique_labels, unique_index = np.unique([item for sublist in labels for item in sublist], return_index=True)
    fig.legend([handles_flat[i] for i in unique_index], unique_labels, loc='upper right')
    return fig, axs



def get_nearest_points(x: np.ndarray, X: np.ndarray, i_idx: int, n_exp_points: int = 100, dummies_idx: list = [],
                       j_idx: int = None):
    """
    Get order of the nearest points

    :param x: sensitivity input dataset
    :param X: experimental input dataset
    :param i_idx: index of the features to be investigated (and removed when evaluating distance)
    :param n_exp_points: number of experimental points to be choose among nearest
    :param j_idx: index of the features to be investigated (and removed when evaluating distance)
    :return order_: order of nearest experimental points

    """
    # eval normalize distance
    x = np.delete(x, dummies_idx, axis=1)
    X = np.delete(X, dummies_idx, axis=1)
    scaler_cls = MinMaxScaler()
    scaler_cls.fit(x)
    scale_X = scaler_cls.transform(X)
    scale_x = scaler_cls.transform(x)
    # remove i_idx feature
    if i_idx < scale_x.shape[1]:
        ref_anc = np.tile(np.delete(scale_x, i_idx, 1)[0], (X.shape[0], 1))
        exp_point = np.delete(scale_X, i_idx, 1)
    else:
        ref_anc = np.tile(scale_x[0], (X.shape[0], 1))
        exp_point = scale_X
    if j_idx:
        if j_idx < ref_anc.shape[1]:
            # erase also j index feature
            ref_anc = np.tile(np.delete(ref_anc, j_idx - 1, 1)[0], (X.shape[0], 1))
            exp_point = np.delete(exp_point, j_idx - 1, 1)
        else:
            ref_anc = np.tile(ref_anc[0], (X.shape[0], 1))
    # eval distance and sort values
    order_ = np.argsort(np.linalg.norm(ref_anc - exp_point, axis=1))[:n_exp_points]
    return order_


def get_most_freq(x: np.ndarray) -> np.ndarray:
    """
    Get most frequent value of array

    :param x: input array
    :return: highest frequent val

    """
    x = pd.Series(x).dropna().values
    unique, pos = np.unique(x, return_inverse=True)
    return unique[np.bincount(pos).argmax()]


def build_slice_x(X: np.ndarray, dummies_idx: list, npoints: int):
    """
    Build slide base on input array

    :param X: input array
    :param dummies_idx: dummies index
    :param npoints: number of points used for dummies
    :return: transformed x slices

    """
    dum_arr = X[:, dummies_idx]
    if dummies_idx:
        lst_dum = []
        for ax_ in range(dum_arr.shape[1]):
            # get most frequent value of the dummy variable
            lst_dum.append(get_most_freq(dum_arr[:, ax_]))
        dum_arr_m = np.array(lst_dum)
    else:
        dum_arr_m = np.array([])
    # delete the dummy value
    num_arr = np.delete(X, dummies_idx, axis=1)
    # perform mean on continuos variable
    num_arr_m = num_arr.mean(axis=0)
    x = num_arr_m
    ord_dummies_idx = np.argsort(dummies_idx).astype(int)
    # set value for dummies variable
    for idx_, val_ in zip(np.array(dummies_idx)[ord_dummies_idx], np.array(dum_arr_m)[ord_dummies_idx]):
        x = np.insert(x, idx_, val_)
    # repeat across axis
    x_d = np.tile(x, (npoints, 1))
    return x_d


def map_arr(ar: np.ndarray, dic: dict) -> np.ndarray:
    """
    Map array using dict

    :param ar: input array
    :param dic: mapping dict
    :return: mapped array

    """
    mapped_ar = [dic[id_] for id_ in ar]
    return np.array(mapped_ar)



def eval_sum(val, res):
    """
    Eval difference

    :param val:
    :param res:
    :return: difference of array and value

    """
    return res - val


def real_to_integer(x, val: int = 10):
    """
    Convert real to intefer

    :param x:
    :param val:
    :return:

    """
    return np.round(x * val) / val


def find_nearest(array, value):
    """
    Find the nearest point.

    :param array:
    :param value:
    :return:

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rename_label(label, replace_dict: dict):
    """
    rename the labels

    :param label:
    :param replace_dict:
    :return:

    """
    for x, val in replace_dict.items():
        label = label.replace(x, val)
    return label
