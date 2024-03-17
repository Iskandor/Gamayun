import math
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 12}
#
# matplotlib.rc('font', **font)
# matplotlib.rc('axes', titlesize=14)
from tqdm import tqdm

from plots.key_values import key_values


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_data_instance(data_x, data_y, window, smooth=True):
    dx = data_x

    if len(data_y) > 256:
        dy = np.concatenate((data_y[:-128], data_y[-256:-128]))  # at the end of data is garbage from unfinished 128 environments
    else:
        dy = data_y

    if smooth:
        for i in range(len(dy))[1:]:
            dy[i] = dy[i - 1] * 0.99 + dy[i] * 0.01

    max_steps = int(dx[-1])
    data = np.interp(np.arange(start=0, stop=max_steps, step=window), dx, dy)

    iv = list(range(0, max_steps, window))

    return iv, data


# fwd model bug fix - added synonyms
def prepare_data(data, master_key, key, window):
    steps = []
    values = []
    iv = None

    for d in data:
        steps.append(d[master_key]['step'])
        values.append(d[master_key][key])

    result = []
    for x, y in zip(steps, values):
        iv, value = prepare_data_instance(x.squeeze(), y.squeeze(), window)
        result.append(value)

    result = np.stack(result)
    sigma = result.std(axis=0)
    mu = result.mean(axis=0)

    return iv, mu, sigma


def plot_curve(axis, stats, independent_values, color, legend, linestyle='solid', alpha=1.0, start=0.0, stop=1.0):
    start = int(len(independent_values) * start)
    stop = int(len(independent_values) * stop)
    if 'val' in stats:
        line, = axis.plot(independent_values[start:stop], stats['val'][start:stop], lw=1, color=color, alpha=alpha, label=legend, linestyle=linestyle)

    if 'sum' in stats:
        line, = axis.plot(independent_values[start:stop], stats['sum'][start:stop], lw=1, color=color, alpha=alpha, label=legend, linestyle=linestyle)

    if 'mean' in stats:
        line, = axis.plot(independent_values[start:stop], stats['mean'][start:stop], lw=1, color=color, alpha=alpha, label=legend, linestyle=linestyle)
        if 'std' in stats:
            axis.fill_between(independent_values[start:stop], stats['mean'][start:stop] + stats['std'][start:stop], stats['mean'][start:stop] - stats['std'][start:stop], facecolor=color, alpha=0.3)

    if 'max' in stats:
        line = axis.plot(independent_values[start:stop], stats['max'][start:stop], lw=2, color=color, alpha=alpha, label=legend, linestyle=linestyle)

    return line


def get_rows_cols(data, template=None):
    n = len(data)

    if template:
        pass

    rows = int(sqrt(n))
    cols = math.ceil(n / rows)

    return rows, cols


def plot_chart(num_rows, num_cols, index, key, data, value_key, window, color, legend, legend_loc=4, linestyle='solid'):
    ax = plt.subplot(num_rows, num_cols, index)
    ax.set_xlabel('steps')
    # ax.set_ylabel(legend)
    ax.grid(visible=True)

    stats = {}
    iv = None

    for k in value_key:
        iv, stats[k] = prepare_data_instance(data[key]['step'].squeeze(), data[key][k].squeeze(), window)

    # plot_curve(ax, stats, iv, color=color, alpha=1.0, start=0.0, stop=1.0)
    plot_curve(ax, stats, iv, color=color, legend=legend, linestyle=linestyle, alpha=1.0, start=0.01, stop=1.0)

    # if type(legend) is not list:
    #     legend = [legend]
    # plt.legend(legend, loc=legend_loc)
    plt.legend()


def plot_multiple_models(keys, data, legend, labels, colors, path, window=1):
    num_rows = 1
    num_cols = len(keys)

    plt.figure(figsize=(num_cols * 6.40, num_rows * 5.12))

    for i, key in enumerate(keys):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.set_xlabel('steps')
        ax.set_ylabel(labels[i])
        ax.grid()

        lines = []
        for index, d in enumerate(data):
            iv, mu, sigma = prepare_data(d, key, key_values[key], window)
            iv = [val / 1e6 for val in iv]
            # mu = np.clip(mu, 0, 0.1)
            lines.append(plot_curve(ax, {'mean': mu, 'std': sigma}, iv, color=colors[index], legend=None, start=0.0))

        if legend is not None:
            ax.legend(lines, legend[:len(data)], loc=0)

    plt.savefig(path + ".png")
    plt.close()


def plot_detail(data, path, template, window=1000):
    linestyle_cycle = ['solid', 'dotted', 'dashed', 'dashdot']
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        index = 0
        for composite in template.elements:
            index += 1
            charts = 0
            for e in composite:
                if e.key in data[i]:
                    plot_chart(num_rows, num_cols, index, e.key, data[i], e.values, window, color=e.color, legend=e.legend, linestyle=linestyle_cycle[charts % len(linestyle_cycle)])
                    charts += 1
            if charts == 0:
                index -= 1

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_baseline(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_cnd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'std', 'max'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'feature_space', data[i], ['mean', 'std'], window, color='green', legend='feature space')
        plot_chart(num_rows, num_cols, 5, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'loss_target', data[i], ['val'], window, color='magenta', legend='loss target', legend_loc=9)
        index = 7
        if 'specificity' in data[i]:
            plot_chart(num_rows, num_cols, index, 'specificity', data[i], ['val'], window, color='orange', legend='TP specificity', legend_loc=9)
            index += 1
        if 'augmentor_loss_var' in data[i]:
            plot_chart(num_rows, num_cols, index, 'augmentor_loss_var', data[i], ['val'], window, color='magenta', legend='augmentor variable loss', legend_loc=9)
            index += 1
        if 'augmentor_loss_con' in data[i]:
            plot_chart(num_rows, num_cols, index, 'augmentor_loss_con', data[i], ['val'], window, color='magenta', legend='augmentor constant loss', legend_loc=9)
            index += 1

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_seer(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'feature_space', data[i], ['mean', 'std'], window, color='green', legend='feature space')

        index = 5
        if 'loss_target' in data[i]:
            plot_chart(num_rows, num_cols, index, 'loss_target', data[i], ['val'], window, color='magenta', legend='loss target', legend_loc=9)
            index += 1
        if 'loss_prediction' in data[i]:
            plot_chart(num_rows, num_cols, index, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss prediction', legend_loc=9)
            index += 1
        if 'loss_forward' in data[i]:
            plot_chart(num_rows, num_cols, index, 'loss_forward', data[i], ['val'], window, color='magenta', legend='loss forward', legend_loc=9)
            index += 1
        if 'loss_distillation' in data[i]:
            plot_chart(num_rows, num_cols, index, 'loss_distillation', data[i], ['val'], window, color='magenta', legend='loss distillation', legend_loc=9)
            index += 1
        if 'loss_hidden' in data[i]:
            plot_chart(num_rows, num_cols, index, 'loss_hidden', data[i], ['val'], window, color='magenta', legend='loss hidden', legend_loc=9)
            index += 1
        if 'distillation_reward' in data[i]:
            plot_chart(num_rows, num_cols, index, 'distillation_reward', data[i], ['mean', 'std'], window, color='red', legend='distillation reward', legend_loc=9)
            index += 1
        if 'forward_reward' in data[i]:
            plot_chart(num_rows, num_cols, index, 'forward_reward', data[i], ['mean', 'std'], window, color='red', legend='forward reward', legend_loc=9)
            index += 1

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_rnd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 7, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_icm(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss', data[i], ['val'], window, color='magenta', legend='loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'inverse_loss', data[i], ['val'], window, color='magenta', legend='inverse model loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 7, 'forward_loss', data[i], ['val'], window, color='magenta', legend='forward model loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 8, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 9, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')
        plot_chart(num_rows, num_cols, 10, 'feature_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='feature space', legend_loc=9)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_fwd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_target', data[i], ['val'], window, color='magenta', legend='loss_target', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'loss_target_norm', data[i], ['val'], window, color='magenta', legend='loss_target_norm', legend_loc=9)
        plot_chart(num_rows, num_cols, 7, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss_prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 8, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 9, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')
        # plot_chart(num_rows, num_cols, 10, 'feature_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='feature space', legend_loc=9)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()
