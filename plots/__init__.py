import os

import matplotlib

from plots.analytic_chart import plot_multiple_models, plot_detail_cnd, plot_detail_rnd, plot_detail_baseline, plot_detail_icm, plot_detail_fwd, plot_detail_seer, plot_detail
from plots.dataloader import prepare_data
from plots.paths import plot_root
from plots.template import ChartTemplates


def plot(name, config, keys, labels=None, legend=None, plot_overview=True, plot_details=None, window=1000):
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 16}
    #
    # matplotlib.rc('font', **font)

    if plot_details is None:
        plot_details = []
    data = prepare_data(config)
    algorithm = config[0]['algorithm']
    env = config[0]['env']

    if labels is None:
        labels = keys

    if legend is None:
        legend = [key['legend'] if 'legend' in key else '{0:s} {1:s}'.format(key['model'], key['id']) for key in config]

    if plot_overview:
        path = os.path.join(plot_root)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, name)
        plot_multiple_models(
            keys,
            data,
            legend,
            labels,
            ['blue', 'red', 'green', 'magenta', 'lightseagreen', 'orange', 'purple', 'gray', 'navy', 'maroon', 'brown', 'apricot', 'olive', 'beige', 'yellow'],
            path,
            window)

        chart_templates = ChartTemplates()

        for index, key in enumerate(config):
            if key['id'] in plot_details:
                d = data[index]
                model = key['model']
                id = key['id']

                # path = os.path.join(plot_root, algorithm, env, model)
                path = os.path.join(plot_root)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, '{0:s}_{1:s}'.format(model, id))
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, '{0:s}_{1:s}_{2:s}'.format(env, model, str(key['id'])))

                plot_detail(d, path, chart_templates[model], window)
                # if model == 'baseline':
                #     plot_detail_baseline(d, path, window)
                # if model == 'snd':
                #     plot_detail_cnd(d, path, window)
                # if model == 'seer':
                #     plot_detail_seer(d, path, window)
                # if model == 'rnd':
                #     plot_detail_rnd(d, path, window)
                # if model == 'qrnd':
                #     plot_detail_rnd(d, path, window)
                # if model == 'icm':
                #     plot_detail_icm(d, path, window)
                # if model == 'fwd':
                #     plot_detail_fwd(d, path, window)
