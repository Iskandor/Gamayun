import matplotlib
import numpy as np
import scipy
import seaborn
import umap
from gtda.mapper import CubicalCover, make_mapper_pipeline, plot_static_mapper_graph
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from tqdm import tqdm


class Representation:
    def __init__(self, label, data):
        self._label = label
        self._data = data

    @property
    def label(self):
        return self._label

    @property
    def data(self):
        return self._data


class RepresentationAnalysisDPM:
    def __init__(self, representation_file, num_rows=4, verbose=False):
        self.verbose = verbose

        self.num_rows = num_rows
        self.num_cols = 3
        self.index = 1
        self.figure = plt.figure(figsize=(self.num_cols * 5.00, self.num_rows * 5.00))

        data = np.load(representation_file, allow_pickle=True).item()

        self.room_ids = np.stack(data['room_ids'])
        self.representations = []
        self.representations.append(Representation('PPO encoder', np.stack(data['zppo_state']).squeeze(1)))
        self.representations.append(Representation('Target encoder', np.stack(data['zt_state']).squeeze(1)))
        self.representations.append(Representation('Forward model encoder', np.stack(data['zf_next_state']).squeeze(1)))

    def _room_colormap(self):
        cmap = matplotlib.cm.get_cmap('coolwarm')

        color_data = []

        max_room = np.max(self.room_ids)
        for room_id in self.room_ids:
            color_data.append(cmap(room_id / max_room))

        return color_data

    def umap(self):
        reducer = umap.UMAP()
        color_data = self._room_colormap()

        for representation in tqdm(self.representations):
            ax = plt.subplot(self.num_rows, self.num_cols, self.index)
            # ax.set_xlabel('steps')
            # ax.set_ylabel(legend)
            ax.grid(visible=True)
            ax.title.set_text(representation.label)

            embeding = reducer.fit_transform(representation.data)
            scatter = ax.scatter(embeding[:, 0], embeding[:, 1], c=self.room_ids, s=3, alpha=1.0, cmap=plt.cm.get_cmap('RdYlBu'))
            plt.colorbar(scatter, ax=ax)

            self.index += 1

        # plt.legend()
        # plt.gca().set_aspect('equal', 'datalim')
        if self.verbose:
            plt.show()

    def eigenvalues(self):
        pca = PCA()

        for representation in tqdm(self.representations):
            ax = plt.subplot(self.num_rows, self.num_cols, self.index)
            ax.set_yscale('log')
            # ax.set_xlabel('steps')
            # ax.set_ylabel(legend)
            ax.grid(visible=True)
            ax.title.set_text(representation.label)

            pca.fit(representation.data)
            eigenvalues = pca.singular_values_ ** 2
            ax.bar(x=range(eigenvalues.shape[0]), height=eigenvalues)

            self.index += 1

        # plt.legend()
        # plt.gca().set_aspect('equal', 'datalim')
        if self.verbose:
            plt.show()

    def differential_entropy(self):
        for representation in tqdm(self.representations):
            ax = plt.subplot(self.num_rows, self.num_cols, self.index)
            # ax.set_xlabel('steps')
            # ax.set_ylabel(legend)
            ax.grid(visible=True)
            ax.title.set_text(representation.label)

            h = []
            for i in range(representation.data.shape[1]):
                h.append(scipy.stats.differential_entropy(representation.data[:, i]))
            h = np.sort(np.array(h))
            # ax.set_ylim([np.min(h), np.max(h)])
            ax.bar(x=range(h.shape[0]), height=h, color='red')
            self.index += 1

        # plt.legend()
        # plt.gca().set_aspect('equal', 'datalim')
        if self.verbose:
            plt.show()

    def distance_matrix(self):
        for representation in tqdm(self.representations):
            ax = plt.subplot(self.num_rows, self.num_cols, self.index)
            # ax.set_xlabel('steps')
            # ax.set_ylabel(legend)
            ax.grid(visible=True)
            ax.title.set_text(representation.label)

            axx = ax.inset_axes([-0.05, -0.25, 1.1, 0.2], )
            axx.plot(range(self.room_ids.shape[0]), self.room_ids, color='blue')
            axx.set_xticks([])

            dist_matrix = scipy.spatial.distance_matrix(representation.data, representation.data)
            seaborn.heatmap(dist_matrix, cmap='coolwarm')
            ax.set_xticks([])

            self.index += 1

        # plt.legend()
        # plt.gca().set_aspect('equal', 'datalim')
        if self.verbose:
            plt.show()

    def save(self, filename):
        plt.savefig(filename)
        plt.close()
