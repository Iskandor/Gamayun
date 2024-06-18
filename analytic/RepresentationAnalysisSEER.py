import matplotlib
import numpy as np
import umap
from gtda.mapper import CubicalCover, make_mapper_pipeline, plot_static_mapper_graph
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


class RepresentationAnalysisSEER:
    def __init__(self, representation_file):
        data = np.load(representation_file, allow_pickle=True).item()

        self.room_ids = np.stack(data['room_ids']).squeeze(1)
        self.targets = np.stack(data['target_repr']).squeeze(1)
        self.next_targets = np.stack(data['target_next_repr']).squeeze(1)

        self.predictions = np.stack(data['predicted_repr']).squeeze(1)
        self.next_predictions = np.stack(data['predicted_next_repr']).squeeze(1)

    def mapper(self):
        # Define the cover (e.g., Cubical cover)
        cover = CubicalCover(n_intervals=10, overlap_frac=0.5)

        # Create the Mapper pipeline
        mapper_pipeline = make_mapper_pipeline(
            cover=cover,
            clusterer=DBSCAN(eps=0.5, min_samples=5),
            filter_func=PCA(n_components=2),
            n_jobs=-1
        )

        self._mapper_call(mapper_pipeline, 'target', self.targets)
        self._mapper_call(mapper_pipeline, 'next_target', self.next_targets)
        self._mapper_call(mapper_pipeline, 'distilled_target', self.predictions)
        self._mapper_call(mapper_pipeline, 'predicted_next_target', self.next_predictions)

    def _mapper_call(self, mapper_pipeline, title, data):
        # Fit the Mapper pipeline to your data
        mapper_pipeline.fit(data)
        # y = mapper_pipeline.transform(self.predictions)

        color_data = self._room_colormap()
        # Plot the Mapper graph
        # plotly_params = {"node_trace": {"marker_colorscale": "Blues"}}
        fig = plot_static_mapper_graph(
            mapper_pipeline,
            data,
            color_data=color_data,
            layout_dim=2,
            # plotly_params=plotly_params,
        )

        fig.layout.title.text = title

        fig.show(config={'scrollZoom': True})
        # fig.write_image(title + '.png')

    def _room_colormap(self):
        cmap = matplotlib.cm.get_cmap('coolwarm')

        color_data = []

        max_room = np.max(self.room_ids)
        for room_id in self.room_ids:
            color_data.append(cmap(room_id/max_room))

        return color_data

    def umap(self):
        reducer = umap.UMAP()

        color_data = self._room_colormap()
        target_embedding = reducer.fit_transform(self.targets)
        next_target_embedding = reducer.transform(self.next_targets)
        # pred_embedding = reducer.transform(self.predictions)
        next_pred_embedding = reducer.transform(self.next_predictions)

        # plt.scatter(target_embedding[:, 0], target_embedding[:, 1], c='blue', s=1, alpha=0.8)
        plt.scatter(next_target_embedding[:, 0], next_target_embedding[:, 1], c='red', s=1, alpha=0.8, label='target')
        # plt.scatter(pred_embedding[:, 0], pred_embedding[:, 1], c='red', s=1, alpha=0.8)
        plt.scatter(next_pred_embedding[:, 0], next_pred_embedding[:, 1], c='blue', s=1, alpha=0.8, label='prediction')
        plt.legend()
        plt.gca().set_aspect('equal', 'datalim')
        plt.show()

    def confidence_plot(self):
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        error = np.linalg.norm(self.next_predictions - self.next_targets, ord=2, axis=1)

        n = self.next_targets.shape[0]

        b, a = np.polyfit(range(n), error, deg=1)
        plt.scatter(x=range(n), y=error, s=1, cmap='coolwarm')
        plt.plot(range(n), a + b * range(n), color="k", lw=2.5)
        plt.xlabel('state order', fontdict=font)
        plt.ylabel('prediction error', fontdict=font)
        plt.show()

