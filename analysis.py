from analytic.RepresentationAnalysis import RepresentationAnalysis

if __name__ == '__main__':
    analysis = RepresentationAnalysis('ConfigMontezumaSEER_asym_v5m9_collect_representations.npy')
    analysis.umap()
    analysis.confidence_plot()
    # analysis.mapper()
