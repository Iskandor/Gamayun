from analytic.RepresentationAnalysis import RepresentationAnalysis

if __name__ == '__main__':
    analysis = RepresentationAnalysis('ConfigMontezumaSEER_asym_v5m8_collect_representations.npy')
    # analysis.umap()
    analysis.mapper()
