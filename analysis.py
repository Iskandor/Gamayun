from analytic.RepresentationAnalysisDPM import RepresentationAnalysisDPM

if __name__ == '__main__':
    analysis = RepresentationAnalysisDPM('ConfigMontezumaDPMAnalysis_v1m2h4_collect_representations.npy', num_rows=4, verbose=False)
    analysis.umap()
    analysis.eigenvalues()
    analysis.differential_entropy()
    analysis.distance_matrix()
    analysis.save('analysis_ConfigMontezumaDPMAnalysis_v1m2h4.png')
    # analysis.confidence_plot()
    # analysis.mapper()
