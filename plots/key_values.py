from analytic.metric.NoveltyMetric import NoveltyMetric

key_values = {
    're': 'sum',
    'score': 'sum',
    'ri': 'mean',
    'error': 'mean',
    'loss_prediction': 'val',
    'loss_target': 'val',
    'loss_reg': 'val',
    'loss_target_norm': 'val',
    'specificity': 'val',
    'state_space': 'mean',
    'feature_space': 'mean',
    'ext_value': 'mean',
    'int_value': 'mean',
    NoveltyMetric.KEY: 'val'
}

for v in NoveltyMetric.VAL:
    key = NoveltyMetric.KEY + v
    key_values[key] = 'val'
