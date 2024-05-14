import os

from plots import plot
from plots.analytic_table import compute_table_values

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def atari_env(plots=True, tables=True):
    config = [
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std', 'legend': 'SND-VIC'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3', 'legend': 'seer asym v3'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3delta', 'legend': 'seer asym v3 delta'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v4', 'legend': 'seer asym v4'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m1', 'legend': 'seer v5 model 1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m2', 'legend': 'seer v5 model 2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m3', 'legend': 'seer v5 model 3'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4', 'legend': 'seer v5 model 4'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4b', 'legend': 'seer v5 model 4M'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a0', 'legend': 'seer v5 model A0'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a1', 'legend': 'seer v5 model A1'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a2', 'legend': 'seer v5 model A2'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a3', 'legend': 'seer v5 model A3'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m5', 'legend': 'seer v5 model 5'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m6', 'legend': 'seer v5 model 6'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m7', 'legend': 'seer v5 model 7'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m9', 'legend': 'seer v5 model 9'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m10', 'legend': 'seer v5 model 10'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m11', 'legend': 'seer v5 model 11'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m12', 'legend': 'seer v5 model 12'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m13', 'legend': 'seer v5 model 13'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m2f1', 'legend': 'seer v5 h256 f01'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m2f2', 'legend': 'seer v5 h256 f005'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v7m1', 'legend': 'seer v7 model 1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v7m2', 'legend': 'seer v7 model 2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v9m1', 'legend': 'seer v9 model 2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'a2', 'id': 'asym', 'legend': 'a2 asym'},
    ]

    compute_table_values(config, keys=['re'])
    plot('montezuma_v5m', config, labels=['external reward', 'intrinsic reward', 'forward reward', 'forward space', 'forward target space'], keys=['re', 'ri', 'forward_reward', 'forward_space', 'target_space'], plot_details=['seer_asym_v5m4a2'], window=10000)
    # plot('montezuma_seer', config, labels=['external reward', 'score', 'forward model loss', 'forward reward'], keys=['re', 'score', 'loss_forward', 'forward_reward'], plot_details=[], window=10000)


def procgen_env(plots=True, tables=True):
    key = 're'

    config = [
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('climber', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)

    config = [
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('caveflyer', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)

    config = [
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('coinrun', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)

    config = [
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline', 'mode': 'mch'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('jumper', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)


if __name__ == '__main__':
    atari_env(plots=True)
    # procgen_env(plots=True)
