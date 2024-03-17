import os

from plots import plot
from plots.analytic_table import compute_table_values

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def atari_env(plots=True, tables=True):
    config = [
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v2', 'legend': 'seer asym v2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3', 'legend': 'seer asym v3'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3delta', 'legend': 'seer asym v3 delta'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v4', 'legend': 'seer asym v4'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5', 'legend': 'seer v5 512 h64'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5-1', 'legend': 'seer v5 2048 h64'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5-2', 'legend': 'seer v5 4096 h256'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'a2', 'id': 'asym', 'legend': 'a2 asym'},
    ]

    plot('montezuma_seer', config, labels=['external reward', 'score', 'forward model loss'], keys=['re', 'score', 'loss_forward'], plot_details=['seer_asym_v5-2'], window=10000)


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
