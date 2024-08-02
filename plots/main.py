import os

from plots import plot
from plots.analytic_table import compute_table_values

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def atari_env(plots=True, tables=True):
    config_dpm = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std', 'legend': 'SND-VIC'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'dpm', 'id': 'v1m1h4', 'legend': 'DPM v1m1h4'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'dpm', 'id': 'v1m2h4', 'legend': 'DPM v1m2h4'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'dpm', 'id': 'v1m3h4', 'legend': 'DPM v1m3h4'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3', 'legend': 'seer asym v3'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v3delta', 'legend': 'seer asym v3 delta'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v4', 'legend': 'seer asym v4'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m1', 'legend': 'seer v5 model 1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m2', 'legend': 'seer v5 model 2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m3', 'legend': 'seer v5 model 3'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4', 'legend': 'seer v5 model 4'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4f1', 'legend': 'seer v5 model 4 f1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4b', 'legend': 'seer v5 model 4M'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a0', 'legend': 'seer v5 model A0'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a1', 'legend': 'seer v5 model A1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a2', 'legend': 'seer v5 model A2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'seer', 'id': 'seer_asym_v5m4a3', 'legend': 'seer v5 model A3'},
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
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'a2', 'id': 'sym_v1h1', 'legend': 'a2 sym scale 1'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'a2', 'id': 'sym_v1h2', 'legend': 'a2 sym scale 0.5'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'a2', 'id': 'sym_v1h3', 'legend': 'a2 sym scale 0.25'},
    ]

    config_sndv2 = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'baseline8M', 'legend': 'SND-VIC'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_m1', 'legend': 'SNDv2 m1'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_m2', 'legend': 'SNDv2 m2'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_m3', 'legend': 'SNDv2 m3'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_m4', 'legend': 'SNDv2 m4'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_v2m1', 'legend': 'SNDv2 v2m1'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_v3m1', 'legend': 'SNDv2 v3m1'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'sndv2', 'id': 'sndv2_v3m2', 'legend': 'SNDv2 v3m2'},
    ]

    # compute_table_values(config_sndv2, keys=['re'])
    # plot('montezuma_v5m', config, labels=['external reward', 'intrinsic reward', 'forward reward', 'forward space', 'forward target space'], keys=['re', 'ri', 'forward_reward', 'forward_space', 'target_space'], plot_details=['seer_asym_v5m4a2'], window=10000)
    # plot('montezuma_a2', config, labels=['external reward', 'score', 'intrinsic reward'], keys=['re', 'score', 'ri'], plot_details=['sym_v1h1', 'sym_v1h2', 'sym_v1h3'], window=10000)
    # plot('montezuma_dpm', config_dpm, labels=['external reward', 'score', 'intrinsic reward'], keys=['re', 'score', 'ri'], plot_details=['v1m3h4'], window=10000)
    plot('montezuma_sndv2', config_sndv2, labels=['external reward', 'score', 'intrinsic reward'], keys=['re', 'score', 'ri'], plot_details=['sndv2_v3m2'], window=10000)


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
