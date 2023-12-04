from plots import plot
from plots.analytic_table import compute_table_values
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def atari_env(plots=True, tables=True):
    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'small_static', 'legend': 'small static'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_static', 'legend': 'big static'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'resnet_static', 'legend': 'resnet static'},
    ]

    if plots:
        plot('montezuma_models', config, labels=['external reward', 'score', 'intrinsic reward', 'features'], keys=['re', 'score', 'ri', 'feature_space'], plot_details=[], window=10000)

    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_static', 'legend': 'big static'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_random', 'legend': 'big random'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_full_random', 'legend': 'big full random'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_full_random_aug', 'legend': 'big full random aug'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std_aug2', 'legend': 'big full random aug 2'},
    ]

    if plots:
        plot('montezuma_aug', config, labels=['external reward', 'score', 'intrinsic reward', 'features'], keys=['re', 'score', 'ri', 'feature_space'], plot_details=[], window=10000)

    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_static', 'legend': 'big static'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std_projector_256', 'legend': 'big static projector 256'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_static_projector_1024', 'legend': 'big static projector 1024'},
    ]

    if plots:
        plot('montezuma_projector', config, labels=['external reward', 'score', 'intrinsic reward', 'features'], keys=['re', 'score', 'ri', 'feature_space'], plot_details=[], window=10000)

    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_full_random', 'legend': 'big full random'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_full_random_gelu', 'legend': 'big full random gelu'},
    ]

    if plots:
        plot('montezuma_gelu', config, labels=['external reward', 'score', 'intrinsic reward', 'features'], keys=['re', 'score', 'ri', 'feature_space'], plot_details=[], window=10000)

    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std', 'legend': 'std2'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'big_full_random', 'legend': 'std'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'snd', 'id': 'std_spac', 'legend': 'std spatial'},
    ]

    if plots:
        plot('montezuma_spac', config, labels=['external reward', 'score', 'intrinsic reward', 'features'], keys=['re', 'score', 'ri', 'feature_space'], plot_details=[], window=10000)



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
