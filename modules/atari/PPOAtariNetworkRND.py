from modules.PPO_AtariModules import PPOAtariMotivationNetwork
from modules.novelty_models.RNDModelAtari import RNDModelAtari


class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkRND, self).__init__(config)
        self.rnd_model = RNDModelAtari(config)
