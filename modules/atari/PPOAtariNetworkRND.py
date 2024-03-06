from modules.PPO_Modules import PPOMotivationNetwork
from modules.novelty_models.RNDModelAtari import RNDModelAtari


class PPOAtariNetworkRND(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkRND, self).__init__(config)
        self.rnd_model = RNDModelAtari(config)
