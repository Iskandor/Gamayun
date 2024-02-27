from modules.PPO_AtariModules import PPOAtariMotivationNetwork
from modules.encoders.EncoderAtari import VICRegEncoderAtari, VICRegEncoderAtariV2
from modules.novelty_models.RNDModelAtari import BarlowTwinsModelAtari, VICRegModelAtari, SpacVICRegModelAtari, STDModelAtari, SNDVModelAtari, VINVModelAtari, TPModelAtari, AMIModelAtari


class PPOAtariNetworkSND(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkSND, self).__init__(config)
        if config.type == 'bt':
            self.cnd_model = BarlowTwinsModelAtari(config)
        if config.type == 'vicreg':
            self.cnd_model = VICRegModelAtari(config, encoder_class=VICRegEncoderAtari)
        if config.type == 'vicreg2':
            self.cnd_model = VICRegModelAtari(config, encoder_class=VICRegEncoderAtariV2)
        if config.type == 'spacvicreg':
            self.cnd_model = SpacVICRegModelAtari(config)
        if config.type == 'st-dim':
            self.cnd_model = STDModelAtari(config)
        if config.type == 'vanilla':
            self.cnd_model = SNDVModelAtari(config)
        if config.type == 'vinv':
            self.cnd_model = VINVModelAtari(config)
        if config.type == 'tp':
            self.cnd_model = TPModelAtari(config)
        if config.type == 'ami':
            self.cnd_model = AMIModelAtari(config)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))

        return value, action, probs, features
