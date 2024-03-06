from modules.PPO_Modules import PPOMotivationNetwork
from modules.forward_models.ForwardModelAtari import ICMModelAtari


class PPOAtariNetworkICM(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkICM, self).__init__(config)
        self.forward_model = ICMModelAtari(config)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features = self.forward_model.encoder(self.forward_model.preprocess(state))

        return value, action, probs, features