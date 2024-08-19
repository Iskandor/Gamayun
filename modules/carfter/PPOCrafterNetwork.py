from modules.PPO_Modules import PPONetwork
from modules.encoders.EncoderCrafter import CrafterStateEncoder


class PPOCrafterNetwork(PPONetwork):

    def __init__(self, config):
        super(PPOCrafterNetwork, self).__init__(config)

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.ppo_feature_dim = config.ppo_feature_dim
        n_kernels = [32, 64, 128]

        self.ppo_encoder = CrafterStateEncoder(self.input_shape, self.ppo_feature_dim, n_kernels)

    def forward(self, state):
        ppo_features = self.ppo_encoder(state)
        value, action, probs = super().forward(ppo_features)

        return value, action, probs
