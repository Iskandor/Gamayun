import torch.nn as nn

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderCrafter import CrafterStateEncoder


class PPOCrafterNetworkSND(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOCrafterNetworkSND, self).__init__(config)

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.ppo_feature_dim = config.ppo_feature_dim
        n_kernels = [32, 64, 128]

        self.ppo_encoder = CrafterStateEncoder(self.input_shape, self.ppo_feature_dim, n_kernels)
        self.target_model = CrafterStateEncoder(self.input_shape, self.feature_dim, n_kernels)
        self.learned_model = CrafterStateEncoder(self.input_shape, self.feature_dim, n_kernels)

        self.learned_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 0.5
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], 0.01)

    def forward(self, state=None, next_state=None, stage=ActivationStage.INFERENCE):
        zt_state = self.target_model(state)
        pzt_state = self.learned_projection(self.learned_model(state))

        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))

            return value, action, probs, zt_state, pzt_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            zt_next_state = self.target_model(next_state)
            return zt_state, pzt_state, zt_next_state
