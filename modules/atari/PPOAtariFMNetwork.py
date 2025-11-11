import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge
from modules.forward_models import ForwardModel, HiddenModel

class PPOAtariFMNetwork(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

# What about batch normalization or dynamic tanh (which proved to be as good if not better recently) and small dropout?
# What about activation and gain value sqrt(2).
# What about making the model deeper. What about slowly decreasing the feature size of deeper layers

class PPOAtariSTDIMNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        # TODO: Augmentations need to be tried later
        # self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, activation=nn.GELU, gain=sqrt(2))
        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)

        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1))

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model


class PPOAtariSTDIMLinearNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1)) + encoded_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1)) + map_state['out']

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model


class PPOAtariSTDIMLinearNetworkWithActionProjection(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.feature_dim)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1)) + action_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1)) + action_state

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model


class PPOAtariIJEPANetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type, hidden_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.hidden_model = HiddenModel.chooseModel(config, hidden_model_type)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        # Tu je otázka, či chceme vraciať aj hidden encoded state a počítať reward aj pomocou neho
        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action, hidden_next_state], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action, hidden_next_state], dim=1))

            map_state_detached = encoded_state.detach()
            map_next_state_detached = encoded_next_state.detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state, hidden_next_state, action_encoder, action_forward_model
        

class PPOAtariIJEPAHiddenHeadNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type, hidden_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.hidden_model = HiddenModel.chooseModel(config, hidden_model_type)

        self.proj_hidden_to_z = nn.Linear(self.hidden_dim, self.feature_dim)
        init_orthogonal(self.proj_hidden_to_z, np.sqrt(2))

        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        # Tu je otázka, či chceme vraciať aj hidden encoded state a počítať reward aj pomocou neho
        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action, hidden_next_state], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action, hidden_next_state], dim=1))

            map_state_detached = encoded_state.detach()
            map_next_state_detached = encoded_next_state.detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state, hidden_next_state, action_encoder, action_forward_model
        

class PPOAtariIJEPAEmaEncoderNetwork(PPOAtariFMNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape
        self.ema_m = getattr(config, "ema_m", 0.996)

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.ppo_encoder_target = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.ppo_encoder_target.load_state_dict(self.ppo_encoder.state_dict())
        for p in self.ppo_encoder_target.parameters():
            p.requires_grad = False

       
        self.forward_model = ForwardModel.ForwardModelSkipConnectionDupe(config, 0)
        self.forward_model_hidden = ForwardModel.ForwardModelSkipConnectionDupe(config, self.hidden_dim)

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def ema_update(self):
        m = self.ema_m
        for p_t, p_o in zip(self.ppo_encoder_target.parameters(), self.ppo_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs
        
        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            encoded_state = self.ppo_encoder(state)
            encoded_state_hidden = self.ppo_encoder_target(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            predicted_next_state_hidden = self.forward_model_hidden(torch.cat([encoded_state_hidden, action, encoded_next_state], dim=1))

            map_state_detached = encoded_state.detach()
            map_next_state_detached = encoded_next_state.detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state, predicted_next_state_hidden, action_encoder, action_forward_model
        

class PPOAtariIJEPANetwork2(PPOAtariFMNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.ForwardModelSkipConnectionDupe(config, 0)
        self.forward_model_hidden = ForwardModel.ForwardModelSkipConnectionDupe(config, self.hidden_dim)

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            predicted_next_state_hidden = self.forward_model_hidden(torch.cat([encoded_state, action, encoded_next_state], dim=1))

            map_state_detached = encoded_state.detach()
            map_next_state_detached = encoded_next_state.detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state, predicted_next_state_hidden, action_encoder, action_forward_model