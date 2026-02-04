import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge, AtariStateEncoderLarge2, AtariStateEncoderLarge2Heads
from modules.forward_models import ForwardModel, HiddenModel, NoiseModel

class PPOAtariFMNetwork(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)


class PPOAtariSTDIMNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.ppo_encoder = AtariStateEncoderLarge2Heads(self.input_shape, self.feature_dim)
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
            encoded_state = self.ppo_encoder.forward_motivation(state)
            encoded_next_state = self.ppo_encoder.forward_motivation(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder.forward_motivation(state, fmaps=True)
            map_next_state = self.ppo_encoder.forward_motivation(next_state, fmaps=True)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1))

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            _, probs_real = self.actor(map_next_state_detached)
            probs_real = probs_real.detach() 
            _, probs_pred = self.actor(predicted_next_state_detached)
            probs_pred = probs_pred.detach()

            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model, probs_real, probs_pred
        

class PPOAtariSTDIMMultiStepNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.horizon = config.motivation_horizon
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

    def forward(self, state=None, action=None, next_states=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            z = self.ppo_encoder(state)
            z = F.normalize(z, p=2, dim=1) 
            value, action, probs = super().forward(z)
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = F.normalize(self.ppo_encoder(state), p=2, dim=1)
            encoded_next_state = F.normalize(self.ppo_encoder(next_states), p=2, dim=1)
            delta = self.forward_model(torch.cat([encoded_state, action], dim=1))
            predicted_next_state = F.normalize(encoded_state + delta, p=2, dim=1)
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            predictions = []
            targets = []

            map_state = self.ppo_encoder(state, fmaps=True)
            initial_z = F.normalize(map_state['out'], p=2, dim=1)

            map_state['out'] = initial_z 
            map_next_state = self.ppo_encoder(next_states[:, 0], fmaps=True)
            target_t1 = F.normalize(map_next_state['out'], p=2, dim=1)
            map_next_state['out'] = target_t1

            delta_0 = self.forward_model(torch.cat([initial_z, action[:, 0]], dim=1))
            current_z = F.normalize(initial_z + delta_0, p=2, dim=1)
            predictions.append(current_z)
            targets.append(target_t1) 

            for i in range(1, self.horizon):
                raw_next_z = self.ppo_encoder(next_states[:, i], fmaps=False)
                map_next_state_z = F.normalize(raw_next_z, p=2, dim=1).detach()
                delta_i = self.forward_model(torch.cat([current_z, action[:, i]], dim=1))
                current_z = F.normalize(current_z + delta_i, p=2, dim=1)
                
                predictions.append(current_z)
                targets.append(map_next_state_z)

            map_state_detached = initial_z.detach()
            map_next_state_detached = targets[0].detach()
            predicted_next_state_detached = predictions[0].detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            
            _, probs_real = self.actor(map_next_state_detached)
            probs_real = probs_real.detach() 
            _, probs_pred = self.actor(predicted_next_state_detached)
            probs_pred = probs_pred.detach() 

            return map_state, map_next_state, predictions, targets, action_encoder, action_forward_model, probs_real, probs_pred 


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

        self.ppo_encoder = AtariStateEncoderLarge2Heads(self.input_shape, self.feature_dim)
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
            value, action, probs = super().forward(self.ppo_encoder.forward_motivation(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder.forward_motivation(state)
            encoded_next_state = self.ppo_encoder.forward_motivation(next_state)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1)) + action_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder.forward_motivation(state, fmaps=True)
            map_next_state = self.ppo_encoder.forward_motivation(next_state, fmaps=True)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1)) + action_state

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            _, probs_real = self.actor(map_next_state_detached)
            probs_real = probs_real.detach() 
            _, probs_pred = self.actor(predicted_next_state_detached)
            probs_pred = probs_pred.detach() 

            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model, probs_real, probs_pred


class PPOAtariSTDIMLinearNetworkWithActionPopulationEmbeddingProjection(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.action_proj = ForwardModel.PopulationActionEmbedding(num_actions=config.action_dim, feature_dim=config.feature_dim)

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
            action_state = self.action_proj(action.argmax(dim=-1))
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1)) + action_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            action_state = self.action_proj(action.argmax(dim=-1))
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1)) + action_state

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model
        

class PPOAtariSTDIMLinearNetworkWithActionProjection2(PPOAtariFMNetwork):
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
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1)) + action_state + encoded_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1)) + action_state + map_state['out']

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model
        

class PPOAtariSTDIMTrulyLinearNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.ppo_encoder = AtariStateEncoderLarge2Heads(self.input_shape, self.feature_dim)
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
            value, action, probs = super().forward(self.ppo_encoder.forward_motivation(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder.forward_motivation(state)
            encoded_next_state = self.ppo_encoder.forward_motivation(next_state)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            dynamic_change = (encoded_state + action_state) - encoded_next_state
            return encoded_state, dynamic_change, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder.forward_motivation(state, fmaps=True)
            map_next_state = self.ppo_encoder.forward_motivation(next_state, fmaps=True)
            action_state = self.action_proj(action)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1))
            dynamic_change = (map_state['out'] + action_state) - map_next_state['out']

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            _, probs_real = self.actor(map_next_state_detached)
            _, probs_pred = self.actor(predicted_next_state_detached)

            return map_state, map_next_state, predicted_next_state, dynamic_change, action_encoder, action_forward_model, probs_real, probs_pred
        

class PPOAtariSTDIMLinearNoiseNetwork(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim
        self.noise_dim = config.noise_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        
        self.noise_generator = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.feature_dim)
        )
        nn.init.uniform_(self.noise_generator[-1].weight, -0.01, 0.01)
    
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
            noise = self.noise_generator(torch.cat([encoded_state, action], dim=1))
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action, noise], dim=1)) + encoded_state
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            noise = self.noise_generator(torch.cat([map_state['out'], action], dim=1))
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action, noise], dim=1)) + map_state['out']

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model, noise


class PPOAtariSTDIMLinearNoiseNetworkWithNoiseResidual(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type, noise_generetor_type, encoder_type=1):
        super().__init__(config)
        self.semanticLossOn = config.semanticLossOn
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim
        self.noise_dim = config.noise_dim

        if encoder_type == 1:
            self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        elif encoder_type == 2:
            self.ppo_encoder = AtariStateEncoderLarge2Heads(self.input_shape, self.feature_dim)
        else:
            self.ppo_encoder = AtariStateEncoderLarge2(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.noise_generator = NoiseModel.chooseModel(config, noise_generetor_type)
    
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
            noise = self.noise_generator(torch.cat([encoded_state, action], dim=1))
            predicted_next_state = encoded_state + self.forward_model(torch.cat([encoded_state, action], dim=1)) + noise
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            noise = self.noise_generator(torch.cat([map_state['out'], action], dim=1))
            predicted_next_state = map_state['out'] + self.forward_model(torch.cat([map_state['out'], action], dim=1)) + noise

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            _, probs_real = self.actor(map_next_state_detached)
            probs_real = probs_real.detach() 
            
            if self.semanticLossOn:
                _, probs_pred = self.actor(predicted_next_state)
            else:
                _, probs_pred = self.actor(predicted_next_state_detached)

            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model, noise, probs_real, probs_pred 


class PPOAtariSTDIMLinearNoiseNetworkWithActionEmbedding(PPOAtariFMNetwork):
    def __init__(self, config, forward_model_type):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim
        self.noise_dim = config.noise_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = ForwardModel.chooseModel(config, forward_model_type)
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.feature_dim)
        )
        
        self.noise_generator = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.feature_dim)
        )
        nn.init.uniform_(self.noise_generator[-1].weight, -0.01, 0.01)
    
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
            noise = self.noise_generator(torch.cat([encoded_state, action], dim=1))
            predicted_next_state = self.forward_model(encoded_state) + action_state + noise
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            action_state = self.action_proj(action)
            noise = self.noise_generator(torch.cat([map_state['out'], action], dim=1))
            predicted_next_state = self.forward_model(map_state['out']) + action_state + noise

            map_state_detached = map_state['out'].detach()
            map_next_state_detached = map_next_state['out'].detach()
            predicted_next_state_detached = predicted_next_state.detach()
            action_encoder = self.inverse_model(torch.cat([map_state_detached, map_next_state_detached], dim=1))
            action_forward_model = self.inverse_model(torch.cat([map_state_detached, predicted_next_state_detached], dim=1))
            return map_state, map_next_state, predicted_next_state, action_encoder, action_forward_model, noise


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