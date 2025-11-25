import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.FMLoss import STDIMLinearLoss, STDIMLoss
from modules.atari.PPOAtariFMNetwork import PPOAtariSTDIMLinearNetwork, PPOAtariSTDIMLinearNetworkWithActionProjection, PPOAtariSTDIMLinearNetworkWithActionProjection2, PPOAtariSTDIMLinearNoiseNetwork
from motivation.FMMotivation import FMMotivation
from utils.StateNorm import ExponentialDecayNorm
from modules.PPO_Modules import ActivationStage
from modules.forward_models.ForwardModel import ForwardModelType

class PPOAtariFMLinearAgent(PPOAtariAgent):
    def __init__(self, config, forward_model_type=ForwardModelType.ForwardModelLinearResidual, type=0):
        super().__init__(config)
        
        if type == 0:
            self.model = PPOAtariSTDIMLinearNetwork(config, forward_model_type).to(config.device)
            self.loss = STDIMLoss(self.model,
                                  self.model.ppo_encoder.hidden_size,
                                  self.model.ppo_encoder.local_layer_depth,
                                  config.device)
        elif type == 1:
            self.model = PPOAtariSTDIMLinearNetworkWithActionProjection(config, forward_model_type).to(config.device)
            self.loss = STDIMLoss(self.model,
                                  self.model.ppo_encoder.hidden_size,
                                  self.model.ppo_encoder.local_layer_depth,
                                  config.device)
        elif type == 2:
            self.model = PPOAtariSTDIMLinearNetworkWithActionProjection2(config, forward_model_type).to(config.device)
            self.loss = STDIMLoss(self.model,
                                  self.model.ppo_encoder.hidden_size,
                                  self.model.ppo_encoder.local_layer_depth,
                                  config.device)
        else:
            self.model = PPOAtariSTDIMLinearNoiseNetwork(config, forward_model_type).to(config.device)
            self.loss = STDIMLinearLoss(self.model,
                                        self.model.ppo_encoder.hidden_size,
                                        self.model.ppo_encoder.local_layer_depth,
                                        config.device)

        self.motivation = FMMotivation(self.model,
                                       self.loss,
                                       config.motivation_lr,
                                       config.eta,
                                       config.device)
        self.ppo = PPO(self.model,
                       config.lr,
                       config.actor_loss_weight,
                       config.critic_loss_weight,
                       config.batch_size,
                       config.trajectory_size,
                       config.beta,
                       config.gamma,
                       ext_adv_scale=2,
                       int_adv_scale=1,
                       ppo_epochs=config.ppo_epochs,
                       n_env=config.n_env,
                       device=config.device,
                       motivation=True)

        #self.hidden_average = ExponentialDecayNorm(config.feature_dim, config.device)

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('feature_space', ['mean', 'std'], 'feature space', 0),
            ('error', ['mean', 'std'], 'error', 0),
            ('loss', ['mean', 'std', 'max'], 'loss', 0),
            ('norm_loss', ['mean', 'std', 'max'], 'norm_loss', 0),
            ('fwd_loss', ['mean', 'std', 'max'], 'fwd_loss', 0),
            ('noise_loss', ['mean', 'std', 'max'], 'noise_loss', 0),
            ('total_loss', ['mean', 'std', 'max'], 'total_loss', 0),
            ('acc_encoder', ['mean', 'std', 'max'], 'acc_encoder', 0),
            ('acc_forward_model', ['mean', 'std', 'max'], 'acc_forward_model', 0)
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), ri=(1,), score=(1,), feature_space=(1,), 
                      error=(1,), loss=(1,), norm_loss=(1,), fwd_loss=(1,), noise_loss=(1,), 
                      total_loss=(1,), acc_encoder=(1,), acc_forward_model=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs = self.model(state, stage=ActivationStage.INFERENCE)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)

            next_state = self._encode_state(next_state)
            #next_state = self.state_average.process(next_state).clip_(-4., 4.)

            #if mode == AgentMode.TRAINING:
            #    self.state_average.update(next_state)

            z_state, z_next_state, p_next_state = self.model(state, action, next_state, stage=ActivationStage.MOTIVATION_INFERENCE)
            error, int_reward = self.motivation.reward(z_next_state, p_next_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        feature_dist = torch.cdist(z_state, z_state).mean(dim=1, keepdim=True)
        self.analytics.update(
            re=ext_reward,
            ri=int_reward,
            score=score,
            feature_space=feature_dist,
            error=error
        )
        self.analytics.end_step()
    

        if mode == AgentMode.TRAINING:
            self.memory.add(state=state.cpu(),
                            next_state=next_state.cpu(),
                            value=value.cpu(),
                            action=action.cpu(),
                            prob=probs.cpu(),
                            reward=reward.cpu(),
                            mask=done.cpu())

            indices = self.memory.indices()

            if indices is not None:
                self.ppo.train(self.memory, indices)
                self.motivation.train(self.memory, indices)
                self.memory.clear()

        return next_state, done