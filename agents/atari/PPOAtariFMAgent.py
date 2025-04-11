import torch

from enum import Enum
from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.FMLoss import STDIMLoss, IJEPALoss
from modules.atari.PPOAtariFMNetwork import PPOAtariSTDIMNetwork, PPOAtariIJEPANetwork
from motivation.FMMotivation import FMMotivation
from utils.StateNorm import ExponentialDecayNorm
from modules.PPO_Modules import ActivationStage


class ArchitectureType(Enum):
    ST_DIM = 0
    I_JEPA = 1


class PPOAtariFMAgent(PPOAtariAgent):
    def __init__(self, config, _type=0):
        super().__init__(config)
        model_class, loss_class = self._set_up(config, _type)
        self.model = model_class
        self.motivation = FMMotivation(self.model,
                                       loss_class,
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

    @staticmethod
    def _set_up(config, _type=ArchitectureType.ST_DIM):
        if _type == ArchitectureType.ST_DIM:
            model_class = PPOAtariSTDIMNetwork(config).to(config.device)
            loss_class = STDIMLoss(model_class,
                                   model_class.ppo_encoder.hidden_size,
                                   model_class.ppo_encoder.local_layer_depth,
                                   config.device)
        else:
            model_class = PPOAtariIJEPANetwork(config).to(config.device)
            loss_class = IJEPALoss(model_class, config.device)

        return model_class, loss_class

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('feature_space', ['mean', 'std'], 'feature space', 0),
            ('loss', ['mean', 'std', 'max'], 'loss', 0),
            ('norm_loss', ['mean', 'std', 'max'], 'norm_loss', 0),
            ('fwd_loss', ['mean', 'std', 'max'], 'fwd_loss', 0),
            ('total_loss', ['mean', 'std', 'max'], 'total_loss', 0)
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), ri=(1,), score=(1,), feature_space=(1,), loss=(1,), norm_loss=(1,), fwd_loss=(1,), total_loss=(1,))
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
            int_reward = self.motivation.reward(z_next_state, p_next_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        feature_dist = torch.cdist(z_state, z_state).mean(dim=1, keepdim=True)
        self.analytics.update(
            re=ext_reward,
            ri=int_reward,
            score=score,
            feature_space=feature_dist
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