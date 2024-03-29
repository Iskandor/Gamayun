import time

import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.SEERLoss import SEERLoss
from modules.atari.PPOAtariNetworkSEER import PPOAtariNetworkSEER
from motivation.SEERMotivation import SEERMotivation


class PPOAtariSEERAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSEER(config).to(config.device)
        self.motivation = SEERMotivation(self.model, SEERLoss(config, self.model), config.motivation_lr, config.distillation_scale, config.forward_scale, config.forward_threshold, config.device)
        self.algorithm = PPO(self.model, self._ppo_eval, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('target_space', ['mean', 'std'], 'target space', 0),
            ('learned_space', ['mean', 'std'], 'learned space', 0),
            ('forward_space', ['mean', 'std'], 'forward space', 0),
            ('hidden_space', ['mean', 'std'], 'hidden space', 0),
            ('next_space', ['mean', 'std'], 'next space', 0),
            ('distillation_reward', ['mean', 'std'], 'dist. error', 0),
            ('forward_reward', ['mean', 'std'], 'forward error', 0),
            ('confidence', ['mean', 'std'], 'confidence', 0),
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env,
                      re=(1,),
                      score=(1,),
                      ri=(1,),
                      target_space=(1,),
                      next_space=(1,),
                      learned_space=(1,),
                      forward_space=(1,),
                      hidden_space=(1,),
                      distillation_reward=(1,),
                      forward_reward=(1,),
                      confidence=(1,)
                      )
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs, zt_state, zl_state = self.model(state=state, stage=0)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)

            next_state = self._encode_state(next_state)
            next_state = self.state_average.process(next_state).clip_(-4., 4.)
            self.state_average.update(next_state)

            p_state, z_next_state, h_next_state, p_next_state = self.model(zl_state=zl_state, action=action, next_state=next_state, stage=1)
            int_reward, distillation_error, forward_error, confidence = self.motivation.reward(zt_state, p_state, z_next_state, h_next_state, p_next_state)

        # state_norm = torch.norm(state, p=2, dim=[1, 2, 3])
        # next_state_norm = torch.norm(next_state, p=2, dim=[1, 2, 3])

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        target_features = torch.norm(zt_state, p=2, dim=1, keepdim=True)
        learned_features = torch.norm(zl_state, p=2, dim=1, keepdim=True)
        forward_features = torch.norm(p_next_state, p=2, dim=1, keepdim=True)
        hidden_features = torch.norm(h_next_state, p=2, dim=1, keepdim=True)
        next_features = torch.norm(z_next_state, p=2, dim=1, keepdim=True)
        self.analytics.update(
            re=ext_reward,
            ri=int_reward,
            score=score,
            target_space=target_features,
            next_space=next_features,
            learned_space=learned_features,
            forward_space=forward_features,
            hidden_space=hidden_features,
            distillation_reward=distillation_error,
            forward_reward=forward_error,
            confidence=confidence,
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
                self.algorithm.train(self.memory, indices)
                self.motivation.train(self.memory, indices)
                self.memory.clear()

        return next_state

    def _ppo_eval(self, state):
        value, _, probs, _, _ = self.model(state=state, stage=0)
        return value, probs
