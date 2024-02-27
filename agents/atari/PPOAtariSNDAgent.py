import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.atari.PPOAtariNetworkSND import PPOAtariNetworkSND
from motivation.SNDMotivation import SNDMotivationFactory


class PPOAtariSNDAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSND(config).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = SNDMotivationFactory.get_motivation(config.type, self.model.cnd_model, config.motivation_lr, config.motivation_scale, config.device)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('feature_space', ['mean', 'std'], 'feature space', 0)
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,), ri=(1,), feature_space=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs, features = self.model(state)
            int_reward = self.motivation.reward(state)
        next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))

        self._check_terminal_states(env, mode, done, next_state)

        next_state = self._encode_state(next_state)
        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        feature_dist = torch.cdist(features, features).mean(dim=1, keepdim=True)
        self.analytics.update(re=ext_reward, ri=int_reward, score=score, feature_space=feature_dist)
        self.analytics.end_step()

        if mode == AgentMode.TRAINING:
            self.memory.add(state=state.cpu(),
                            value=value.cpu(),
                            action=action.cpu(),
                            prob=probs.cpu(),
                            reward=reward.cpu(),
                            mask=done.cpu())

            self.motivation_memory.add(state=state.cpu(),
                                       next_state=next_state.cpu(),
                                       action=action.cpu(),
                                       done=done.cpu())

            indices = self.memory.indices()
            motivation_indices = self.motivation_memory.indices()

            if indices is not None:
                self.algorithm.train(self.memory, indices)
                self.memory.clear()

            if motivation_indices is not None:
                self.motivation.train(self.motivation_memory, motivation_indices)
                self.motivation_memory.clear()

        return next_state
