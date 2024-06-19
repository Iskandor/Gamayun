import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.atari.PPOAtariNetworkRND import PPOAtariNetworkRND
from motivation.RNDMotivation import RNDMotivation


class PPOAtariRNDAgent(PPOAtariAgent):
    def __init__(self, config,):
        super().__init__(config)
        self.model = PPOAtariNetworkRND(config).to(config.device)
        self.motivation = RNDMotivation(self.model.rnd_model, config.motivation_lr, config.motivation_scale, config.device)
        self.ppo = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,), ri=(1,))
        # metric = NoveltyMetric(self.network.input_shape[1],
        #                        self.network.input_shape[2],
        #                        NoveltyMetric.Greyscale,
        #                        self.network.rnd_model.learned_model,
        #                        self.network.rnd_model.target_model,
        #                        self.config.batch_size,
        #                        self.config.device)
        # analysis.add_metric(metric)
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs = self.model(state)
            int_reward = self.motivation.reward(state)
        next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))

        self._check_terminal_states(env, mode, done, next_state)

        next_state = self._encode_state(next_state)
        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        self.analytics.update(re=ext_reward, ri=int_reward, score=score)
        self.analytics.end_step()

        if mode == AgentMode.TRAINING:
            self.memory.add(state=state.cpu(),
                            value=value.cpu(),
                            action=action.cpu(),
                            prob=probs.cpu(),
                            reward=reward.cpu(),
                            mask=done.cpu())
            indices = self.memory.indices()
            self.ppo.train(self.memory, indices)
            self.motivation.train(self.memory, indices)

            if indices is not None:
                self.memory.clear()

        return next_state
