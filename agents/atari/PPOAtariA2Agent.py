import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.A2Loss import A2Loss
from modules.atari.PPOAtariNetworkA2 import PPOAtariNetworkA2
from motivation.A2Motivation import A2Motivation


class PPOAtariA2Agent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkA2(config).to(config.device)
        self.motivation = A2Motivation(self.model,
                                       A2Loss(config, self.model),
                                       config.motivation_lr,
                                       config.motivation_scale,
                                       config.device)

        self.algorithm = PPO(self.model,
                             self._ppo_eval,
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

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('za_state_norm', ['mean', 'std'], 'feature space A norm', 0),
            ('zb_state_norm', ['mean', 'std'], 'feature space B norm', 0),
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env,
                      re=(1,),
                      score=(1,),
                      ri=(1,),
                      za_state_norm=(1,),
                      zb_state_norm=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        state = self.state_average.process(state).clip_(-4., 4.)
        if mode == AgentMode.TRAINING:
            self.state_average.update(state)

        with torch.no_grad():
            value, action, probs = self.model(state)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)

            next_state = self._encode_state(next_state)

            za_state, zb_state, _, _, pa_state, pb_state = self.model(state, loss=True)
            int_reward = self.motivation.reward(za_state, zb_state, pa_state, pb_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        za_state_norm = torch.norm(za_state, p=2, dim=1, keepdim=True)
        zb_state_norm = torch.norm(zb_state, p=2, dim=1, keepdim=True)
        self.analytics.update(re=ext_reward,
                              ri=int_reward,
                              score=score,
                              za_state_norm=za_state_norm,
                              zb_state_norm=zb_state_norm)
        self.analytics.end_step()

        if mode == AgentMode.TRAINING:
            self.memory.add(state=state.cpu(),
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

        return next_state, done

    def save(self, path):
        state = {
            'model_state': self.model.state_dict(),
            'state_average': self.state_average.get_state(),
        }
        torch.save(state, path + '.pth')

    def load(self, path):
        state = torch.load(path + '.pth', map_location='cpu')

        self.model.load_state_dict(state['model_state'])
        self.state_average.set_state(state['state_average'])
