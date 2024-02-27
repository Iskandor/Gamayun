import torch

from agents.PPOAgent import PPOAgentBase, AgentMode
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.PPO_AtariModules import PPOAtariNetwork


class PPOAtariAgent(PPOAgentBase):
    class AgentState:
        def __init__(self):
            self.state = None
            self.action = None
            self.probs = None
            self.value = None

            self.next_state = None
            self.reward = None
            self.done = None
            self.trunc = None
            self.info = None

    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetwork(config).to(config.device)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)

    def _encode_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.config.device)

    @staticmethod
    def _convert_action(action):
        a = torch.argmax(action, dim=1).numpy()
        return a

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('score', ['sum'], 'score', 0),
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs = self.model(state)
        next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))

        self._check_terminal_states(env, mode, done, next_state)

        next_state = self._encode_state(next_state)
        reward = torch.tensor(reward, dtype=torch.float32)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        self.analytics.update(re=reward, score=score)
        self.analytics.end_step()

        if mode == AgentMode.TRAINING:
            self.memory.add(state=state.cpu(),
                            value=value.cpu(),
                            action=action.cpu(),
                            prob=probs.cpu(),
                            reward=reward.cpu(),
                            mask=done.cpu())
            indices = self.memory.indices()
            self.algorithm.train(self.memory, indices)

            if indices is not None:
                self.memory.clear()

        return next_state
