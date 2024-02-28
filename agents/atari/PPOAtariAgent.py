import time

import torch

from agents.PPOAgent import PPOAgentBase, AgentMode
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.PPO_AtariModules import PPOAtariNetwork
from utils.RunningAverage import RunningStatsSimple


class PPOAtariAgent(PPOAgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetwork(config).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)

        self.state_average = RunningStatsSimple(config.input_shape, config.device)

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
        state = self.state_average.process(state)
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
                # self._train(self.memory, indices)
                self.memory.clear()

        return next_state

    # Experimental feature
    def _train(self, memory, indices):
        start = time.time()
        sample = memory.sample(indices, False)

        states = sample.state
        values = sample.value
        actions = sample.action
        probs = sample.prob
        rewards = sample.reward
        dones = sample.mask

        ref_values, adv_values = self.algorithm.calc_advantage(values, rewards, dones, self.config.gamma, self.config.n_env)

        permutation = torch.randperm(self.config.trajectory_size)

        states = states.reshape(-1, *states.shape[2:])[permutation].reshape(-1, self.config.batch_size, *states.shape[2:])
        actions = actions.reshape(-1, *actions.shape[2:])[permutation].reshape(-1, self.config.batch_size, *actions.shape[2:])
        probs = probs.reshape(-1, *probs.shape[2:])[permutation].reshape(-1, self.config.batch_size, *probs.shape[2:])
        adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation].reshape(-1, self.config.batch_size, *adv_values.shape[2:])
        ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation].reshape(-1, self.config.batch_size, *ref_values.shape[2:])

        n = states.shape[0]

        for epoch in range(self.config.ppo_epochs):
            for i in range(n):
                new_values, _, new_probs = self.model(states[i].to(self.config.device))
                self.optimizer.zero_grad()
                loss = self.algorithm.loss(
                    new_values,
                    new_probs,
                    ref_values[i].to(self.config.device),
                    adv_values[i].to(self.config.device),
                    actions[i].to(self.config.device),
                    probs[i].to(self.config.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        end = time.time()
        print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self.config.trajectory_size, self.config.batch_size, self.config.ppo_epochs, end - start))

