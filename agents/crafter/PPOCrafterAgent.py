import time

import torch
import gymnasium as gym

from agents.PPOAgent import PPOAgentBase, AgentMode
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.carfter.PPOCrafterNetwork import PPOCrafterNetwork
from utils.StateNorm import PreciseNorm, ExponentialDecayNorm


class PPOCrafterAgent(PPOAgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOCrafterNetwork(config).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.ppo = PPO(self.model,
                       config.lr,
                       config.actor_loss_weight,
                       config.critic_loss_weight,
                       config.batch_size,
                       config.trajectory_size,
                       config.beta,
                       config.gamma,
                       ppo_epochs=config.ppo_epochs,
                       n_env=config.n_env,
                       device=config.device,
                       motivation=False)

        self.state_average = PreciseNorm(config.input_shape, config.device)

    def _encode_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.config.device)

    @staticmethod
    def _convert_action(action):
        a = torch.argmax(action, dim=1).numpy()
        return a

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        state = self.state_average.process(state)
        with torch.no_grad():
            value, action, probs = self.model(state)
        next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))

        self._check_terminal_states(env, mode, done, next_state)

        next_state = self._encode_state(next_state)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(1 - done, dtype=torch.float32)

        self.analytics.update(re=reward)
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
                self._train(self.memory, indices)
                self.memory.clear()

        return next_state, done

    def _train(self, memory, indices):
        start = time.time()

        n = self.config.trajectory_size // self.config.batch_size
        self.ppo.prepare(memory, indices)

        for epochs in range(self.config.ppo_epochs):
            states, actions, probs, adv_values, ref_values = self.ppo.batches(self.config.batch_size)
            for i in range(n):
                new_values, new_probs = self.model.ppo_eval(states[i].to(self.config.device))
                self.optimizer.zero_grad()
                ppo_loss = self.ppo.loss(
                    new_values,
                    new_probs,
                    ref_values[i].to(self.config.device),
                    adv_values[i].to(self.config.device),
                    actions[i].to(self.config.device),
                    probs[i].to(self.config.device))

                loss = ppo_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        end = time.time()
        print("Baseline agent training: trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self.config.trajectory_size, self.config.batch_size, self.config.ppo_epochs,
                                                                                                                      end - start))

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

    def inference_loop(self, env, name, trial):
        video_path = name + '.mp4'
        video_recorder = gym.wrappers.RecordVideo(env.envs_list[0], video_path, enabled=video_path is not None)

        self.info = self._initialize_info(trial)
        self.analytics = self._initialize_analysis()

        state = self._encode_state(self._initialize_env(env))
        stop = False

        while not stop:
            env.render(0)
            video_recorder.capture_frame()
            state, done = self._step(env, trial, state, AgentMode.INFERENCE)
            stop = done.item() == 0.
            self.time_estimator.update(self.config.n_env)

        video_recorder.close()

        env.close()
