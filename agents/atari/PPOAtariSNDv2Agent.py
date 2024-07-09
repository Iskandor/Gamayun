import time

import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.atari.PPOAtariNetworkSNDv2 import PPOAtariNetworkSNDv2
from motivation.SNDMotivation import SNDMotivationFactory


class PPOAtariSNDv2Agent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSNDv2(config).to(config.device)
        self.motivation = SNDMotivationFactory.get_motivation(config,
                                                              self.model
                                                              )
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

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
        analysis.init(self.config.n_env,
                      re=(1,),
                      score=(1,),
                      ri=(1,),
                      feature_space=(1,))
        return analysis

    def _step(self, env, trial, state, mode):
        state = self.state_average.process(state).clip_(-4., 4.)
        self.state_average.update(state)

        with torch.no_grad():
            value, action, probs, z_state, pz_state = self.model(state)
            int_reward, distillation_error = self.motivation.reward(z_state, pz_state)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)
            next_state = self._encode_state(next_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        feature_dist = torch.cdist(z_state, z_state).mean(dim=1, keepdim=True)

        self.analytics.update(re=ext_reward,
                              ri=int_reward,
                              score=score,
                              feature_space=feature_dist)
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
                loss = self.ppo.loss(
                    new_values,
                    new_probs,
                    ref_values[i].to(self.config.device),
                    adv_values[i].to(self.config.device),
                    actions[i].to(self.config.device),
                    probs[i].to(self.config.device))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.motivation.prepare(memory, indices)
        snd_states, snd_next_states = self.motivation.batches(self.config.batch_size)

        for i in range(n):
            self.optimizer.zero_grad()
            loss = self.motivation.loss(snd_states[i].to(self.config.device),
                                        snd_next_states[i].to(self.config.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        end = time.time()
        print("SND agent training: trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self.config.trajectory_size, self.config.batch_size, self.config.ppo_epochs, end - start))

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
