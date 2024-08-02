import time
from collections import Counter

import torch
from tqdm import tqdm

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.DPMLoss import DPMLoss
from modules.PPO_Modules import ActivationStage
from modules.atari.PPOAtariNetworkDPM import PPOAtariNetworkDPM
from motivation.DPMMotivation import DPMMotivation
import numpy as np


class PPOAtariDPMAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkDPM(config).to(config.device)
        self.motivation = DPMMotivation(self.model,
                                        DPMLoss(config, self.model),
                                        config.motivation_lr,
                                        config.motivation_scale,
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('ppo_space', ['mean', 'std'], 'PPO space', 0),
            ('target_space', ['mean', 'std'], 'target space', 0),
            ('forward_space', ['mean', 'std'], 'forward space', 0),
            ('distillation_error', ['mean', 'std'], 'distillation error', 0),
            ('prediction_error', ['mean', 'std'], 'prediction error', 0)
        ]
        info = InfoCollector(trial, self.step_counter, self.reward_avg, info_points)

        return info

    def _initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env,
                      re=(1,),
                      score=(1,),
                      ri=(1,),
                      ppo_space=(1,),
                      target_space=(1,),
                      forward_space=(1,),
                      distillation_error=(1,),
                      prediction_error=(1,)
                      )
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs, ppo_features = self.model(state)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)
            next_state = self._encode_state(next_state)
            next_state = self.state_average.process(next_state).clip_(-4., 4.)
            if mode == AgentMode.TRAINING:
                self.state_average.update(next_state)

            z_state, pz_state, z_next_state, pz_next_state = self.model(state, action, next_state, stage=ActivationStage.MOTIVATION_INFERENCE)
            int_reward, distillation_error, prediction_error = self.motivation.reward(z_state, pz_state, z_next_state, pz_next_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        ppo_space = torch.cdist(ppo_features, ppo_features).mean(dim=1, keepdim=True)
        target_space = torch.cdist(z_state, z_state).mean(dim=1, keepdim=True)
        forward_space = torch.cdist(z_next_state, z_next_state).mean(dim=1, keepdim=True)

        self.analytics.update(re=ext_reward,
                              ri=int_reward,
                              score=score,
                              ppo_space=ppo_space,
                              target_space=target_space,
                              forward_space=forward_space,
                              distillation_error=distillation_error,
                              prediction_error=prediction_error
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
                self._train(self.memory, indices)
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

    def analytic_loop(self, env, name, task):
        if task == 'collect_states':
            states = []
            actions = []
            next_states = []
            room_ids = []
            state_raw = self._initialize_env(env)
            state = self._encode_state(state_raw)
            stop = False

            while not stop:
                states.append(np.copy(state_raw))
                next_state, done, next_state_raw, action, info = self._collect_state_step(env, state, AgentMode.INFERENCE)
                actions.append(action.cpu().numpy())
                stop = done.item() == 0.

                state = next_state
                state_raw = next_state_raw
                next_states.append(np.copy(next_state_raw))
                room_ids.append(info['room_id'].item())

            print('Explored rooms: ', len(Counter(room_ids).values()))

            np.save('{0:s}_{1:s}{2:s}'.format(name, task, '.npy'),
                    {
                        'states': states,
                        'actions': actions,
                        'next_states': next_states,
                        'room_ids': room_ids,
                    })

        if task == 'collect_representations':
            data = np.load('{0:s}_{1:s}{2:s}'.format(name, 'collect_states', '.npy'), allow_pickle=True).item()
            states, actions, next_states, room_ids = data['states'], data['actions'], data['next_states'], data['room_ids']
            n = len(states)

            zppo_state = []
            zt_state = []
            zf_next_state = []

            for state, action, next_state in tqdm(zip(states, actions, next_states), total=n):
                ppo_features, z_state, z_next_state = self._collect_representation_step(state, action, next_state)
                zppo_state.append(ppo_features.cpu().numpy())
                zt_state.append(z_state.cpu().numpy())
                zf_next_state.append(z_next_state.cpu().numpy())

            np.save('{0:s}_{1:s}{2:s}'.format(name, task, '.npy'),
                    {
                        'zppo_state': zppo_state,
                        'zt_state': zt_state,
                        'zf_next_state': zf_next_state,
                        'room_ids': room_ids,
                    })

        env.close()

    def _collect_state_step(self, env, state, mode):
        with torch.no_grad():
            _, action, _, _ = self.model(state)
            next_state_raw, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state_raw)

            next_state = self._encode_state(next_state_raw)

        done = torch.tensor(1 - done, dtype=torch.float32)

        return next_state, done, next_state_raw, action, info

    def _collect_representation_step(self, state, action, next_state):
        state = self._encode_state(state)
        next_state = self._encode_state(next_state)
        action = torch.tensor(action, device=self.config.device)

        with torch.no_grad():
            _, _, _, ppo_features = self.model(state)
            z_state, pz_state, z_next_state, pz_next_state = self.model(state, action, next_state, stage=ActivationStage.MOTIVATION_INFERENCE)

        return ppo_features, z_state, z_next_state

    def _train(self, memory, indices):
        start = time.time()

        n = self.config.trajectory_size // self.config.batch_size
        self.ppo.prepare(memory, indices)
        self.motivation.prepare(memory, indices)
        im_batch_size = self.config.batch_size // self.config.ppo_epochs
        im_states, im_actions, im_next_states = self.motivation.batches(im_batch_size)

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

                im_loss = self.motivation.loss(
                    im_states[i + epochs * n].to(self.config.device),
                    im_actions[i + epochs * n].to(self.config.device),
                    im_next_states[i + epochs * n].to(self.config.device),
                )

                loss = ppo_loss + im_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        end = time.time()
        print("DPM agent training: trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self.config.trajectory_size, self.config.batch_size, self.config.ppo_epochs, end - start))
