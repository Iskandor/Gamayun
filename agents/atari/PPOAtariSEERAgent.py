import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.SEERLoss import SEERLoss_V1
from modules.PPO_Modules import ActivationStage
from modules.atari.PPOAtariNetworkSEER import PPOAtariNetworkSEER_V1
from motivation.SEERMotivation import SEERMotivation
from utils.StateNorm import ExponentialDecayNorm


class PPOAtariSEERAgent_V1(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSEER_V1(config).to(config.device)
        self.motivation = SEERMotivation(self.model,
                                         SEERLoss_V1(config, self.model),
                                         config.motivation_lr,
                                         config.distillation_scale,
                                         config.forward_scale,
                                         config.forward_threshold,
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

    def _initialize_info(self, trial):
        info_points = [
            ('re', ['sum', 'step'], 'ext. reward', 0),
            ('ri', ['mean', 'std', 'max'], 'int. reward', 0),
            ('score', ['sum'], 'score', 0),
            ('target_space', ['mean', 'std'], 'target space', 0),
            ('distillation_space', ['mean', 'std'], 'distillation space', 0),
            ('forward_space', ['mean', 'std'], 'forward space', 0),
            ('target_forward_space', ['mean', 'std'], 'target forward space', 0),
            ('distillation_reward', ['mean', 'std'], 'dist. error', 0),
            ('forward_reward', ['mean', 'std'], 'forward error', 0),
            ('forward_model_ia', ['mean', 'std'], 'forward model IA', 0),
            ('forward_model_kl', ['mean', 'std'], 'forward model KL', 0)
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
                      distillation_space=(1,),
                      forward_space=(1,),
                      target_forward_space=(1,),
                      hidden_space=(1,),
                      distillation_reward=(1,),
                      forward_reward=(1,),
                      forward_model_ia=(1,),
                      forward_model_kl=(1,),
                      )
        return analysis

    def _step(self, env, trial, state, mode):
        with torch.no_grad():
            value, action, probs = self.model(state, stage=ActivationStage.INFERENCE)
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)

            next_state = self._encode_state(next_state)
            next_state = self.state_average.process(next_state).clip_(-4., 4.)

            if mode == AgentMode.TRAINING:
                self.state_average.update(next_state)

            zt_state, pz_state, z_next_state, pz_next_state, next_probs, p_next_probs, p_action, pp_action = self.model(state, action, next_state, stage=ActivationStage.MOTIVATION_INFERENCE)

            int_reward, distillation_error, forward_error = self.motivation.reward(zt_state, pz_state, z_next_state, pz_next_state)

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        reward = torch.cat([ext_reward, int_reward.cpu()], dim=1)
        score = torch.tensor(info['raw_score']).unsqueeze(-1)
        done = torch.tensor(1 - done, dtype=torch.float32)

        target_features = torch.norm(zt_state, p=2, dim=1, keepdim=True)
        distillation_features = torch.norm(pz_state, p=2, dim=1, keepdim=True)
        forward_features = torch.norm(pz_next_state, p=2, dim=1, keepdim=True)
        target_forward_features = torch.norm(z_next_state, p=2, dim=1, keepdim=True)

        forward_model_ia = (torch.argmax(p_action, dim=1) == torch.argmax(pp_action, dim=1)).sum() / p_action.shape[0]
        forward_model_kl = F.kl_div(torch.log(next_probs), p_next_probs, reduction='batchmean')

        self.analytics.update(
            re=ext_reward,
            ri=int_reward,
            score=score,
            target_space=target_features,
            distillation_space=distillation_features,
            forward_space=forward_features,
            target_forward_space=target_forward_features,
            distillation_reward=distillation_error,
            forward_reward=forward_error,
            forward_model_ia=forward_model_ia,
            forward_model_kl=forward_model_kl
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

    # def _ppo_eval(self, state):
    #     value, _, probs, _ = self.model(state=state, stage=0)
    #     return value, probs

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
                room_ids.append(info['room_id'])

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

            target_repr = []
            target_next_repr = []
            predicted_repr = []
            predicted_next_repr = []

            for state, action, next_state in tqdm(zip(states, actions, next_states), total=n):
                z_state, p_state, z_next_state, _, p_next_state = self._collect_representation_step(state, action, next_state)
                target_repr.append(z_state.cpu().numpy())
                target_next_repr.append(z_next_state.cpu().numpy())
                predicted_repr.append(p_state.cpu().numpy())
                predicted_next_repr.append(p_next_state.cpu().numpy())

            np.save('{0:s}_{1:s}{2:s}'.format(name, task, '.npy'),
                    {
                        'target_repr': target_repr,
                        'target_next_repr': target_next_repr,
                        'predicted_repr': predicted_repr,
                        'predicted_next_repr': predicted_next_repr,
                        'room_ids': room_ids,
                    })

        env.close()

    def _collect_state_step(self, env, state, mode):
        with torch.no_grad():
            _, action, _, _ = self.model(state=state, stage=0)
            next_state_raw, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state_raw)

            next_state = self._encode_state(next_state_raw)
            next_state = self.state_average.process(next_state).clip_(-4., 4.)

        done = torch.tensor(1 - done, dtype=torch.float32)

        return next_state, done, next_state_raw, action, info

    def _collect_representation_step(self, state, action, next_state):
        state = self._encode_state(state)
        state = self.state_average.process(state).clip_(-4., 4.)
        next_state = self._encode_state(next_state)
        next_state = self.state_average.process(next_state).clip_(-4., 4.)
        action = torch.tensor(action, device=self.config.device)
        with torch.no_grad():
            _, _, _, z_state = self.model(state=state, stage=0)
            p_state, z_next_state, h_next_state, p_next_state = self.model(state, action, next_state, h_next_state=self.hidden_average.mean(), stage=1)
            # int_reward, distillation_error, forward_error, confidence = self.motivation.reward(zt_state, p_state, z_next_state, h_next_state, p_next_state)

        return z_state, p_state, z_next_state, h_next_state, p_next_state
