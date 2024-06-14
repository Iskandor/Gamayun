import torch

from agents.PPOAgent import AgentMode
from agents.atari.PPOAtariAgent import PPOAtariAgent
from algorithms.PPO import PPO
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from loss.DPMLoss import DPMLoss
from modules.PPO_Modules import ActivationStage
from modules.atari.PPOAtariNetworkDPM import PPOAtariNetworkDPM
from motivation.DPMMotivation import DPMMotivation


class PPOAtariDPMAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkDPM(config).to(config.device)
        self.motivation = DPMMotivation(self.model,
                                        DPMLoss(config, self.model),
                                        config.motivation_lr,
                                        config.motivation_scale,
                                        config.device)
        self.algorithm = PPO(self.model,
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
                self.algorithm.train(self.memory, indices)
                self.motivation.train(self.memory, indices)
                self.memory.clear()

        return next_state, done
