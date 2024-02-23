import torch

from agents.PPOAgent import PPOAgentBase, AgentMode
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from analytic.InfoCollector import InfoCollector
from analytic.ResultCollector import ResultCollector
from modules.PPO_AtariModules import PPOAtariNetwork, PPOAtariNetworkRND, PPOAtariNetworkSND, PPOAtariNetworkSP, PPOAtariNetworkICM
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation
from motivation.SNDMotivation import SNDMotivationFactory


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


class PPOAtariRNDAgent(PPOAtariAgent):
    def __init__(self, config,):
        super().__init__(config)
        self.model = PPOAtariNetworkRND(config).to(config.device)
        self.motivation = RNDMotivation(self.model.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
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
            self.algorithm.train(self.memory, indices)
            self.motivation.train(self.memory, indices)

            if indices is not None:
                self.memory.clear()

        return next_state


class PPOAtariICMAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkICM(config).to(config.device)
        self.motivation = ForwardModelMotivation(self.model.forward_model, config.motivation_lr, config.motivation_eta, config.device)
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
            next_state, reward, done, trunc, info = env.step(self._convert_action(action.cpu()))
            self._check_terminal_states(env, mode, done, next_state)

            next_state = self._encode_state(next_state)
            int_reward = self.motivation.reward(state, action, next_state)

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
                            next_state=next_state.cpu(),
                            reward=reward.cpu(),
                            mask=done.cpu())

            indices = self.memory.indices()

            if indices is not None:
                self.algorithm.train(self.memory, indices)
                self.memory.clear()

        return next_state


class PPOAtariSPAgent(PPOAtariICMAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSP(config).to(config.device)
        self.motivation = ForwardModelMotivation(self.model.forward_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)


class PPOAtariSNDAgent(PPOAtariAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSND(config).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = SNDMotivationFactory.get_motivation(config.type, self.model.cnd_model, config.motivation_lr, config.motivation_eta, config.device)
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
