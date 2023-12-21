from dataclasses import dataclass

import numpy
import torch

from agents.PPOAgent import PPOAgent
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from analytic.ResultCollector import ResultCollector
from analytic.metric.NoveltyMetric import NoveltyMetric
from modules.PPO_AtariModules import PPOAtariNetwork, PPOAtariNetworkRND, PPOAtariNetworkSND, PPOAtariNetworkSP, PPOAtariNetworkICM
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation, ASPDMotivation
from motivation.SNDMotivation import SNDMotivationFactory


class PPOAtariAgent(PPOAgent):
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

    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOAtariNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)

    def encode_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.config.device)

    def initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,))
        return analysis

    def step(self, env, agent_state):
        with torch.no_grad():
            agent_state.value, agent_state.action, agent_state.probs = self.get_action(agent_state.state)
        agent_state.next_state, agent_state.reward, agent_state.done, agent_state.trunc, agent_state.info = env.step(self.convert_action(agent_state.action.cpu()))

        return agent_state

    def print_step_info(self, trial, stats, i):
        print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} steps {4:d}  mean reward {5:f} score {6:f})]'.format(
            trial,
            self.step_counter.steps,
            self.step_counter.limit,
            stats['re'].sum[i],
            int(stats['re'].step[i]),
            self.reward_avg.value().item(),
            stats['score'].sum[i]))

    def update_analysis(self, agent_state, analysis):
        analysis.update(re=agent_state.reward)
        if 'raw_score' in agent_state.info:
            score = torch.tensor(agent_state.info['raw_score']).unsqueeze(-1)
            analysis.update(score=score)
        analysis.end_step()

    def train(self, agent_state):
        agent_state.next_state = self.encode_state(agent_state.next_state)
        agent_state.reward = torch.tensor(agent_state.reward, dtype=torch.float32)
        agent_state.done = torch.tensor(1 - agent_state.done, dtype=torch.float32)

        self.memory.add(state=agent_state.state.cpu(), value=agent_state.value.cpu(), action=agent_state.action.cpu(), prob=agent_state.probs.cpu(), reward=agent_state.reward.cpu(),
                        mask=agent_state.done.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()

        agent_state.state = agent_state.next_state


class PPOAtariRNDAgent(PPOAtariAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOAtariNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,), ri=(1,))
        metric = NoveltyMetric(self.network.input_shape[1],
                               self.network.input_shape[2],
                               NoveltyMetric.Greyscale,
                               self.network.rnd_model.learned_model,
                               self.network.rnd_model.target_model,
                               self.config.batch_size,
                               self.config.device)
        analysis.add_metric(metric)
        return analysis

    def print_step_info(self, trial, stats, i):
        print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})] novelty score {10:f}'.format(
            trial,
            self.step_counter.steps,
            self.step_counter.limit,
            stats['re'].sum[i],
            stats['ri'].max[i],
            stats['ri'].mean[i],
            stats['ri'].std[i],
            int(stats['re'].step[i]),
            self.reward_avg.value().item(),
            stats['score'].sum[i],
            stats[NoveltyMetric.KEY]))

    def update_analysis(self, agent_state, analysis):
        analysis.update(re=agent_state.ext_reward,
                        ri=agent_state.int_reward)

        if 'raw_score' in agent_state.info:
            score = torch.tensor(agent_state.info['raw_score']).unsqueeze(-1)
            analysis.update(score=score)
        analysis.end_step()

    def train(self, agent_state):
        agent_state.ext_reward = torch.tensor(agent_state.reward, dtype=torch.float32)
        agent_state.int_reward = self.motivation.reward(agent_state.state).cpu().clip(0.0, 1.0)

        agent_state.next_state = self.encode_state(agent_state.next_state)
        agent_state.reward = torch.cat([agent_state.ext_reward, agent_state.int_reward], dim=1)
        agent_state.done = torch.tensor(1 - agent_state.done, dtype=torch.float32)

        self.memory.add(state=agent_state.state.cpu(), value=agent_state.value.cpu(), action=agent_state.action.cpu(), prob=agent_state.probs.cpu(), reward=agent_state.reward.cpu(),
                        mask=agent_state.done.cpu())

        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()

        agent_state.state = agent_state.next_state


class PPOAtariICMAgent(PPOAtariAgent):
    class AgentState(PPOAtariAgent.AgentState):
        def __init__(self):
            self.ext_reward = None
            self.int_reward = None
            self.features = None

    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOAtariNetworkICM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.forward_model.encoder(self.network.forward_model.preprocess(state))

        return features.detach(), value.detach(), action, probs.detach()

    def initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,), ri=(1,), feature_space=(1,))
        return analysis

    def step(self, env, agent_state):
        with torch.no_grad():
            agent_state.features, agent_state.value, agent_state.action, agent_state.probs = self.get_action(agent_state.state)
        agent_state.next_state, agent_state.reward, agent_state.done, agent_state.trunc, agent_state.info = env.step(self.convert_action(agent_state.action.cpu()))

        return agent_state

    def print_step_info(self, trial, stats, i):
        print(
            'Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f}))]'.format(
                trial,
                self.step_counter.steps,
                self.step_counter.limit,
                stats['re'].sum[i],
                stats['ri'].max[i],
                stats['ri'].mean[i],
                stats['ri'].std[i],
                int(stats['re'].step[i]),
                self.reward_avg.value().item(),
                stats['score'].sum[i],
                stats['feature_space'].max[i],
                stats['feature_space'].mean[i],
                stats['feature_space'].std[i])
        )

    def update_analysis(self, agent_state, analysis):
        analysis.update(re=agent_state.ext_reward,
                        ri=agent_state.int_reward,
                        feature_space=agent_state.features.norm(p=2, dim=1, keepdim=True).cpu())

        if 'raw_score' in agent_state.info:
            score = torch.tensor(agent_state.info['raw_score']).unsqueeze(-1)
            analysis.update(score=score)
        analysis.end_step()

    def train(self, agent_state):
        agent_state.next_state = self.encode_state(agent_state.next_state)

        agent_state.ext_reward = torch.tensor(agent_state.reward, dtype=torch.float32)
        agent_state.int_reward = self.motivation.reward(agent_state.state, agent_state.action, agent_state.next_state).cpu().clip(0.0, 1.0)

        agent_state.reward = torch.cat([agent_state.ext_reward, agent_state.int_reward], dim=1)
        agent_state.done = torch.tensor(1 - agent_state.done, dtype=torch.float32)

        self.memory.add(state=agent_state.state.cpu(),
                        value=agent_state.value.cpu(),
                        action=agent_state.action.cpu(),
                        prob=agent_state.probs.cpu(),
                        next_state=agent_state.next_state.cpu(),
                        reward=agent_state.reward.cpu(),
                        mask=agent_state.done.cpu())

        indices = self.memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.motivation.train(self.memory, indices)

            self.memory.clear()

        agent_state.state = agent_state.next_state


class PPOAtariSPAgent(PPOAtariICMAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOAtariNetworkSP(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)


class PPOAtariSNDAgent(PPOAtariAgent):
    class AgentState(PPOAtariAgent.AgentState):
        def __init__(self):
            self.ext_reward = None
            self.int_reward = None
            self.features = None

    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOAtariNetworkSND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = SNDMotivationFactory.get_motivation(config.type, self.network.cnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))

        return features.detach(), value.detach(), action, probs.detach()

    def initialize_analysis(self):
        analysis = ResultCollector()
        analysis.init(self.config.n_env, re=(1,), score=(1,), ri=(1,), feature_space=(1,))
        metric = NoveltyMetric(self.network.input_shape[1],
                               self.network.input_shape[2],
                               NoveltyMetric.Greyscale,
                               self.network.cnd_model.learned_model,
                               self.network.cnd_model.target_model,
                               self.config.batch_size,
                               self.config.device)
        analysis.add_metric(metric)

        return analysis

    def step(self, env, agent_state):
        self.motivation.update_state_average(agent_state.state)
        with torch.no_grad():
            agent_state.features, agent_state.value, agent_state.action, agent_state.probs = self.get_action(agent_state.state)
        agent_state.next_state, agent_state.reward, agent_state.done, agent_state.trunc, agent_state.info = env.step(self.convert_action(agent_state.action.cpu()))

        return agent_state

    def print_step_info(self, trial, stats, i):
        print(
            'Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f})] novelty score {13:f}'.format(
                trial,
                self.step_counter.steps,
                self.step_counter.limit,
                stats['re'].sum[i],
                stats['ri'].max[i],
                stats['ri'].mean[i],
                stats['ri'].std[i],
                int(stats['re'].step[i]),
                self.reward_avg.value().item(),
                stats['score'].sum[i],
                stats['feature_space'].max[i],
                stats['feature_space'].mean[i],
                stats['feature_space'].std[i],
                stats[NoveltyMetric.KEY]))

    def update_analysis(self, agent_state, analysis):
        analysis.update(re=agent_state.ext_reward,
                        ri=agent_state.int_reward,
                        feature_space=agent_state.features.norm(p=2, dim=1, keepdim=True).cpu())

        if 'raw_score' in agent_state.info:
            score = torch.tensor(agent_state.info['raw_score']).unsqueeze(-1)
            analysis.update(score=score)
        analysis.end_step()

    def train(self, agent_state):
        agent_state.ext_reward = torch.tensor(agent_state.reward, dtype=torch.float32)
        agent_state.int_reward = self.motivation.reward(agent_state.state).cpu().clip(0.0, 1.0)

        agent_state.next_state = self.encode_state(agent_state.next_state)
        agent_state.reward = torch.cat([agent_state.ext_reward, agent_state.int_reward], dim=1)
        agent_state.done = torch.tensor(1 - agent_state.done, dtype=torch.float32)

        self.memory.add(state=agent_state.state.cpu(),
                        value=agent_state.value.cpu(),
                        action=agent_state.action.cpu(),
                        prob=agent_state.probs.cpu(),
                        reward=agent_state.reward.cpu(),
                        mask=agent_state.done.cpu())

        self.motivation_memory.add(state=agent_state.state.cpu(),
                                   next_state=agent_state.next_state.cpu(),
                                   action=agent_state.action.cpu(),
                                   done=agent_state.done.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

        agent_state.state = agent_state.next_state
