from agents.atari.PPOAtariICMAgent import PPOAtariICMAgent
from algorithms.PPO import PPO
from modules.atari.PPOAtariNetworkSP import PPOAtariNetworkSP
from motivation.ForwardModelMotivation import ForwardModelMotivation


class PPOAtariSPAgent(PPOAtariICMAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = PPOAtariNetworkSP(config).to(config.device)
        self.motivation = ForwardModelMotivation(self.model.forward_model, config.motivation_lr, config.motivation_scale, config.device)
        self.algorithm = PPO(self.model, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)