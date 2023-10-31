class ConfigPPO:
    def __init__(self, steps, lr, n_env, gamma, device='cpu'):
        self.steps = steps
        self.lr = lr
        self.n_env = n_env
        self.gamma = gamma
        self.batch_size = 512
        self.trajectory_size = 16384
        self.ppo_epochs = 4
        self.actor_loss_weight = 1
        self.critic_loss_weight = 0.5
        self.beta = 0.001
        self.device = device
