import random
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from analytic.ResultCollector import ResultCollector
from analytic.metric.NoveltyMetric import NoveltyMetric
from modules import init_orthogonal
from modules.encoders.EncoderAtari import ST_DIMEncoderAtari, BarlowTwinsEncoderAtari, VICRegEncoderAtari, SNDVEncoderAtari, AtariStateEncoderSmall, AMIEncoderAtari, AtariStateEncoderLarge, \
    AtariStateEncoderResNet, SpacVICRegEncoderAtari
from utils.StateNorm import PreciseNorm


class RNDModelAtari(nn.Module):
    def __init__(self, config):
        super(RNDModelAtari, self).__init__()

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.target_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.learned_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.learned_model[0], np.sqrt(2))
        init_orthogonal(self.learned_model[2], np.sqrt(2))
        init_orthogonal(self.learned_model[4], np.sqrt(2))
        init_orthogonal(self.learned_model[7], np.sqrt(2))
        init_orthogonal(self.learned_model[9], np.sqrt(2))
        init_orthogonal(self.learned_model[11], np.sqrt(2))

    def prepare_input(self, state):
        x = state - self.state_average.mean
        return x[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        x = self.prepare_input(state)
        predicted_code = self.learned_model(x)
        target_code = self.target_model(x)
        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = torch.sum(torch.pow(target - prediction, 2), dim=1).unsqueeze(-1) / 2

        return error

    def loss_function(self, state):
        prediction, target = self(state)
        # loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        loss = torch.pow(target - prediction, 2)
        mask = torch.rand_like(loss) < 0.25
        loss *= mask
        loss = loss.sum() / mask.sum()

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss.unsqueeze(-1).detach())
        analytic.update_metric()

        return loss

    def update_state_average(self, state):
        self.state_average.update(state)


class STDModelAtari(nn.Module):
    def __init__(self, config):
        super(STDModelAtari, self).__init__()

        self.config = config

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        input_channels = 1
        # input_channels = input_shape[0]
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        self.target_model = ST_DIMEncoderAtari(self.input_shape, self.feature_dim)

        self.learned_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 0.5
        init_orthogonal(self.learned_model[0], gain)
        init_orthogonal(self.learned_model[2], gain)
        init_orthogonal(self.learned_model[4], gain)
        init_orthogonal(self.learned_model[6], gain)
        init_orthogonal(self.learned_model[9], gain)
        init_orthogonal(self.learned_model[11], gain)
        init_orthogonal(self.learned_model[13], gain)

    def preprocess(self, state):
        if self.config.cnd_preprocess == 0:
            x = state
        if self.config.cnd_preprocess == 1:
            x = state - self.state_average.mean
        if self.config.cnd_preprocess == 2:
            x = ((state - self.state_average.mean) / self.state_average.std).clip(-1., 1.)

        return x[:, 0, :, :].unsqueeze(1)

    def forward(self, state, fmaps=False):
        s = self.preprocess(state)

        f5 = self.learned_model[:6](s)
        predicted_code = self.learned_model[6:](f5)

        if fmaps:
            target_code = self.target_model(s, fmaps)

            return {
                'predicted_f5': f5.permute(0, 2, 3, 1),
                'predicted_code': predicted_code,
                'target_f5': target_code['f5'],
                'target_code': target_code['out']
            }
        else:
            target_code = self.target_model(s).detach()

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)

            # if self.config.cnd_error_k == 2:
            #     error = torch.mean(torch.pow(target - prediction, 2), dim=1, keepdim=True)
            # if self.config.cnd_error_k == 1:
            #     error = torch.mean(torch.abs(target - prediction), dim=1, keepdim=True)

            error = self.k_distance(self.config.cnd_error_k, prediction, target, reduction='mean')

        return error

    def loss_function_crossentropy(self, state, next_state):
        out = self(state, fmaps=True)
        prediction_f5, prediction, target_f5, target = out['predicted_f5'], out['predicted_code'], out['target_f5'], out['target_code']

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='sum')  # + nn.functional.mse_loss(prediction_f5, target_f5.detach(), reduction='sum')

        loss_target, loss_target_norm = self.target_model.loss_function_crossentropy(self.preprocess(state), self.preprocess(next_state))
        # loss_target_uniform = nn.functional.mse_loss(torch.matmul(target.T, target), torch.eye(self.feature_dim, self.feature_dim, device=self.config.device), reduction='sum')
        # target_logits = torch.pow(target, 2)  # 42
        # target_logits = torch.abs(target)  # 43
        # target_logits = (target_logits / target_logits.sum(dim=0)) + 1e-8
        # loss_target_uniform = torch.sum(target_logits * target_logits.log(), dim=1).mean()
        loss_target_uniform = -torch.std(target, dim=1).mean()  # 40
        # beta1 = 1e-6
        beta1 = 1e-4
        beta2 = self.config.cnd_loss_target_reg

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), loss_target_norm=loss_target_norm.detach() * beta2,
                        loss_reg=loss_target_uniform.detach() * beta1)
        analytic.update_metric()

        return loss_prediction * self.config.cnd_loss_pred + (loss_target + loss_target_uniform * beta1 + loss_target_norm * beta2) * self.config.cnd_loss_target

    def loss_function_cdist(self, state, next_state):
        prediction, target = self(state)
        loss_prediction = nn.functional.mse_loss(prediction, target)
        loss_target = self.target_model.loss_function_cdist(self.preprocess(state), self.preprocess(next_state))

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), loss_reg=torch.zeros(1), loss_target_norm=torch.zeros(1))

        return loss_prediction * self.config.cnd_loss_pred + loss_target * self.config.cnd_loss_target

    def loss_function(self, state, next_state):
        return self.loss_function_crossentropy(state, next_state)

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        self.state_average.update(state)


class SNDVModelAtari(nn.Module):
    def __init__(self, config):
        super(SNDVModelAtari, self).__init__()

        self.config = config
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        input_channels = 1
        # input_channels = input_shape[0]
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)


        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        self.target_model = SNDVEncoderAtari(self.input_shape, self.feature_dim, config)

        self.learned_model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_model[0], gain)
        init_orthogonal(self.learned_model[2], gain)
        init_orthogonal(self.learned_model[4], gain)
        init_orthogonal(self.learned_model[6], gain)
        init_orthogonal(self.learned_model[9], gain)
        init_orthogonal(self.learned_model[11], gain)
        init_orthogonal(self.learned_model[13], gain)

    def preprocess(self, state):
        if self.config.cnd_preprocess == 0:
            x = state
        if self.config.cnd_preprocess == 1:
            x = state - self.state_average.mean
        if self.config.cnd_preprocess == 2:
            x = ((state - self.state_average.mean) / self.state_average.std).clip(-1., 1.)

        return x[:, 0:self.input_shape[0], :, :]

    def forward(self, state):
        s = self.preprocess(state)

        predicted_code = self.learned_model(s)
        target_code = self.target_model(s)

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = ((target - prediction) ** 2).mean(dim=1, keepdim=True)

        return error

    def sample_states(self, states, batch_size, far_ratio=0.5, device='cpu'):
        count = self.config.trajectory_size

        indices_a = torch.randint(0, count, size=(batch_size,), device=device)

        indices_close = indices_a

        indices_far = torch.randint(0, count, size=(batch_size,), device=device)

        labels = (torch.rand(batch_size, device=device) > far_ratio)

        # label 0 = close states
        # label 1 = distant states
        indices_b = torch.logical_not(labels) * indices_close + labels * indices_far

        states_a = torch.index_select(states, dim=0, index=indices_a).float()
        states_b = torch.index_select(states, dim=0, index=indices_b).float()
        labels_t = labels.float()

        return states_a.to(self.config.device), states_b.to(self.config.device), labels_t.to(self.config.device)

    def loss_function(self, state_batch, state, dropout=0.75):
        state_norm_t = self.preprocess(state_batch).detach()

        features_predicted_t = self.learned_model(state_norm_t)
        features_target_t = self.target_model(state_norm_t).detach()

        loss_cnd = (features_target_t - features_predicted_t) ** 2

        # random loss regularization, 25% non zero for 128envs, 100% non zero for 32envs
        '''
        prob            = 1.0 - dropout
        random_mask     = torch.rand(loss_cnd.shape).to(loss_cnd.device)
        random_mask     = 1.0*(random_mask < prob) 
        loss_cnd        = (loss_cnd*random_mask).sum() / (random_mask.sum() + 0.00000001)
        '''
        random_mask = (torch.rand_like(loss_cnd) > dropout).float()
        loss_cnd = (loss_cnd * random_mask).sum() / (random_mask.sum() + 0.00000001)

        states_a, states_b, target = self.sample_states(state, self.config.batch_size // 8)
        loss_target = self.target_model.loss_function(self.preprocess(states_a), self.preprocess(states_b), target)

        return loss_cnd + loss_target

    def update_state_average(self, state):
        self.state_average.update(state)


class BarlowTwinsModelAtari(nn.Module):
    def __init__(self, config):
        super(BarlowTwinsModelAtari, self).__init__()

        self.config = config

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        input_channels = 1
        # input_channels = input_shape[0]
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)


        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        self.target_model = BarlowTwinsEncoderAtari(self.input_shape, self.feature_dim)

        self.learned_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_model[0], gain)
        init_orthogonal(self.learned_model[2], gain)
        init_orthogonal(self.learned_model[4], gain)
        init_orthogonal(self.learned_model[6], gain)
        init_orthogonal(self.learned_model[9], gain)
        init_orthogonal(self.learned_model[11], gain)
        init_orthogonal(self.learned_model[13], gain)

    def preprocess(self, state):
        return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        predicted_code = self.learned_model(self.preprocess(state))
        target_code = self.target_model(self.preprocess(state))

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = self.k_distance(self.config.cnd_error_k, prediction, target, reduction='mean')

        return error

    def loss_function(self, state, next_state):
        prediction, target = self(state)

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='mean')
        loss_target = self.target_model.loss_function(self.preprocess(state), self.preprocess(next_state))

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach())

        return loss_prediction + loss_target

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        self.state_average.update(state)


class VICRegModelAtari(nn.Module):
    def __init__(self, config, encoder_class=VICRegEncoderAtari):
        super(VICRegModelAtari, self).__init__()

        self.config = config

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        input_channels = 1
        # input_channels = input_shape[0]
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)

        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        self.target_model = encoder_class(self.input_shape, self.feature_dim)
        self.learned_model = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=sqrt(2))

        self.learned_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], gain)

    def preprocess(self, state):
        return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        predicted_code = self.learned_projection(self.learned_model(self.preprocess(state)))
        target_code = self.target_model(self.preprocess(state))

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = self.k_distance(2, prediction, target, reduction='mean')

        return error

    def loss_function(self, state, next_state):
        prediction, target = self(state)

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='mean')

        loss_target = self.target_model.loss_function(self.preprocess(state), self.preprocess(next_state))

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach())
        analytic.update_metric()

        return loss_prediction + loss_target

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        self.state_average.update(state)


class SpacVICRegModelAtari(VICRegModelAtari):
    def __init__(self, config):
        super(SpacVICRegModelAtari, self).__init__(config)

        self.target_model = SpacVICRegEncoderAtari(self.input_shape, self.feature_dim)


class VINVModelAtari(VICRegModelAtari):
    def __init__(self, config):
        super(VINVModelAtari, self).__init__(config)

        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.action_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.inv_model[0], gain)
        init_orthogonal(self.inv_model[2], gain)
        init_orthogonal(self.inv_model[4], gain)

    def loss_function(self, state, next_state, action):
        prediction, target = self(state)
        next_target = self.target_model(self.preprocess(next_state))
        tnt = torch.cat([target, next_target], dim=1)
        action_logits = self.inv_model(tnt)
        action_target = torch.argmax(action, dim=1)
        action_prediction = torch.argmax(nn.functional.softmax(action_logits), dim=1)
        accuracy = (action_prediction == action_target).float().mean() * 100

        iota = 0.5
        loss_inv = nn.functional.cross_entropy(action_logits, action_target) * iota
        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='mean')
        loss_target = self.target_model.loss_function(self.preprocess(state), self.preprocess(next_state))

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), inv_accuracy=accuracy.unsqueeze(-1).detach())
        analytic.update_metric()

        return loss_prediction + loss_target + loss_inv


class TPModelAtari(VICRegModelAtari):
    def __init__(self, config):
        super(TPModelAtari, self).__init__(config)

        self.terminal_model = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 2)
        )

        gain = sqrt(2)
        init_orthogonal(self.terminal_model[0], gain)
        init_orthogonal(self.terminal_model[2], gain)
        init_orthogonal(self.terminal_model[4], gain)

    def loss_function(self, state, next_state, done):
        prediction, target = self(state)
        next_target = self.target_model(self.preprocess(next_state))
        done_logits = self.terminal_model(next_target)

        done_predicted = torch.argmax(torch.softmax(done_logits, dim=1), dim=1, keepdim=True)

        negative = (done == 0).sum()
        if negative == 0:
            specificity = torch.zeros(1, device=self.config.device, dtype=torch.float32).squeeze()
        else:
            true_negative = ((done_predicted == done) * (done == 0)).float().sum()
            specificity = true_negative / negative

        iota = 0.2

        weight_const = (done.shape[0] - torch.count_nonzero(done)) / done.shape[0]
        weight = torch.ones(2, device=self.config.device, dtype=torch.float32)
        weight[1] *= weight_const.item()

        loss_terminal = F.cross_entropy(done_logits, done.flatten().long(), weight=weight, reduction='sum') * iota
        loss_prediction = F.mse_loss(prediction, target.detach(), reduction='mean')
        loss_target = self.target_model.loss_function(self.preprocess(state), self.preprocess(next_state))

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), specificity=specificity.unsqueeze(-1).detach())
        analytic.update_metric()

        return loss_prediction + loss_target + loss_terminal


class AMIModelAtari(nn.Module):
    def __init__(self, config):
        super(AMIModelAtari, self).__init__()

        self.config = config

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        input_channels = 1
        # input_channels = input_shape[0]
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.state_average = PreciseNorm((4, input_height, input_width), config.device)

        self.target_model = VICRegEncoderAtari(self.input_shape, self.feature_dim)

        self.learned_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_model[0], gain)
        init_orthogonal(self.learned_model[2], gain)
        init_orthogonal(self.learned_model[4], gain)
        init_orthogonal(self.learned_model[7], gain)
        init_orthogonal(self.learned_model[9], gain)
        init_orthogonal(self.learned_model[11], gain)

        self.action_encoder = AMIEncoderAtari(self.input_shape, self.action_dim, config)

    def preprocess(self, state):
        return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        predicted_code = self.learned_model(self.preprocess(state))
        target_code = self.target_model(self.preprocess(state))

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = self.k_distance(self.config.cnd_error_k, prediction, target, reduction='mean')

        return error

    def loss_function(self, state, next_state, actions):
        prediction, target = self(state)
        _, next_target = self(next_state)

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='mean')

        loss_target = self.target_model.loss_function(state, next_state)

        loss_action = self.action_encoder.loss_function(target, next_target, actions)

        analytic = ResultCollector()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), loss_action=loss_action.unsqueeze(-1).detach())

        return loss_prediction + loss_target + loss_action * 0.2

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        pass
