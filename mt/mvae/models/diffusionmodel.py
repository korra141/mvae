import torch.nn as nn
import torch
from ..components import Component
from typing import List
from .. import utils
from ..stats import BatchStats
import numpy as np

_scaling_min = 0.001


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return torch.sigmoid(x) * x


class DiffusionModel(nn.Module):

    def __init__(self, components: List[Component], dimh, T,dataset):
        super(DiffusionModel, self).__init__()
        self.components = nn.ModuleList(components)
        self.total_z_dim = sum(component.dim for component in components)
        self.in_dim = dataset.in_dim
        self.reconstruction_loss = dataset.reconstruction_loss
        num_hidden_layers = 4

        layers = [nn.Linear(self.in_dim+ 1, dimh), Swish()]

        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(dimh, dimh), Swish()])
        layers.extend([nn.Linear(dimh, self.in_dim)])

        self.net = torch.nn.Sequential(*layers)
        self.T = T
        for component in components:
            component.init_layers(dimh, self.T, scalar_parametrization=False)

    def forward(self,x,t):
        z_mani = []
        for component in self.components:
            assert len(x.shape) == 2
            bs, dim = x.shape
            assert dim == self.in_dim
            x = x.view(bs, self.in_dim)
            z = component(x,t)
            z_mani.append(z)

        concat_z = torch.cat(tuple(manis for manis in z_mani),dim=-1)
        return self.net(torch.cat([concat_z,t],1))\
            # .view(-1,bs,self.in_dim).squeeze(dim=0)

    def evaluate_exact_logp(self,dataloader, a_func, bm_sde):
        logps = []
        counter = 0
        for x_ in dataloader:
            counter += 1
            if counter % 100 == 0:
                print(counter)
            logp = self.exact_logp(x_ambient=x_, bm_sde=bm_sde, a_func=a_func, prior=prior,
                                    steps=args.evaluation_num_steps)
            logps.append(logp.detach())
            print(counter, 'logp.mean until now:', torch.cat(logps).mean())
        logps = torch.cat(logps, axis=0)

    def exact_logp(x_ambient, bm_sde, a_func, prior, steps, method="closest"):
        from ..ops.sde import ExactLogPODE

        a_func.eval()
        logp = ExactLogPODE(bm_sde, a_func).exact_logp_via_integration(
            x_ambient, bm_sde.T, prior=prior, steps=steps, method=method
        )
        a_func.train()
        return logp

    # def log_likelihood(self, x: Tensor, n: int = 500) -> Tuple[Tensor, Tensor, Tensor]:
    #     """
    #     :param x: Mini-batch of inputs.
    #     :param n: Number of MC samples
    #     :return: Monte Carlo estimate of log-likelihood.
    #     """
    #     sample_shape = torch.Size([n])
    #     batch_size = x.shape[0]
    #     prob_shape = torch.Size([n, batch_size])
    #
    #     x_encoded = self.encode(x)
    #     log_p_z = torch.zeros(prob_shape, device=x.device)
    #     log_q_z_x = torch.zeros(prob_shape, device=x.device)
    #     zs = []
    #     for component in self.components:
    #         q_z, p_z, z_params = component(x_encoded)
    #
    #         # Numerically more stable.
    #         z, log_q_z_x_, log_p_z_ = component.sampling_procedure.rsample_log_probs(sample_shape, q_z, p_z)
    #         zs.append(z)
    #
    #         log_p_z += log_p_z_
    #         log_q_z_x += log_q_z_x_
    #
    #     concat_z = torch.cat(zs, dim=-1)
    #     x_mb_ = self.decode(concat_z)
    #     x_orig = x.repeat((n, 1, 1))
    #     log_p_x_z = -self.reconstruction_loss(x_mb_, x_orig).sum(dim=-1)
    #
    #     assert log_p_x_z.shape == log_p_z.shape
    #     assert log_q_z_x.shape == log_p_z.shape
    #     joint = (log_p_x_z + log_p_z - log_q_z_x)
    #     log_p_x = joint.logsumexp(dim=0) - np.log(n)
    #
    #     assert log_q_z_x.shape == log_p_z.shape
    #     mi = (log_q_z_x - log_p_z).logsumexp(dim=0) - np.log(n)
    #
    #     mean_z = torch.mean(concat_z, dim=1, keepdim=True)
    #     mean_x = torch.mean(x_orig, dim=1, keepdim=True)
    #     cov_norm = torch.bmm((x - mean_x).transpose(1, 2), concat_z - mean_z).mean(dim=0).norm()
    #
    #     return log_p_x, mi, cov_norm

    def compute_batch_stats(self,
                            x_mb,
                            x_mb_,
                            beta: float,
                            likelihood_n: int = 0) -> BatchStats:
        bce = self.reconstruction_loss(x_mb_, x_mb).sum(dim=-1)
        # assert torch.isfinite(bce).all()
        # assert (bce >= 0).all()

        component_kl = []
        for i, component in enumerate(self.components):
            t = utils.stratified_uniform(x_mb.size(0), self.T)
            kl_comp = component.kl_loss(x_mb,t,self.forward)
            assert torch.isfinite(kl_comp).all()
            component_kl.append(kl_comp)

        log_likelihood = None
        mi = None
        cov_norm = None
        if likelihood_n:
            log_likelihood, mi, cov_norm = self.log_likelihood(x_mb, n=likelihood_n)

        return BatchStats(bce, component_kl, beta, log_likelihood, mi, cov_norm)

    def train_step(self, optimizer: torch.optim.Optimizer, x_mb,
                   beta: float):
        optimizer.zero_grad()

        x_mb = x_mb
        t = utils.stratified_uniform(x_mb.size(0), self.T)
        x_mb_ = self.forward(x_mb,t)
        assert x_mb_.shape == x_mb.shape
        batch_stats = self.compute_batch_stats(x_mb, x_mb_, likelihood_n=0, beta=beta)

        loss = -batch_stats.elbo  # Maximize elbo instead of minimizing it.
        assert torch.isfinite(loss).all()
        loss.backward()
        c_params = [v for k, v in self.named_parameters() if "curvature" in k]
        if c_params:  # TODO: Look into this, possibly disable it.
            torch.nn.utils.clip_grad_norm_(c_params, max_norm=1.0, norm_type=2)  # Enable grad clip?
        optimizer.step()

        return batch_stats.convert_to_float(), x_mb_




class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependant init."""

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0, learn_scale=True):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter(
            "b", nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        )
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                "logs",
                nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
            )

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(
                    torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                        torch.log(self.scale / (torch.sqrt(var) + 1e-6))
                        / self.logscale_factor
                )
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale

            return output.view(input_shape)
        else:
            return output.view(input_shape)
