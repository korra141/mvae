import torch

class Loss:

    def sample_rademacher(*shape, device="cpu"):
        return (torch.rand(*shape, device=device) > 0.5).float() * 2 - 1

    def elbo(y,
            s,
            a_func,
            drift=None,
            proj2manifold=None,
            proj2tangent=None,
            T=None,
            method="closest",
            dim=None,
    ):
        """

        :param y: data
        :param s: time step
        :param a_func: variational degree of freedom
        :param drift: drift variable
        :param proj2manifold: closest-point projection to manifold (used if method=`closest`)
        :param proj2tangent: tangential projection
        :param T: estimation time interval (terminal time)
        :param method: computation method
        :param dim: dimensionality of the manifold (used if method=`qr`)
        :return: elbo estimate
        """
        if method == "closest":
            py = proj2manifold(y)
            a = a_func(py, s)
            if drift is not None:
                drift = proj2tangent(py, drift)
            else:
                drift = 0

            v0 = proj2tangent(py, a) - drift

            div_v0 = 0
            for i in range(y.size(1)):
                div_v0 += torch.autograd.grad(
                    v0[:, i].sum(), y, create_graph=True, retain_graph=True
                )[0][:, i]
        elif method == "qr":
            print(f"y {y}")
            a = a_func(y, s)
            # print(a.shape)
            # print(y.shape)
            if drift is not None:
                drift = proj2tangent(y, drift)
            else:
                drift = 0

            v0 = proj2tangent(y, a) - drift
            print(f"v0 {v0}")

            basis = torch.linalg.qr(
                proj2tangent(y, torch.randn(y.shape + (dim,), device=y.device))
            )[0]

            div_v0 = 0
            print(y.shape)
            for i in range(dim):
                q = basis[:, :, i]
                print((v0 * q).sum().shape)
                div_v0 += (
                        torch.autograd.grad(
                            (v0 * q).sum(), y, create_graph=True, allow_unused=True
                        )[0]
                        * q
                ).sum(1)
        elif method == "hutchinson-normal":
            a = a_func(y, s)
            if drift is not None:
                drift = proj2tangent(y, drift)
            else:
                drift = 0
            v0 = proj2tangent(y, a) - drift

            q = proj2tangent(y, torch.randn_like(y))
            div_v0 = (
                    torch.autograd.grad(
                        (v0 * q).sum(), y, create_graph=True, retain_graph=True
                    )[0]
                    * q
            ).sum(1)
        elif method == "hutchinson-rademacher":
            a = a_func(y, s)
            if drift is not None:
                drift = proj2tangent(y, drift)
            else:
                drift = 0
            v0 = proj2tangent(y, a) - drift

            q = proj2tangent(y, Loss.sample_rademacher(y.shape, device=y.device))
            div_v0 = (
                    torch.autograd.grad(
                        (v0 * q).sum(), y, create_graph=True, retain_graph=True
                    )[0]
                    * q
            ).sum(1)

        else:
            raise NotImplementedError

        return (-0.5 * (proj2tangent(y, a) ** 2).sum(dim=1) - div_v0) * T

