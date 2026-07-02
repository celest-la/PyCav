from deepinv.optim.prior import Prior

class TVprior(Prior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
    def nabla(self, x, args, **kwargs):
        return x
    def fn(self, x, args, **kwargs):
        return (
            0.5 * torch.linalg.vector_norm(x, dim=tuple(range(1, x.dim())), ord=2) ** 2
        )