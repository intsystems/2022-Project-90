import torch
import torch.nn.functional as F


def lp_dist(manifold_elems: torch.Tensor, last_elem: torch.Tensor, p: float = 2) -> torch.Tensor:
    """
    :param manifold_elems: input tensor of shape (N, D)
    :param last_elem: input tensor of shape (D,)
    :param p:p value for the p-norm distance
    :return: the output will have shape (N,)
    """
    if p <= 0:
        raise ValueError(f'p should be from 0 to +inf, got {p}')

    return torch.cdist(manifold_elems, last_elem.reshape(1, -1), p).squeeze(1)


class TemperatureScheduler:
    def __init__(self, start_temp: float = 1.0, last_epoch: int = -1):
        """
        :param start_temp: An initial temperature. Default: 1
        :param last_epoch: The index of last epoch. Default: -1
        """
        self.temp = start_temp
        self.last_epoch = last_epoch
        self._step_count = 0

    def _step(self):
        self._step_count += 1

    def get_temp(self):
        return self.temp


class TemperatureLRScheduler(TemperatureScheduler):
    def __init__(self, start_temp: float, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        """
        :param start_temp: An initial temperature
        :param step_size: Period of temperature decay.
        :param gamma: Multiplicative factor of temperature decay
        :param last_epoch: The index of last epoch. Default: -1
        """
        super().__init__(start_temp, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_temp(self):
        if self.last_epoch != -1 and self._step_count >= self.last_epoch:
            return self.temp

        if (self._step_count + 1) % self.step_size == 0:
            self.temp *= self.gamma

        self._step()
        return self.temp


class AbstractCCMMetric:
    def __init__(self, t_scheduler: TemperatureScheduler):
        self.t_scheduler = t_scheduler

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        raise NotImplementedError

    def _check_arguments(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        assert x_elems.size()[0] == y_elems.size()[0]
        assert x_elems.ndim == y_elems.ndim == 2


class CanonicalCCM(AbstractCCMMetric):
    @staticmethod
    def pearson_corr_coef(x_arr: torch.Tensor, y_arr: torch.Tensor):
        assert x_arr.size() == y_arr.size() and x_arr.ndim == 1

        x_centered = x_arr - x_arr.mean()
        y_centered = y_arr - y_arr.mean()

        cov = torch.dot(x_centered, y_centered)
        x_std = torch.sqrt(torch.dot(x_centered, x_centered))
        y_std = torch.sqrt(torch.dot(y_centered, y_centered))

        if cov.item() == 0:
            return cov
        else:
            return cov / x_std / y_std

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        # x_elems = [n_samples, n_features]
        # y_elems = [n_samples, n_targets]
        self._check_arguments(x_elems, y_elems)

        manifold_x_elems, last_x_elem = x_elems[:-1, :], x_elems[-1, :]
        manifold_y_elems, last_y_elem = y_elems[:-1, :], y_elems[-1, :]

        distances = lp_dist(manifold_x_elems, last_x_elem)
        coefs = F.softmax(-distances / self.t_scheduler.get_temp(), dim=0)

        y_hat = torch.sum(torch.multiply(manifold_y_elems, coefs.unsqueeze(1)), dim=0)

        return CanonicalCCM.pearson_corr_coef(y_hat, last_y_elem)


class ModifiedCCM(CanonicalCCM):
    def __init__(self, t_scheduler: TemperatureScheduler, min_manifold_size: int = 100):
        super().__init__(t_scheduler)
        self.min_manifold_size = min_manifold_size

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        # x_elems = [n_samples, n_features]
        # y_elems = [n_samples, n_targets]
        self._check_arguments(x_elems, y_elems)

        smaller_manifold_size = min(x_elems.size()[0] // 3, self.min_manifold_size)

        return super().__call__(x_elems, y_elems) - \
               super().__call__(x_elems[:smaller_manifold_size, :],
                                y_elems[:smaller_manifold_size, :])


class LipschitzCCM(AbstractCCMMetric):
    def __init__(self, t_scheduler: TemperatureScheduler, k_neighbours: int):
        super().__init__(t_scheduler)
        self.k_neighbours = k_neighbours

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        self._check_arguments(x_elems, y_elems)

        manifold_x_elems, last_x_elem = x_elems[:-1, :], x_elems[-1, :]
        manifold_y_elems, last_y_elem = y_elems[:-1, :], y_elems[-1, :]

        x_distances = lp_dist(manifold_x_elems, last_x_elem)
        y_distances = lp_dist(manifold_y_elems, last_y_elem)

        x_distances = torch.sort(x_distances).values[:self.k_neighbours]
        y_distances = torch.sort(y_distances).values[:self.k_neighbours]

        return torch.mean(x_distances) / torch.mean(y_distances)

    def _check_arguments(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        super()._check_arguments(x_elems, y_elems)
        assert x_elems.size()[0] > self.k_neighbours
