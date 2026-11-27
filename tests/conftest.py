import pytest
import torch

from bendflow.posterior import EmpiricalPosterior

N, D = 20, 5


@pytest.fixture(autouse=True, scope="session")
def numerical_settings():
    old_dtype, old_threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(min(old_threads, 2))
    yield
    torch.set_default_dtype(old_dtype)
    torch.set_num_threads(old_threads)


def rng(seed):
    return torch.Generator().manual_seed(seed)


def normal(seed, shape):
    return torch.randn(shape, generator=rng(seed))


@pytest.fixture
def posterior():
    return EmpiricalPosterior(normal(0, (N, D)))


@pytest.fixture(params=[0.15, 0.5, 0.8])
def point(request, posterior):
    t = torch.tensor(request.param)
    index = torch.randint(N, (), generator=rng(int(100 * t)))
    return t * posterior.data[index] + (1 - t) * normal(int(100 * t) + 1, (D,)), t
