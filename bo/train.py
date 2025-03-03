import gpytorch
import torch
from tqdm.notebook import tqdm


def fit_gp_model(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_train_iters=500,
):
    """
    optimize the hyperparameters of the GP model

    Args:
        model (gpytorch.models.ExactGP): GP model
        likelihood: likelihood

    Returns:
        model: optimized GP model
        likelihood: optimized
    """
    # noise = 1e-4

    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = GPModel(train_x, train_y, likelihood)
    # model.likelihood.noise = noise

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in tqdm(range(num_train_iters)):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood
