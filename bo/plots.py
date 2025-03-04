from .objectives import forrester_1d
from .models import BotorchGPModel
from .train import fit_gp_model
from matplotlib.figure import figaspect
import gpytorch
import botorch
import matplotlib.pyplot as plt
import torch


# Customize plot.
plt.style.use("bmh")  # "fivefirtyeight" is also a good choice.

# Set aspect ratio to 4:3
plt.rc("figure", figsize=figaspect(3 / 4))


def visualize_gp_belief(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    xs: torch.Tensor,
    ys: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_samples=5,
):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

    plt.plot(xs, ys, label="objective", c="r")
    plt.scatter(train_x, train_y, marker="x", c="k", label="observation")

    plt.plot(xs, predictive_mean, label="mean")
    plt.fill_between(
        xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
    )

    torch.manual_seed(0)
    for i in range(num_samples):
        plt.plot(xs, predictive_distribution.sample(), alpha=0.5, linewidth=2)

    plt.legend()


def visualize_gp_belief_and_policy(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    xs: torch.Tensor,
    ys: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    policy=None,
    next_x=None,
):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

        if policy is not None:
            acquisition_score = policy(xs.unsqueeze(1))

    if policy is None:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(xs, ys, label="objective", c="r")
        plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
        plt.plot(xs, predictive_mean, label="mean")
        plt.fill_between(
            xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
        )
        plt.legend()

        return fig
    else:
        fig, ax = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # GP belief
        ax[0].plot(xs, ys, label="objective", c="r")
        ax[0].scatter(train_x, train_y, marker="x", c="k", label="observations")
        ax[0].plot(xs, predictive_mean, label="mean")
        ax[0].fill_between(
            xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
        )

        if next_x is not None:
            ax[0].axvline(next_x.item(), linestyle="dotted", c="k")

        ax[0].legend()
        ax[0].set_ylabel("predictive")

        # Acquisition score
        ax[1].plot(xs, acquisition_score, c="g")
        ax[1].fill_between(xs.flatten(), acquisition_score, 0, color="g", alpha=0.5)

        if next_x is not None:
            ax[1].axvline(next_x.item(), linestyle="dotted", c="k")

        ax[1].set_ylabel("acquisition core")

        return fig


def visualize_improvement(strategy: str, **kwargs):
    """
    Visualize the improvement of the GP model with the given acquisition function.

    Args:
        strategy (str): acquisition function name
        **kwargs: additional arguments for the acquisition function (e.g., beta for UCB)
    """
    strategy = strategy.upper()
    if strategy not in ["POI", "EI", "UCB"]:
        raise ValueError(f"{strategy} is not supported.")

    bound = 5
    num_queries = 10

    xs = torch.linspace(-bound, bound, bound * 100 + 1).unsqueeze(1)
    ys = forrester_1d(xs)

    train_x = torch.tensor([[1.0], [2.0]])
    train_y = forrester_1d(train_x)

    for i in range(num_queries):
        # print("iteration", i)
        # print("incumbent", train_x[train_y.argmax()], train_y.max())

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = BotorchGPModel(train_x, train_y, likelihood)
        model.likelihood.noise = 1e-4

        fit_gp_model(model, likelihood, train_x, train_y)

        if strategy == "POI":
            policy = botorch.acquisition.analytic.ProbabilityOfImprovement(
                model, best_f=train_y.max()
            )
        if strategy == "EI":
            policy = botorch.acquisition.analytic.LogExpectedImprovement(
                model, best_f=train_y.max()
            )
        if strategy == "UCB":
            policy = botorch.acquisition.analytic.UpperConfidenceBound(
                model, **kwargs
            )  # beta=1

        next_x, acq_val = botorch.optim.optimize_acqf(
            policy,
            bounds=torch.tensor([[-bound * 1.0], [bound * 1.0]]),
            q=1,
            num_restarts=20,
            raw_samples=50,
        )

        fig = visualize_gp_belief_and_policy(
            model,
            likelihood,
            xs,
            ys,
            train_x,
            train_y,
            policy=policy,
            next_x=next_x,
        )

        max_x = train_x[train_y.argmax()].item()
        max_y = train_y.max()

        fig.suptitle(
            f"{strategy} acquisition function (step={i+1}, x={max_x:.2f}, y={max_y:.2f})",
        )

        plt.savefig(f"tmp/{strategy}_{i}.png")
        plt.close(fig)  # Don't display the plot.

        next_y = forrester_1d(next_x)

        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])

    return train_x, train_y
