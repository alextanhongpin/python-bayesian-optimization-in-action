```python
import botorch
import gpytorch
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm, trange

plt.style.use("fivethirtyeight")
```


```python
# %load bo.py
import matplotlib.pyplot as plt
import torch

# Customize plot.
plt.style.use("fivethirtyeight")
plt.rc("figure", figsize=(16, 8))


def visualize_gp_belief(model, likelihood, num_samples=5):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

    plt.figure(figsize=(8, 6))

    plt.plot(xs, ys, label="objective", c="r")
    plt.scatter(train_x, train_y, marker="x", c="k", label="observation")

    plt.plot(xs, predictive_mean, label="mean")
    plt.fill_between(
        xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
    )

    torch.manual_seed(0)
    for i in range(num_samples):
        plt.plot(xs, predictive_distribution.sample(), alpha=0.5, linewidth=2)

    plt.legend(fontsize=15)
    plt.show()


def visualize_gp_belief_and_policy(model, likelihood, policy=None, next_x=None):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

        if policy is not None:
            acquisition_score = policy(xs.unsqueeze(1))

    if policy is None:
        plt.figure(figsize=(16, 6))
        plt.plot(xs, ys, label="objective", c="r")
        plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
        plt.plot(xs, predictive_mean, label="mean")
        plt.fill_between(
            xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
        )
        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
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
        plt.show()

        pass


def forrester_1d(x):
    # a modification of https://www.sfu.ca/~ssurjano/forretal08.html
    y = -((x + 1) ** 2) * torch.sin(2 * x + 2) / 5 + 1
    return y.squeeze(-1)


def ackley(x):
    # a modification of https://www.sfu.ca/~ssurjano/ackley.html
    return -20 * torch.exp(
        -0.2 * torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2) / 2)
    ) - torch.exp(torch.cos(2 * pi * x[:, 0] / 3) + torch.cos(2 * pi * x[:, 1]))
```


```python
class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    num_outputs = 1

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```


```python
def fit_gp_model(train_x, train_y, num_train_iters=500):
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

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
```


```python
bound = 5


xs = torch.linspace(-bound, bound, bound * 100 + 1).unsqueeze(1)
ys = forrester_1d(xs)
```


```python
torch.manual_seed(2)

train_x = torch.tensor([[1.0], [2.0]])
train_y = forrester_1d(train_x)

print(torch.hstack([train_x, train_y.unsqueeze(1)]))

model, likelihood = fit_gp_model(train_x, train_y)
```

    tensor([[1.0000, 1.6054],
            [2.0000, 1.5029]])



      0%|          | 0/500 [00:00<?, ?it/s]


## Finding improvement in Bayesian Optimization


```python
def plot_improvement(acquisition_fn):
    num_queries = 10

    train_x = torch.tensor([[1.0], [2.0]])
    train_y = forrester_1d(train_x)

    for i in range(num_queries):
        print("iteration", i)
        print("incumbent", train_x[train_y.argmax()], train_y.max())

        model, likelihood = fit_gp_model(train_x, train_y)

        policy = acquisition_fn(model, best_f=train_y.max())

        next_x, acq_val = botorch.optim.optimize_acqf(
            policy,
            bounds=torch.tensor([[-bound * 1.0], [bound * 1.0]]),
            q=1,
            num_restarts=20,
            raw_samples=50,
        )

        visualize_gp_belief_and_policy(model, likelihood, policy, next_x=next_x)

        next_y = forrester_1d(next_x)

        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
```


```python
plot_improvement(botorch.acquisition.analytic.ProbabilityOfImprovement)
```

    iteration 0
    incumbent tensor([1.]) tensor(1.6054)



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/botorch/optim/optimize.py:652: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
    [OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]
    Trying again with a new set of initial conditions.
      return _optimize_acqf_batch(opt_inputs=opt_inputs)



    
![png](004_policies_files/004_policies_8_3.png)
    


    iteration 1
    incumbent tensor([1.]) tensor(1.6054)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_6.png)
    


    iteration 2
    incumbent tensor([1.0036]) tensor(1.6114)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_9.png)
    


    iteration 3
    incumbent tensor([1.0238]) tensor(1.6447)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_12.png)
    


    iteration 4
    incumbent tensor([1.0788]) tensor(1.7347)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_15.png)
    


    iteration 5
    incumbent tensor([1.1453]) tensor(1.8398)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_18.png)
    


    iteration 6
    incumbent tensor([1.2300]) tensor(1.9631)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_21.png)
    


    iteration 7
    incumbent tensor([1.3250]) tensor(2.0790)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_24.png)
    


    iteration 8
    incumbent tensor([1.4211]) tensor(2.1625)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_27.png)
    


    iteration 9
    incumbent tensor([1.4917]) tensor(2.1964)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_8_30.png)
    


## Optimizing the expected value of improvement


```python
plot_improvement(botorch.acquisition.analytic.LogExpectedImprovement)
```

    iteration 0
    incumbent tensor([1.]) tensor(1.6054)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_2.png)
    


    iteration 1
    incumbent tensor([1.]) tensor(1.6054)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_5.png)
    


    iteration 2
    incumbent tensor([1.]) tensor(1.6054)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_8.png)
    


    iteration 3
    incumbent tensor([1.4906]) tensor(2.1961)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_11.png)
    


    iteration 4
    incumbent tensor([1.4906]) tensor(2.1961)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_14.png)
    


    iteration 5
    incumbent tensor([1.4906]) tensor(2.1961)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_17.png)
    


    iteration 6
    incumbent tensor([-5.]) tensor(4.1659)



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/botorch/optim/optimize.py:652: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
    [OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]
    Trying again with a new set of initial conditions.
      return _optimize_acqf_batch(opt_inputs=opt_inputs)



    
![png](004_policies_files/004_policies_10_21.png)
    


    iteration 7
    incumbent tensor([5.]) tensor(4.8633)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_24.png)
    


    iteration 8
    incumbent tensor([4.7870]) tensor(6.6084)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_27.png)
    


    iteration 9
    incumbent tensor([4.6996]) tensor(6.9750)



      0%|          | 0/500 [00:00<?, ?it/s]



    
![png](004_policies_files/004_policies_10_30.png)
    

