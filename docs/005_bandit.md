```python
from IPython.display import Image, clear_output, display
from ipywidgets import interact, widgets

from bo.models import BotorchGPModel
from bo.plots import visualize_improvement
```


```python
visualize_improvement("ucb", GPModel=BotorchGPModel, beta=1)
```


      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/botorch/optim/optimize.py:652: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
    [OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]
    Trying again with a new set of initial conditions.
      return _optimize_acqf_batch(opt_inputs=opt_inputs)



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[ 1.0000],
             [ 2.0000],
             [-1.0775],
             [ 1.3378],
             [ 1.4685],
             [ 1.5211],
             [ 1.5248],
             [ 1.5240],
             [ 1.5242],
             [ 1.5240],
             [ 1.5230],
             [ 1.5231]]),
     tensor([1.6054, 1.5029, 1.0002, 2.0923, 2.1881, 2.2027, 2.2031, 2.2030, 2.2031,
             2.2030, 2.2029, 2.2029]))




```python
play = widgets.Play(
    value=0,
    min=0,
    max=9,
    step=1,
    interval=500,
    description="Press play",
    disabled=False,
)
slider = widgets.IntSlider(min=0, max=9)
widgets.jslink((play, "value"), (slider, "value"))
widgets.HBox([play, slider])


@interact(play=play, slider=slider)
def f(play, slider):
    clear_output(wait=True)
    display(Image(f"tmp/ucb_{play}.png"))
```


    interactive(children=(Play(value=0, description='Press play', interval=500, max=9), IntSlider(value=0, descrip…



```python
visualize_improvement("ucb", GPModel=BotorchGPModel, beta=2)
```


      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/botorch/optim/optimize.py:652: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
    [OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]
    Trying again with a new set of initial conditions.
      return _optimize_acqf_batch(opt_inputs=opt_inputs)



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[ 1.0000],
             [ 2.0000],
             [-1.3384],
             [ 4.9695],
             [ 4.4508],
             [ 4.1582],
             [ 4.6024],
             [ 4.5767],
             [ 4.5773],
             [ 4.5772],
             [ 4.5776],
             [ 4.5775]]),
     tensor([1.6054, 1.5029, 1.0143, 5.1835, 6.9161, 5.1404, 7.1405, 7.1426, 7.1428,
             7.1427, 7.1428, 7.1428]))




```python
play = widgets.Play(
    value=0,
    min=0,
    max=9,
    step=1,
    interval=500,
    description="Press play",
    disabled=False,
)
slider = widgets.IntSlider(min=0, max=9)
widgets.jslink((play, "value"), (slider, "value"))
widgets.HBox([play, slider])


@interact(play=play, slider=slider)
def f(play, slider):
    clear_output(wait=True)
    display(Image(f"tmp/ucb_{play}.png"))
```


    interactive(children=(Play(value=0, description='Press play', interval=500, max=9), IntSlider(value=0, descrip…


## Smart Sampling with Thompson policy sampling


```python
import botorch
import gpytorch
import matplotlib.pyplot as plt
import torch

from bo.objectives import forrester_1d
from bo.plots import visualize_gp_belief_and_policy
from bo.train import fit_gp_model
from bo.models import BotorchGPModel

bound = 5


xs = torch.linspace(-bound, bound, bound * 100 + 1).unsqueeze(1)
ys = forrester_1d(xs)

train_x = torch.tensor([[1.0], [2.0]])
train_y = forrester_1d(train_x)

num_candidates = 1000
num_queries = 10


torch.manual_seed(1)

for i in range(num_queries):
    # print("iteration", i)
    # print("incumbent", train_x[train_y.argmax()], train_y.max())

    sobol = torch.quasirandom.SobolEngine(1, scramble=True)

    candidate_x = sobol.draw(num_candidates)
    candidate_x = 10 * candidate_x - 5

    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BotorchGPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    fit_gp_model(model, likelihood, train_x, train_y)

    ts = botorch.generation.MaxPosteriorSampling(model, replacement=False)
    next_x = ts(candidate_x, num_samples=1)

    fig = visualize_gp_belief_and_policy(
        model, likelihood, xs, ys, train_x, train_y, next_x=next_x
    )
    fig.suptitle(
        f"TS acquisition function (step={i+1}, x={train_x[train_y.argmax()].item():.2f}, y={train_y.max():.2f})",
        fontsize=20,
    )

    image_path = f"tmp/ts_{i}.png"
    plt.savefig(image_path, bbox_inches="tight")
    plt.close(fig)

    next_y = forrester_1d(next_x)

    train_x = torch.cat([train_x, next_x])
    train_y = torch.cat([train_y, next_y])
```


      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/linear_operator/utils/cholesky.py:40: NumericalWarning: A not p.d., added jitter of 1.0e-06 to the diagonal
      warnings.warn(



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/linear_operator/utils/cholesky.py:40: NumericalWarning: A not p.d., added jitter of 1.0e-05 to the diagonal
      warnings.warn(
    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/linear_operator/utils/cholesky.py:40: NumericalWarning: A not p.d., added jitter of 1.0e-04 to the diagonal
      warnings.warn(



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/linear_operator/utils/cholesky.py:40: NumericalWarning: A not p.d., added jitter of 1.0e-03 to the diagonal
      warnings.warn(



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



```python
play = widgets.Play(
    value=0,
    min=0,
    max=9,
    step=1,
    interval=500,
    description="Press play",
    disabled=False,
)
slider = widgets.IntSlider(min=0, max=9)
widgets.jslink((play, "value"), (slider, "value"))
widgets.HBox([play, slider])


@interact(play=play, slider=slider)
def f(play, slider):
    clear_output(wait=True)
    display(Image(f"tmp/ts_{play}.png"))
```


    interactive(children=(Play(value=0, description='Press play', interval=500, max=9), IntSlider(value=0, descrip…

