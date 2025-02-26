```python
from IPython.display import Image, clear_output, display
from ipywidgets import interact, widgets

from bo import visualize_improvement
```


```python
visualize_improvement("ucb", beta=1)
```


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



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[ 1.0000],
             [ 2.0000],
             [-1.0771],
             [ 1.3371],
             [ 1.4680],
             [ 1.5210],
             [ 1.5248],
             [ 1.5249],
             [ 1.5243],
             [ 1.5238],
             [ 1.5237],
             [ 1.5225]]),
     tensor([1.6054, 1.5029, 1.0002, 2.0917, 2.1879, 2.2027, 2.2031, 2.2031, 2.2031,
             2.2030, 2.2030, 2.2029]))




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
visualize_improvement("ucb", beta=2)
```


      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[ 1.0000],
             [ 2.0000],
             [-1.3380],
             [ 5.0000],
             [ 4.4613],
             [ 4.2093],
             [ 4.6015],
             [ 4.5784],
             [ 4.5788],
             [ 4.5780],
             [ 4.5781],
             [ 4.5783]]),
     tensor([1.6054, 1.5029, 1.0143, 4.8633, 6.9494, 5.5486, 7.1408, 7.1430, 7.1431,
             7.1429, 7.1429, 7.1430]))




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
import matplotlib.pyplot as plt
import torch

from bo import fit_gp_model, forrester_1d, visualize_gp_belief_and_policy

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

    model, likelihood = fit_gp_model(train_x, train_y)

    ts = botorch.generation.MaxPosteriorSampling(model, replacement=False)
    next_x = ts(candidate_x, num_samples=1)

    fig = visualize_gp_belief_and_policy(
        model, likelihood, next_x=next_x, xs=xs, ys=ys, train_x=train_x, train_y=train_y
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

