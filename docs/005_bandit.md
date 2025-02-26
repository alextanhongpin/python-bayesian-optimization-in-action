```python
from IPython.display import Image, clear_output, display
from ipywidgets import interact, widgets

from bo import visualize_improvement
```


```python
visualize_improvement("ucb", beta=1)


@interact(slides=widgets.IntSlider(min=0, max=9))
def f(slides):
    clear_output(wait=True)
    display(Image(f"tmp/ucb_{slides}.png"))
```


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



      0%|          | 0/500 [00:00<?, ?it/s]


    /Users/alextanhongpin/Library/Caches/pypoetry/virtualenvs/python-bayesian-optimization-in-action-aU6qUxK9-py3.12/lib/python3.12/site-packages/botorch/optim/optimize.py:652: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
    [OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]
    Trying again with a new set of initial conditions.
      return _optimize_acqf_batch(opt_inputs=opt_inputs)



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]



    interactive(children=(IntSlider(value=0, description='slides', max=9), Output()), _dom_classes=('widget-intera…



```python
visualize_improvement("ucb", beta=2)


@interact(slides=widgets.IntSlider(min=0, max=9))
def f(slides):
    clear_output(wait=True)
    display(Image(f"tmp/ucb_{slides}.png"))
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



    interactive(children=(IntSlider(value=0, description='slides', max=9), Output()), _dom_classes=('widget-intera…


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
@interact(slides=widgets.IntSlider(min=0, max=9))
def f(slides):
    clear_output(wait=True)
    display(Image(f"tmp/ts_{slides}.png"))
```


    interactive(children=(IntSlider(value=0, description='slides', max=9), Output()), _dom_classes=('widget-intera…

