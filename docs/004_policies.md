```python
from IPython.display import Image, clear_output, display
from ipywidgets import interact, widgets

from bo.models import BotorchGPModel
from bo.plots import visualize_improvement
```

## Finding improvement in Bayesian Optimization


```python
visualize_improvement("poi", GPModel=BotorchGPModel)
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



      0%|          | 0/500 [00:00<?, ?it/s]



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[1.0000],
             [2.0000],
             [0.5985],
             [1.0036],
             [1.0239],
             [1.0790],
             [1.1453],
             [1.2280],
             [1.3234],
             [1.4198],
             [1.4920],
             [1.5242]]),
     tensor([1.6054, 1.5029, 1.0283, 1.6114, 1.6448, 1.7350, 1.8399, 1.9603, 2.0773,
             2.1617, 2.1965, 2.2031]))




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
    display(Image(f"tmp/poi_{play}.png"))
```


    interactive(children=(Play(value=0, description='Press play', interval=500, max=9), IntSlider(value=0, descrip…


## Optimizing the expected value of improvement


```python
visualize_improvement("ei", GPModel=BotorchGPModel)
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
             [-0.6874],
             [ 1.3487],
             [-2.7626],
             [ 1.5062],
             [ 5.0000],
             [ 4.6726],
             [ 4.5229],
             [ 4.5885],
             [ 3.5243],
             [ 4.5878]]),
     tensor([ 1.6054,  1.5029,  0.9886,  2.1032,  0.7674,  2.2001,  4.8633,  7.0464,
              7.0928,  7.1437, -0.5037,  7.1438]))




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
    display(Image(f"tmp/ei_{play}.png"))
```


    interactive(children=(Play(value=0, description='Press play', interval=500, max=9), IntSlider(value=0, descrip…

