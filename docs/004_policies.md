```python
from IPython.display import Image, clear_output, display
from ipywidgets import interact, widgets

from bo import visualize_improvement
```

## Finding improvement in Bayesian Optimization


```python
visualize_improvement("poi")
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





    (tensor([[1.0000],
             [2.0000],
             [0.5984],
             [1.0037],
             [1.0240],
             [1.0792],
             [1.1457],
             [1.2294],
             [1.3247],
             [1.4205],
             [1.4910],
             [1.5243]]),
     tensor([1.6054, 1.5029, 1.0282, 1.6115, 1.6450, 1.7353, 1.8404, 1.9623, 2.0787,
             2.1621, 2.1962, 2.2031]))




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
visualize_improvement("ei")
```


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



      0%|          | 0/500 [00:00<?, ?it/s]





    (tensor([[ 1.0000],
             [ 2.0000],
             [-0.6874],
             [ 3.4937],
             [-2.2586],
             [ 1.4894],
             [-3.3328],
             [-1.5316],
             [-5.0000],
             [-5.0000],
             [ 5.0000],
             [ 4.7935]]),
     tensor([ 1.6054,  1.5029,  0.9886, -0.7108,  1.1852,  2.1957, -0.0872,  1.0494,
              4.1659,  4.1659,  4.8633,  6.5727]))




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

