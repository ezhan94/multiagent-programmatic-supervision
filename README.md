# Generative Multi-Agent Behavioral Cloning

Code is written with PyTorch (Python 3.6.1).

First, download the data from [here](https://drive.google.com/drive/folders/1g6jlyYGH8rIrJfZ7TrGsCyS0Kf2d0lY-?usp=sharing) into the `bball_data/data/` folder.

## Example Run

You can edit `train_model.sh` and run that script, or run `train.py` directly with command-line parameters.

```javascript
$ ./train_model.py
$ python sample.py -t 105 -b 10 -n 10
$ python plot.py -t 105 --animate
```
Trained models for RNN_GAUSS (101), VRNN_SINGLE (102), VRNN_INDEP (103), and our model MACRO_VRNN (104) are included.

## Files

`model.py` contains the models. MACRO_VRNN is our model with macro-goals.

`train.py` contains the training process, and can be called with `train_model.sh`.

`sample.py` is used to sample rollouts from a trained model.

`plot.py` is used to plot the samples as well as animate them (with `--animate` flag). `--showmacro` will display macro-goals, where applicable.

`model_utils.py` contains functions for sampling and calculating various losses.

`bball_data\__init__.py` contains the `Dataset` object.

`bball_data\cfg.py` contains constants for the data.

`bball_data\macro_goals.py` is the script used to extract macro-goals. Don't need to run again.

`bball_data\utils.py` contains the functions for plotting and animating.
