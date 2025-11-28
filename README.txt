LeNet-5 on Fashion-MNIST
=========================

This project implements and trains a LeNet-5–style convolutional neural network
on the Fashion-MNIST dataset under several different settings (baseline,
dropout, weight decay, batch normalization). It also allows you to test all
saved models (best and final checkpoints) using a separate testing script.

Files
-----

- main.py              : training script (trains all configurations, saves weights, plots, summary)
- testing_models_3.py  : testing script (loads saved weights and evaluates on the test set)

Both scripts should live in the same folder.

Requirements
------------

Install the required Python packages (you can use a virtual environment):

    pip install torch torchvision matplotlib tqdm

The scripts use:
- torch, torchvision
- matplotlib
- tqdm
- numpy
- standard library: os

Dataset
-------

The code uses the Fashion-MNIST dataset. It will be downloaded automatically
into a local 'data/' directory the first time you run main.py.

Training Settings
-----------------

Global hyperparameters are defined near the top of main.py:

    BATCH_SIZE    = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS    = 30
    DROPOUT_RATE  = 0.5
    WEIGHT_DECAY  = 1e-4

The LeNet-5 model is defined in main.py and supports:
- use_dropout    (True/False)
- use_batch_norm (True/False)

During training, four configurations are used (inside main()):

    configurations = [
        ('Baseline',           False, False, False),
        ('Dropout',            True,  False, False),
        ('Weight_Decay',       False, False, True),
        ('Batch_Normalization',False, True,  False),
    ]
    #               (name,   use_dropout, use_batch_norm, use_weight_decay)

Interpretation:

- Baseline
  - Dropout:        OFF
  - BatchNorm:      OFF
  - Weight Decay:   OFF

- Dropout
  - Dropout:        ON
  - BatchNorm:      OFF
  - Weight Decay:   OFF

- Weight_Decay
  - Dropout:        OFF
  - BatchNorm:      OFF
  - Weight Decay:   ON (via Adam's weight_decay argument)

- Batch_Normalization
  - Dropout:        OFF
  - BatchNorm:      ON
  - Weight Decay:   OFF

How to Train
------------

### 1. Train all configurations

From the folder containing main.py, run:

    python main.py

This will:

1. For each configuration in 'configurations':
   - Build a LeNet5 model with the specified use_dropout/use_batch_norm.
   - Train for NUM_EPOCHS on the Fashion-MNIST training set.
   - Evaluate on the test set after each epoch.
   - Keep track of the best test accuracy.
   - Save:
        models/<CONFIG_NAME>_best.pth   (weights of the best test accuracy)
        models/<CONFIG_NAME>_final.pth  (weights after the last epoch)

2. Plot convergence curves (train vs test accuracy) and save them as:
        plots/<CONFIG_NAME>_convergence.png

3. Create a summary of final accuracies and save it to:
        results_summary.txt

Directories created (if they do not exist):

- models/
- plots/
- data/

### 2. Train a single configuration only (optional)

If you want to train only one setting (for example, Dropout), edit the
'configurations' list in main() to contain only that configuration. For example:

    configurations = [
        ('Dropout', True, False, False),
    ]

Then run:

    python main.py

This will only train and save weights for the Dropout configuration.

Saved Weights
-------------

For each configuration, two files are saved in the 'models/' directory:

- models/<CONFIG_NAME>_best.pth
    Weights of the model that achieved the highest test accuracy during training.

- models/<CONFIG_NAME>_final.pth
    Weights of the model after the last training epoch.

Example expected files:

- models/Baseline_best.pth
- models/Baseline_final.pth
- models/Dropout_best.pth
- models/Dropout_final.pth
- models/Weight_Decay_best.pth
- models/Weight_Decay_final.pth
- models/Batch_Normalization_best.pth
- models/Batch_Normalization_final.pth

How to Test with Saved Weights
------------------------------

The script testing_models_3.py is provided to load each saved model and evaluate
it on the Fashion-MNIST test set.

It expects that:
- main.py has already been run (so that the weights exist in 'models/').
- testing_models_3.py is in the same folder as main.py.

### 1. Test all models (best and final)

Run:

    python testing_models_3.py

What this script does:

1. Defines the same logical configurations as in training:

       configurations = [
           ('Baseline',          False, False),
           ('Dropout',           True,  False),
           ('Weight_Decay',      False, False),
           ('Batch_Normalization',False, True),
       ]
       #              (name, use_dropout, use_batch_norm)

   Note: Weight decay is only relevant during training (it is part of the
   optimizer), so testing only needs use_dropout and use_batch_norm.

2. Loads the test data once using get_data_loaders() from main.py.

3. For each configuration and for each model_type in ['best', 'final']:
   - Build a LeNet5 model with the correct flags (matching training).
   - Load the corresponding weights:
         models/<CONFIG_NAME>_best.pth
         models/<CONFIG_NAME>_final.pth
   - Evaluate test accuracy with:
         evaluate(model, test_loader, training_mode=False)
   - Print the accuracy or an error if the file is missing.

4. At the end, it prints a summary table of all tested models with their test
   accuracies.

Example console output structure:

    ======================================================================
    Testing: Baseline
    ======================================================================
        best:  89.12%
       final:  88.75%

    ======================================================================
    Testing: Dropout
    ======================================================================
        best:  90.34%
       final:  90.10%
    ...

    ======================================================================
    SUMMARY
    ======================================================================
    Configuration              Type       Test Accuracy
    ------------------------------------------------------
    Baseline                   best             89.12%
    Baseline                   final            88.75%
    Dropout                    best             90.34%
    Dropout                    final            90.10%
    ...

You do not need to manually set flags in this script: it already matches
use_dropout and use_batch_norm to the configurations used during training.

### 2. Test a single model manually (optional)

If you prefer to test one configuration manually (for example, in a notebook),
you can do something like:

    import torch
    from main import LeNet5, get_data_loaders, evaluate, device

    config_name    = "Baseline"
    use_dropout    = False
    use_batch_norm = False   # Must match the training setting

    _, test_loader = get_data_loaders()

    model = LeNet5(use_dropout=use_dropout, use_batch_norm=use_batch_norm).to(device)
    model.load_state_dict(torch.load(f"models/{config_name}_best.pth", map_location=device))

    test_acc = evaluate(model, test_loader, training_mode=False)
    print(f"Test accuracy ({config_name}, best): {test_acc:.2f}%")

The important points when testing manually:

- The architecture (use_dropout, use_batch_norm) must match how the model was
  trained for that configuration.
- Weight decay does NOT need to be specified at test time (it is only used in
  the optimizer during training).
- Use training_mode=False when calling evaluate() so that dropout is disabled
  during evaluation.

Notes
-----

- Always run main.py first to generate and save the model weights.
- testing_models_3.py is a convenient way to evaluate all models (best and
  final) in one command.
- Plots of training and test accuracy over epochs are saved in the 'plots/'
  directory for each configuration.
