# GPI: Genrative-AI Powered Inference
[![PyPI version](https://img.shields.io/pypi/v/gpi_pack.svg)](https://pypi.org/project/gpi_pack/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gpi_pack.svg)](https://pypi.org/project/gpi_pack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[gpi_pack](https://gpi-pack.github.io/) is a Python package for statistical inference with text and image data powered by Large Language Models.

## Table of Contents
- [Installation](#installation)
- [Quick Guide](#quick-guide)
  - [Extracting Hidden States from an LLM](#extracting-hidden-states-from-an-llm)
  - [Estimating Causal Effect](#estimating-causal-effect)
  - [Hyperparameter Tuning (Optinoal)](#hyperparameter-tuning)
- [License](#license)
- [Contact](#contact)

## Installation
The package requires Python 3.7 or higher. The main dependencies are listed in the [requirements.txt file](requirements.txt).

### Installing via PyPI
You can install gpi_pack directly using pip:

```bash
pip install gpi_pack
```

### Installing via github
You can install the latest version directly from GitHub with pip:

```bash
pip install git+https://github.com/gpi-pack/gpi_pack.git
```

## Quick Guide

Please visit [our website](https://gpi-pack.github.io/index.html#) for the detailed explanation.

### Extracting Hidden States from an LLM

Firstly, you need to load your favorite generative model.
Below, I give the example to load [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
## Specify checkpoint (load LLaMa 3.1-8B)
checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct' #You can replace this if you want to change the model

## Load tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = <YOUR HUGGINGFACE TOKEN>)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float16
)
```

Suppose that you have the following dataframe. Your text can be either prompts to generate the new texts or the existing texts.

```python
import pandas #load pandas module for data manipulation
df = pd.DataFrame({
    'OutcomeVar': [...], #Outcome Variable
    'TreatmentVar': [...], #Treatment Variable
    'Texts': [...], #Texts
    'conf1': [...], #control variable
    'conf2': [...], #control variable
})
```

You then generate the texts and extract the hidden states.
```python
from gpi_pack.llm import extract_and_save_hidden_states

extract_and_save_hidden_states(
    prompts = df['Texts'].values, #texts or prompts
    output_hidden_dir = <YOUR HIDDEN DIR>, #directory to save hidden states
    save_name = <YOUR SAVE NAME>, #path and file name to save generated texts
    tokenizer = tokenizer,
    model = model,
    task_type = "repeat" #'repeat' is when you use LLM to regenerate texts and get hidden states
    # if you want to generate new texts, set task_type == "create"
    # You can specify any task by writing the task prompt and set task_type == <YOUR TASK>
)
```

### Estimating Causal Effect

Once you extract the hidden states, then you are ready to estimate the treatment effect!

```python
from gpi_pack.TarNet import estimate_k_ate, load_hiddens

# load hidden states stored as .pt files
hidden_dir = <YOUR-DIRECTORY> # directory containing hidden states (e.g., "hidden_last_1.pt" for text indexed 1)
hidden_states = load_hiddens(
    directory = hidden_dir, 
    hidden_list= df.index.tolist(), # list of indices for hidden states
    prefix = "hidden_last_", # prefix of hidden states (e.g., "hidden_last_" for "hidden_last_1.pt")
)

# If you want to supply the covariates, you can use either of the following methods:
# Method 1: supply covariates with a formula and DataFrame
ate, se = estimate_k_ate(
    R= hidden_states,
    Y= df['OutcomeVar'].values,
    T= df['TreatmentVar'].values,
    formula_c="conf1 + conf2",
    data=df,
    K=2, #K-fold cross-fitting
    lr = 2e-5, #learning rate
    architecture_y = [200, 1], #outcome model architecture
    architecture_z = [2048], #deconfounder architecture
)
print("ATE:", ate, "SE:", se)
    
# Method 2: supply covariates using a design matrix
import numpy as np #load numpy module
C_mat = np.column_stack([df['conf1'].values, df['conf2'].values])
ate, se = estimate_k_ate(
    R= hidden_states,
    Y= df['OutcomeVar'].values,
    T= df['TreatmentVar'].values,
    C=C_mat, #design matrix of confounding variable
    K=2, #K-fold cross-fitting
    lr = 2e-5, #learning rate
    #Outcome model architecture
    # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
    # and then it passes to the output layer with size 1.
    architecture_y = [200, 1],
    #Deconfounder model architecture:
    # [2048] means that the input (hidden states) is passed to the intermediate layer with size 2048.
    # The size of last layer (last number in the list) corresponds to the dimension of the deconfounder.
    architecture_z = [2048],
)
print("ATE:", ate, "SE:", se)
```

### Hyperparameter Tuning
You can easily fine-tune the parameter of the outcome model as follows.
Our framework is built on [Optuna](https://optuna.org/).
You only need to specify the range of hyperparameters, and it searches the best hyperparameter to minimize the loss function.

```python
from gpi_pack.TarNet import TarNetHyperparameterTuner
import optuna

# Load data and set hyperparameters
obj = TarNetHyperparameterTuner(
    # Data
    T = df['TreatmentVar'].values, 
    Y = df['OutcomeVar'].values, 
    R = hidden_states, 

    # Hyperparameters
    epoch = ["100", "200"], #try either 100 epochs or 200 epochs
    learning_rate = [1e-4, 1e-5], #draw learning rate in the range (1e-4, 1e-5)
    dropout = [0.1, 0.2], #draw dropout rate in the range (1e-4, 1e-5)
    # Outcome model architecture:
    # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
    # and then it passes to the output layer with size 1.
    architecture_y = ["[200, 1]", "[100,1]"], #either [200, 1] or [100, 1] (size of layers)
    #Deconfounder model architecture:
    # [1024] means that the input (hidden states) is passed to the intermediate layer with size 1024.
    # The size of last layer (last number in the list) corresponds to the dimension of the deconfounder.
    architecture_z = ["[1024]", "[2048]"] #either [1024] or [2048]
)

# Hyperparameter tuning with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(obj.objective, n_trials=100) #runs 100 trials to seek the best hyperparameter

#Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
Please refer to [the original paper](https://arxiv.org/abs/2410.00903) for detailed information on the methodology and findings.

```bibtex
@article{imai2024causal,
  title={Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments},
  author={Imai, Kosuke and Nakamura, Kentaro},
  journal={arXiv preprint arXiv:2410.00903},
  year={2024}
}
```

## Contact
For questions or suggestions, please open an issue or contact [knakamura@g.harvard.edu](mailto:knakamura@g.harvard.edu)