# GPI: Genrative-AI Powered Inference
[![PyPI version](https://img.shields.io/pypi/v/gpi_pack.svg)](https://pypi.org/project/gpi_pack/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gpi_pack.svg)](https://pypi.org/project/gpi_pack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[gpi_pack](https://gpi-pack.github.io/) is a Python package for statistical
inference with text, image, and video data powered by generative AI models.

## Table of Contents
- [Installation](#installation)
- [Quick Guide for Static Setting](#quick-guide-for-static-setting)
  - [Extracting Hidden States from an LLM](#extracting-hidden-states-from-an-llm)
  - [Regenerating Videos and Extracting Representations](#regenerating-videos-and-extracting-representations)
  - [Estimating Causal Effect (Static Setup)](#estimating-causal-effect-static-setup)
- [Quick Guide for Dynamic Setting](#quick-guide-for-dynamic-setting)
  - [Extracting Hidden States for Videos](#extracting-hidden-states-for-videos)
  - [Estimating Causal Effect (Dynamic Setup)](#estimating-causal-effect-dynamic-setup)
- [Hyperparameter Tuning (Optional)](#hyperparameter-tuning)
- [License](#license)
- [Contact](#contact)

## Installation
The package requires Python 3.9 or higher. The main dependencies are listed in
the [requirements.txt file](requirements.txt).

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

## Quick Guide for Static Setting

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

### Regenerating Videos and Extracting Representations

Install the optional video codecs and use `extract_videos` to split each input
video, reconstruct each segment with the NVIDIA Cosmos tokenizer, and save the
model representation used by the video paper:

```bash
pip install "gpi_pack[video]"
```

```python
import torch
from gpi_pack.video import extract_videos

outputs = extract_videos(
    videos="path/to/videos",          # one file, one directory, or a list
    output_hidden_dir="outputs/hidden",
    output_video_dir="outputs/reconstructed",  # omit to skip decoding
    segment_seconds=5,
    frame_size=(320, 480),            # optional; preprocessing is recorded
)

payload = torch.load(outputs[0].representation_path, weights_only=True)
representation = payload["representation"]  # unbatched [C, H, W]
```

The saved representation is the final Cosmos tensor immediately before the
decoder, mean-pooled over latent time. The paper reports a `(16, 40, 60)`
representation, but does not specify its resize or frame-sampling policy, so
the package exposes these choices as `frame_size` and `max_frames` and records
them in every payload. The default `max_frames=121` follows the supported
context listed for the default Cosmos checkpoint. Set `save_decoder_input=True`
or `save_latent=True` to retain the unpooled state or encoder latent. Audio is
not reconstructed.

For an in-memory RGB array, reuse one loaded model:

```python
from gpi_pack.video import CosmosVideoExtractor

extractor = CosmosVideoExtractor(frame_size=(320, 480))
result = extractor.reconstruct_video(frames)  # frames: [T, H, W, 3]
reconstructed = result.reconstruction         # [1, 3, T, H, W]
representation = result.representation        # [1, C, H_latent, W_latent]
```

### Estimating Causal Effect (Static Setup)

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
    formula_C="conf1 + conf2",
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

## Quick Guide for Dynamic Setting

### Extracting Hidden States for Videos

The dynamic examples use the following dimension notation:

| Symbol | Meaning |
| --- | --- |
| `N` | Number of units (for example, people or source videos). |
| `T` | Common padded sequence length: the maximum number of video segments across units. |
| `S_i` | Number of observed video segments for unit `i`, where `S_i <= T`. |
| `F` or `F_text` | Number of vector or text features for each segment. |
| `C_video` | Number of feature channels in the unpooled Cosmos decoder representation. |
| `T_latent` | Number of latent video-time positions retained within one segment. |
| `D` | Depth passed to the 3D video encoder. With temporal pooling, `D = C_video`; without temporal pooling, `D = T_latent`. |
| `H`, `W` | Height and width of the latent video representation, not necessarily the raw video resolution. |
| `P` | Number of structured covariates. |
| `d_z` | Number of learned deconfounder features. This is the final value in `architecture_z`. |
| `J` | Number of interventions evaluated. |
| `K` | Number of cross-fitting folds. |

There are therefore two distinct sequence axes for unpooled video input: outer
`T` indexes padded video-segment positions, of which unit `i` observes `S_i`,
while inner `T_latent` indexes latent video time within one segment.

Firstly, install the optional dependencies for reading videos and loading the
NVIDIA Cosmos tokenizer:

```bash
pip install "gpi_pack[video]"
```

Suppose that you have one video for each unit. Each video is divided into
fixed-length segments, which form the longitudinal sequence in the dynamic
setup.

```python
import pandas as pd

df_dynamic = pd.DataFrame({
    "OutcomeVar": [...],       # one outcome for each unit
    "VideoPath": [...],        # path to the video for each unit
    "TreatmentSeq": [...],     # list of binary treatments for its segments
})
```

You can then regenerate each video segment and extract its internal
representation. Omitting `output_video_dir` skips reconstruction and only
saves the representations, which is faster.

```python
import torch
from gpi_pack.video import extract_videos

video_hidden_states = []

for video_path in df_dynamic["VideoPath"]:
    outputs = extract_videos(
        videos=video_path,
        output_hidden_dir="outputs/video_hidden",
        output_video_dir="outputs/reconstructed",  # omit to skip reconstruction
        segment_seconds=5,                          # length of one segment
        frame_size=(320, 480),
        temporal_pooling="temporal_mean",
    )

    # `outputs` follows the segment order in this video.
    representations = [
        torch.load(output.representation_path, weights_only=True)[
            "representation"
        ].numpy()
        for output in outputs
    ]
    video_hidden_states.append(representations)
```

Before temporal pooling, the unbatched Cosmos decoder representation has shape
`[C_video, T_latent, H, W]`. The two pooling choices produce:

- `temporal_pooling="temporal_mean"`: average over `T_latent` and save
  `[C_video, H, W]`. Dynamic GPI reads this as a single-channel volume
  `[D, H, W]`, where `D = C_video`. Stacking and padding produces
  `R_video` with shape `[N, T, D, H, W]`.
- `temporal_pooling="none"`: retain latent time and save
  `[C_video, T_latent, H, W]`. Dynamic GPI reads this as
  `[C_video, D, H, W]`, where `D = T_latent`. Stacking and padding produces
  `R_video` with shape `[N, T, C_video, D, H, W]`; pass
  `video_in_channels=C_video` to `estimate_k_ipsi`.

For unit `i` before padding, these shapes are respectively
`[S_i, D, H, W]` and `[S_i, C_video, D, H, W]`. The saved `representation`
field is the input used by dynamic GPI.

Because units can have different numbers of segments, pad the representations
to the longest sequence. Use zeros for padded hidden states and trailing `NaN`
values in the treatment array. The `NaN` values, rather than the hidden-state
values, tell `estimate_k_ipsi` which positions are padding.

```python
import numpy as np

N = len(video_hidden_states)
T = max(len(sequence) for sequence in video_hidden_states)
D, H, W_video = video_hidden_states[0][0].shape  # temporal_mean representation

R_video = np.zeros((N, T, D, H, W_video), dtype=np.float32)
W = np.full((N, T), np.nan, dtype=np.float32)

for i, sequence in enumerate(video_hidden_states):
    S_i = len(sequence)
    treatment_i = np.asarray(
        df_dynamic.iloc[i]["TreatmentSeq"], dtype=np.float32
    )
    if len(treatment_i) != S_i:
        raise ValueError("Each video segment must have one treatment value.")

    R_video[i, :S_i] = np.stack(sequence)
    W[i, :S_i] = treatment_i
```

When text and video are used together, construct the aligned text hidden states
as `R_text` with shape `[N, T, F_text]`, using the same unit and segment order.
Fill its padded positions with zeros. `estimate_k_ipsi` automatically uses the
multimodal model when `R_video` is supplied.

### Estimating Causal Effect (Dynamic Setup)

Once you prepare the aligned hidden states and treatments, you are ready to
estimate the incremental-intervention curve. The main inputs are:

- `R`: vector or text hidden states. Use `[N, T, F]` for the multimodal setup.
- `R_video`: optional video hidden states with shape `[N, T, D, H, W]` or
  `[N, T, C_video, D, H, W]`.
- `W`: binary treatments with shape `[N, T]`. Observed values must be `0` or
  `1`, and unused trailing positions must be `NaN`.
- `Y`: one scalar outcome for each unit, with shape `[N]`.
- `C`: optional static covariates `[N, P]` or segment-varying covariates
  `[N, T]` or `[N, T, P]`.

A finite `0` in `W` is an observed control treatment, not padding. If you
already have a separate sequence mask, convert it before estimation with
`W = np.where(mask.astype(bool), W, np.nan)`.

The dynamic estimator and tuner now use the same architecture convention as
the static TarNet API:

| Previous dynamic name | Current name |
| --- | --- |
| `nsplits` | `K` |
| `epochs` | `nepoch` |
| `learning_rate` | `lr` |
| `architecture_fr` | `architecture_z` |
| `head_hidden=[16]` | `architecture_y=[16, 1]` |
| `include_Ti_in_head` | `include_Si_in_head` (lower-level `DynamicTarNet` API) |
| `include_Ti_in_rep` | `include_Si_in_rep` (lower-level `DynamicTarNet` API) |

The final `1` in `architecture_y` is the scalar outcome layer. The lower-level
`DynamicTarNet` class follows `TarNet` and therefore continues to use
`epochs` and `learning_rate`; `estimate_k_ipsi` and its tuner use `nepoch` and
`lr`, matching `estimate_k_ate`.

```python
import numpy as np
from gpi_pack.dyn_gpi import estimate_k_ipsi

delta_seq = np.array([0.5, 1.0, 2.0])

result = estimate_k_ipsi(
    R=R_text,                         # text hidden states: [N, T, F_text]
    R_video=R_video,                  # pooled video states: [N, T, D, H, W]
    W=W,                              # binary treatment; trailing padding is NaN
    Y=df_dynamic["OutcomeVar"].values, # scalar outcome: [N]
    delta_seq=delta_seq,
    K=5,                              # K-fold cross-fitting
    nepoch=200,                       # training epochs for Dynamic TarNet
    batch_size=32,
    lr=2e-5,                          # learning rate for Dynamic TarNet
    dropout=0.3,
    # Outcome model architecture. The final 1 is the scalar output layer.
    architecture_y=[16, 1],
    # Deconfounder architecture. Its last value is the representation size.
    architecture_z=[64, 32],
    # Multimodal encoder architectures
    text_hidden_dims=[1024, 256],
    text_out_dim=128,
    video_channels=[8, 16, 32],
    video_out_dim=128,
    n_boot=1000,                      # multiplier-bootstrap replications
    random_state=42,
)

print("Estimate:", result["est"])
print("Standard error:", result["se"])
print("Pointwise 95% interval:", result["ll1"], result["ul1"])
print("Uniform 95% band:", result["ll2"], result["ul2"])
```

Each value in `delta_seq` multiplies the treatment odds at every observed
segment. A value smaller than `1` decreases the treatment odds, `1` retains the
observed treatment mechanism, and a value larger than `1` increases the odds.
A one-dimensional `delta_seq` evaluates several constant interventions. You
can instead supply an array with shape `[J, T]` to evaluate `J` segment-specific
intervention schedules.

`result["est"]` and `result["se"]` contain the estimate and standard error for
each intervention. `ll1` and `ul1` are pointwise 95% confidence intervals.
Setting `n_boot` to a positive integer additionally calculates 95% Rademacher
multiplier-bootstrap bands in `ll2` and `ul2`, simultaneous over the supplied
intervention grid. With the default `n_boot=0`, `ll2` and `ul2` are `None`.
The returned `ifvals` array contains the unit-level influence-function values,
and `n_eff` is the number of units used for inference.

Supplying `R_video` automatically enables multimodal estimation. If it is
omitted, `R` can be supplied as either `[N, T, F]` or `[F, N, T]` for the
vector-only setup. Both modes use the same cross-fitting, nuisance-estimation,
and influence-function implementation. This API estimates a scalar-outcome
estimand (`Y` is `[N]`); it does not implement a repeated outcome trajectory
with `Y` shaped `[N, T]`.


### Hyperparameter Tuning

Install the optional [Optuna](https://optuna.org/) dependency:

```bash
pip install "gpi_pack[tune]"
```

The static tuner minimizes the best held-out factual-outcome MSE reached during
training. Native Python architectures are accepted; the older string format is
still supported.

```python
from gpi_pack.TarNet import TarNetHyperparameterTuner

tuner = TarNetHyperparameterTuner(
    T=df["TreatmentVar"].values,
    Y=df["OutcomeVar"].values,
    R=hidden_states,
    epoch=(100, 200),
    learning_rate=(1e-5, 1e-4),
    dropout=(0.1, 0.2),
    architecture_y=((200, 1), (100, 1)),
    architecture_z=((1024,), (2048,)),
)

study = tuner.tune(n_trials=100, refit=True)
print(tuner.best_params_)
best_model = tuner.best_model_
```

Dynamic GPI uses the same workflow. Supplying `R_video` automatically tunes
the multimodal encoders, and `best_params_` is ready to pass to
`estimate_k_ipsi`:

```python
from gpi_pack import DynamicGPIHyperparameterTuner, estimate_k_ipsi

tuner = DynamicGPIHyperparameterTuner(
    R=R_text,
    R_video=R_video,  # omit for vector-only representations
    W=W,              # [N, T], trailing padding is NaN
    Y=Y,
    nepoch=(100, 200),
    lr=(1e-5, 1e-3),
    architecture_y=((16, 1), (32, 16, 1)),
    architecture_z=((64, 32), (128, 64)),
)
study = tuner.tune(n_trials=50)

result = estimate_k_ipsi(
    R=R_text,
    R_video=R_video,
    W=W,
    Y=Y,
    delta_seq=delta_seq,
    **tuner.best_params_,
)
```

For strict causal cross-fitting, tune on an independent sample or within each
outer training fold; globally tuning on all outcomes can leak outcome
information into held-out causal estimates. These tuners optimize the factual
outcome network, not the downstream propensity or backward-regression models.
Use the default `n_jobs=1` for reproducible PyTorch trials; parallel tuning is
best run with process-isolated Optuna workers.

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
