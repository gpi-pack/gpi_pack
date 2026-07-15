import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


tarnet_module = importlib.import_module("gpi_pack.TarNet")
dyn_gpi = importlib.import_module("gpi_pack.dyn_gpi")


class FakeTrial:
    def __init__(self, number=0, values=None):
        self.number = number
        self.values = values or {}
        self.params = {}
        self.user_attrs = {}
        self.calls = []
        self.reports = []
        self.value = None

    def suggest_float(self, name, low, high, *, log=False):
        self.calls.append(("float", name, low, high, log))
        value = self.values.get(name, low)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        self.calls.append(("int", name, low, high))
        value = self.values.get(name, low)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices)))
        value = self.values.get(name, choices[0])
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def report(self, value, step):
        self.reports.append((step, value))

    def should_prune(self):
        return False


class FakeStudy:
    def __init__(self, direction="minimize"):
        self.direction = direction
        self.trials = []
        self.best_trial = None

    def optimize(self, objective, n_trials, **kwargs):
        for number in range(n_trials):
            trial = FakeTrial(number=number)
            trial.value = objective(trial)
            self.trials.append(trial)
        self.best_trial = min(self.trials, key=lambda trial: trial.value)


def fake_optuna(created):
    class TPESampler:
        def __init__(self, seed):
            self.seed = seed

    class MedianPruner:
        def __init__(self, n_startup_trials):
            self.n_startup_trials = n_startup_trials

    def create_study(**kwargs):
        created.update(kwargs)
        return FakeStudy(direction=kwargs["direction"])

    return SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=TPESampler),
        pruners=SimpleNamespace(MedianPruner=MedianPruner),
        create_study=create_study,
    )


def test_tarnet_tuner_uses_typed_search_space_and_refits(monkeypatch):
    instances = []

    class FakeTarNet:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.best_valid_loss = None
            self.fit_calls = []
            instances.append(self)

        def fit(self, R, Y, T, **kwargs):
            self.fit_calls.append((R.copy(), Y.copy(), T.copy(), kwargs))
            score = float(self.kwargs["dropout"] + self.kwargs["learning_rate"])
            callback = kwargs.get("epoch_callback")
            if callback is not None:
                callback(0, score)
            self.best_valid_loss = score
            return score

    monkeypatch.setattr(tarnet_module, "TarNet", FakeTarNet)
    created = {}
    monkeypatch.setattr(tarnet_module, "require_optuna", lambda: fake_optuna(created))

    R = np.arange(24, dtype=float).reshape(8, 3)
    T = np.array([0, 1] * 4)
    Y = np.linspace(0, 1, 8)
    C = np.arange(16, dtype=float).reshape(8, 2)
    conv_layers = [{"in_channels": 1, "out_channels": 2, "kernel_size": 1}]
    tuner = tarnet_module.TarNetHyperparameterTuner(
        T,
        Y,
        R,
        C=C,
        epoch=["3", "4"],
        batch_size=(2, 4),
        learning_rate=(1e-3, 1e-4),
        dropout=(0.1, 0.2),
        architecture_y=("[4, 1]", "[2, 1]"),
        architecture_z=[8, 4],
        conv_layers=conv_layers,
        patience_min=2,
        patience_max=2,
    )

    study = tuner.tune(n_trials=2, refit=True)

    assert created["direction"] == "minimize"
    assert len(instances) == 3
    assert instances[0] is not instances[1] is not instances[2]
    assert tuner.best_model_ is instances[-1]
    assert tuner.best_params_["epochs"] == 3
    assert tuner.best_params_["architecture_y"] == [4, 1]
    assert tuner.best_params_["architecture_z"] == [8, 4]
    assert all(model.kwargs["model_dir"] is None for model in instances[:2])
    assert instances[0].kwargs["random_state"] == instances[1].kwargs["random_state"]
    assert instances[-1].fit_calls[0][3]["C"] is C
    assert instances[0].fit_calls[0][0].shape == R.shape
    assert study.best_trial.reports

    calls = study.trials[0].calls
    assert ("float", "learning_rate", 1e-4, 1e-3, True) in calls
    assert not any(call[1] in {"conv_layers", "conv_activation"} for call in calls)


def test_tarnet_formula_confounders_are_not_concatenated_twice():
    R = np.ones((6, 2))
    data = pd.DataFrame({"x": np.arange(6)})
    tuner = tarnet_module.TarNetHyperparameterTuner(
        np.array([0, 1, 0, 1, 0, 1]),
        np.arange(6),
        R,
        formula_C="x",
        data=data,
        epoch=1,
        architecture_y=[1],
        architecture_z=[2],
        patience_min=1,
        patience_max=1,
    )
    assert tuner.R.shape == (6, 2)
    assert tuner.C.shape == (6, 1)

    direct_tuner = tarnet_module.TarNetHyperparameterTuner(
        np.array([0, 1, 0, 1, 0, 1]),
        np.arange(6),
        R,
        C=np.arange(6),
        epoch=1,
        architecture_y=[1],
        architecture_z=[2],
        patience_min=1,
        patience_max=1,
    )
    assert direct_tuner.C.shape == (6, 1)


def test_tarnet_validation_is_sample_weighted():
    class ZeroModel(torch.nn.Module):
        def forward(self, inputs, treatments, confounders=None):
            return torch.zeros_like(treatments), inputs

    model = tarnet_module.TarNet(
        epochs=1,
        batch_size=2,
        architecture_y=[1],
        architecture_z=[2],
        verbose=False,
    )
    model.model = ZeroModel()
    model.valid_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.zeros((3, 1)),
            torch.zeros((3, 1)),
            torch.tensor([[0.0], [0.0], [3.0]]),
        ),
        batch_size=2,
    )
    assert float(model.validate_step()) == pytest.approx(3.0)


def test_tarnet_fit_with_confounders_returns_best_validation_loss():
    n = 10
    model = tarnet_module.TarNet(
        epochs=2,
        batch_size=3,
        learning_rate=1e-3,
        architecture_y=[1],
        architecture_z=[4],
        dropout=0.0,
        verbose=False,
        random_state=7,
    )
    score = model.fit(
        np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 30,
        np.linspace(0, 1, n, dtype=np.float32),
        np.array([0, 1] * (n // 2), dtype=np.float32),
        C=np.linspace(-1, 1, n, dtype=np.float32)[:, None],
        valid_perc=0.2,
        plot_loss=False,
    )

    assert np.isfinite(score)
    assert score == pytest.approx(model.best_valid_loss)
    assert float(model.valid_loss) == pytest.approx(score)


@pytest.mark.parametrize(
    ("n", "batch_size", "valid_perc"),
    [(5, 4, 0.2), (8, 5, 0.25)],
)
def test_tarnet_batch_norm_handles_singleton_remainders(
    n, batch_size, valid_perc
):
    model = tarnet_module.TarNet(
        epochs=1,
        batch_size=batch_size,
        learning_rate=1e-3,
        architecture_y=[1],
        architecture_z=[2],
        dropout=0.0,
        bn=True,
        verbose=False,
    )
    score = model.fit(
        np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 10,
        np.linspace(0, 1, n, dtype=np.float32),
        np.arange(n, dtype=np.float32) % 2,
        valid_perc=valid_perc,
        plot_loss=False,
    )

    assert np.isfinite(score)


def test_dynamic_tuner_infers_vector_and_multimodal_inputs(monkeypatch):
    tuner_signature = inspect.signature(dyn_gpi.DynamicGPIHyperparameterTuner)
    for name in ("nepoch", "lr", "architecture_y", "architecture_z"):
        assert name in tuner_signature.parameters
    for legacy_name in (
        "epochs",
        "learning_rate",
        "architecture_fr",
        "head_hidden",
    ):
        assert legacy_name not in tuner_signature.parameters

    instances = []

    class FakeDynamicTarNet:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.best_valid_loss = None
            self.fit_kwargs = None
            instances.append(self)

        def fit(self, R, Y, W, mask, **kwargs):
            self.fit_kwargs = kwargs
            self.R = R.copy()
            self.W = W.copy()
            self.mask = mask.copy()
            score = float(self.kwargs["dropout"] + 0.25)
            callback = kwargs.get("epoch_callback")
            if callback is not None:
                callback(0, score)
            self.best_valid_loss = score
            return score

    monkeypatch.setattr(dyn_gpi, "DynamicTarNet", FakeDynamicTarNet)

    n, t, f = 6, 3, 2
    W = np.array(
        [[0, 1, 0], [1, 0, np.nan], [0, np.nan, np.nan]] * 2,
        dtype=float,
    )
    mask = ~np.isnan(W)
    R = np.arange(f * n * t, dtype=float).reshape(f, n, t)
    R[:, ~mask] = np.nan
    Y = np.arange(n, dtype=float)
    trial = FakeTrial()
    vector_tuner = dyn_gpi.DynamicGPIHyperparameterTuner(
        R,
        W,
        Y,
        nepoch=2,
        batch_size=2,
        lr=1e-3,
        dropout=0.1,
        architecture_y=[2, 1],
        architecture_z=[4, 2],
    )
    assert vector_tuner.objective(trial) == pytest.approx(0.35)
    vector_model = instances[-1]
    assert vector_model.kwargs["multimodal"] is False
    assert vector_model.fit_kwargs["R_video"] is None
    np.testing.assert_array_equal(vector_model.mask, mask.astype(np.float32))
    assert np.all(np.isfinite(vector_model.W))
    assert np.all(np.isfinite(vector_model.R))
    vector_params = vector_tuner._resolved_params_by_trial[0]
    assert vector_params["nepoch"] == 2
    assert vector_params["lr"] == pytest.approx(1e-3)
    assert vector_params["architecture_y"] == [2, 1]
    assert vector_params["architecture_z"] == [4, 2]
    assert not {
        "epochs",
        "learning_rate",
        "architecture_fr",
        "head_hidden",
    } & set(vector_params)
    assert vector_model.kwargs["epochs"] == 2
    assert vector_model.kwargs["learning_rate"] == pytest.approx(1e-3)
    assert vector_model.kwargs["architecture_y"] == [2, 1]
    assert vector_model.kwargs["architecture_z"] == [4, 2]

    text = np.transpose(np.nan_to_num(R), (1, 2, 0))
    video = np.zeros((n, t, 2, 2, 2, 2), dtype=float)
    text[~mask] = np.nan
    video[~mask] = np.nan
    multimodal_tuner = dyn_gpi.DynamicGPIHyperparameterTuner(
        text,
        W,
        Y,
        R_video=video,
        nepoch=2,
        batch_size=2,
        lr=1e-3,
        dropout=0.1,
        architecture_y=[2, 1],
        architecture_z=[4, 2],
        text_hidden_dims=[3],
        text_out_dim=2,
        video_channels=[2],
        video_out_dim=2,
    )
    multimodal_tuner.objective(FakeTrial())
    multimodal_model = instances[-1]
    assert multimodal_model.kwargs["multimodal"] is True
    assert multimodal_model.kwargs["text_input_dim"] == f
    assert multimodal_model.kwargs["video_in_channels"] == 2
    assert multimodal_model.fit_kwargs["R_video"].shape == (n, t, 2, 2, 2, 2)
    assert multimodal_tuner._resolved_params_by_trial[0]["video_in_channels"] == 2

    estimator_parameters = set(inspect.signature(dyn_gpi.estimate_k_ipsi).parameters)
    assert set(multimodal_tuner._resolved_params_by_trial[0]) <= estimator_parameters
    assert (
        dyn_gpi.DynamicTarNetHyperparameterTuner
        is dyn_gpi.DynamicGPIHyperparameterTuner
    )


def test_dynamic_tuner_normalizes_time_varying_confounders():
    n, t, f, c = 5, 3, 2, 4
    W = np.array(
        [[0, 1, 0], [1, 0, np.nan], [0, np.nan, np.nan], [1, 1, 0], [0, 1, np.nan]],
        dtype=float,
    )
    observed = np.isfinite(W)
    R = np.zeros((f, n, t), dtype=float)
    C = np.arange(c * n * t, dtype=float).reshape(c, n, t)
    C[:, ~observed] = np.nan

    tuner = dyn_gpi.DynamicGPIHyperparameterTuner(
        R,
        W,
        np.arange(n, dtype=float),
        C=C,
        nepoch=1,
        architecture_y=[2, 1],
        architecture_z=[4],
    )

    assert tuner.C.shape == (n, t, c)
    assert np.all(np.isfinite(tuner.C))
    assert np.all(tuner.C[~observed] == 0)


def test_best_trial_attrs_take_priority_over_stale_trial_cache():
    tuner = tarnet_module.TarNetHyperparameterTuner(
        [0, 1, 0, 1],
        [0.0, 1.0, 0.0, 1.0],
        np.zeros((4, 2)),
        epoch=1,
        architecture_y=[1],
        architecture_z=[2],
        patience_min=1,
        patience_max=1,
    )
    tuner._resolved_params_by_trial[0] = {"epochs": 999}
    resolved = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "dropout": 0.0,
        "step_size": None,
        "architecture_y": [1],
        "architecture_z": [2],
        "bn": False,
        "patience": 1,
    }
    best_trial = SimpleNamespace(
        number=0,
        value=0.25,
        user_attrs={"resolved_params": resolved},
    )

    tuner._set_best_from_study(
        SimpleNamespace(direction="minimize", best_trial=best_trial)
    )

    assert tuner.best_params_ == resolved


def test_tune_has_actionable_optional_dependency_error(monkeypatch):
    tuner = tarnet_module.TarNetHyperparameterTuner(
        [0, 1, 0, 1],
        [0.0, 1.0, 0.0, 1.0],
        np.zeros((4, 2)),
        epoch=1,
        architecture_y=[1],
        architecture_z=[2],
        patience_min=1,
        patience_max=1,
    )

    def missing_optuna():
        raise ImportError("Optuna is required; install gpi_pack[tune]")

    monkeypatch.setattr(tarnet_module, "require_optuna", missing_optuna)
    with pytest.raises(ImportError, match=r"gpi_pack\[tune\]"):
        tuner.tune(n_trials=1)


def test_tuners_reject_non_minimizing_studies(monkeypatch):
    monkeypatch.setattr(tarnet_module, "require_optuna", lambda: object())
    monkeypatch.setattr(dyn_gpi, "require_optuna", lambda: object())

    static_tuner = tarnet_module.TarNetHyperparameterTuner(
        [0, 1, 0, 1],
        [0.0, 1.0, 0.0, 1.0],
        np.zeros((4, 2)),
        epoch=1,
        architecture_y=[1],
        architecture_z=[2],
        patience_min=1,
        patience_max=1,
    )
    dynamic_tuner = dyn_gpi.DynamicGPIHyperparameterTuner(
        np.zeros((2, 4, 2)),
        np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float),
        np.arange(4, dtype=float),
        nepoch=1,
        architecture_y=[1],
        architecture_z=[2],
    )

    for tuner in (static_tuner, dynamic_tuner):
        study = FakeStudy(direction="maximize")
        with pytest.raises(ValueError, match="minimize"):
            tuner.tune(n_trials=1, study=study)
        with pytest.raises(ValueError, match="minimize"):
            tuner.fit_best(study=study)
        assert study.trials == []
