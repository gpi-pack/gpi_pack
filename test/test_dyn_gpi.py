import inspect

import numpy as np
import pytest
import torch

import gpi_pack.dyn_gpi as dyn_gpi
from gpi_pack.dyn_gpi import DynamicTarNet


def _sequence_data():
    rng = np.random.default_rng(7)
    n, t, f = 5, 3, 4
    r = rng.normal(size=(n, t, f))
    w = rng.integers(0, 2, size=(n, t))
    mask = np.array(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
        dtype=float,
    )
    return rng, r, w, mask


def test_generic_model_preserves_input_layout_api_and_masks_padding():
    _, r, w, mask = _sequence_data()
    model = DynamicTarNet(
        architecture_y=[4, 1],
        architecture_z=[6],
        batch_size=2,
        verbose=False,
        device="cpu",
    )

    y_ntf, h_ntf, h_flat_ntf = model.predict(r, w, mask, return_rep=True)
    y_fnt, h_fnt, h_flat_fnt = model.predict(
        np.transpose(r, (2, 0, 1)), w, mask, return_rep=True
    )

    torch.testing.assert_close(y_ntf, y_fnt)
    torch.testing.assert_close(h_ntf, h_fnt)
    torch.testing.assert_close(h_flat_ntf, h_flat_fnt)
    assert y_ntf.shape == (r.shape[0], 1)
    assert h_ntf.shape == (r.shape[0], r.shape[1], 6)
    assert h_flat_ntf.shape == (r.shape[0], r.shape[1] * 6)
    assert torch.count_nonzero(h_ntf[torch.as_tensor(mask == 0)]) == 0


def test_multimodal_model_fuses_video_and_ignores_padded_inputs():
    rng, text, w, mask = _sequence_data()
    video_5d = rng.normal(size=(*text.shape[:2], 4, 4, 4))
    video_6d = video_5d[:, :, None, ...]
    model = DynamicTarNet(
        architecture_y=[3, 1],
        architecture_z=[5],
        batch_size=2,
        verbose=False,
        device="cpu",
        multimodal=True,
        text_input_dim=text.shape[-1],
        text_hidden_dims=(3,),
        text_out_dim=2,
        video_channels=(2,),
        video_out_dim=2,
    )

    y5, h5, h_flat5 = model.predict(
        text, w, mask, return_rep=True, R_video=video_5d
    )
    y6, h6, h_flat6 = model.predict(
        text, w, mask, return_rep=True, R_video=video_6d
    )
    torch.testing.assert_close(y5, y6)
    torch.testing.assert_close(h5, h6)
    torch.testing.assert_close(h_flat5, h_flat6)

    padded = mask == 0
    changed_text = text.copy()
    changed_video = video_5d.copy()
    changed_w = w.astype(float)
    changed_text[padded] = np.nan
    changed_video[padded] = np.nan
    changed_w[padded] = np.nan
    y_changed, h_changed, _ = model.predict(
        changed_text, changed_w, mask, return_rep=True, R_video=changed_video
    )
    torch.testing.assert_close(y5, y_changed)
    torch.testing.assert_close(h5, h_changed)
    assert torch.count_nonzero(h5[torch.as_tensor(padded)]) == 0

    model.model.train()
    model.optim.zero_grad()
    loss = model.model(
        torch.as_tensor(text, dtype=torch.float32),
        torch.as_tensor(w, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.float32),
        r_video=torch.as_tensor(video_6d, dtype=torch.float32),
    ).sum()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.model.video_encoder.parameters()
    )


def test_multimodal_shape_and_mask_validation():
    _, text, w, mask = _sequence_data()
    model = DynamicTarNet(
        architecture_y=[2, 1],
        architecture_z=[4],
        verbose=False,
        device="cpu",
        multimodal=True,
        text_input_dim=text.shape[-1],
        text_hidden_dims=(3,),
        text_out_dim=2,
        video_channels=(2,),
        video_out_dim=2,
    )

    with pytest.raises(ValueError, match="R_video is required"):
        model.predict(text, w, mask)
    with pytest.raises(ValueError, match="agree with W"):
        model.predict(text, w, mask, R_video=np.zeros((len(text) - 1, 3, 4, 4, 4)))

    nonbinary_w = w.astype(float)
    nonbinary_w[0, 0] = 0.5
    with pytest.raises(ValueError, match="finite and binary"):
        model.predict(
            text,
            nonbinary_w,
            mask,
            R_video=np.zeros((*text.shape[:2], 4, 4, 4)),
        )
    with pytest.raises(ValueError, match="finite and binary"):
        dyn_gpi.estimate_k_ipsi(
            text,
            nonbinary_w,
            np.zeros(len(text)),
            np.array([1.0]),
            K=2,
            verbose=False,
        )

    non_prefix_mask = mask.copy()
    non_prefix_mask[1] = [1, 0, 1]
    with pytest.raises(ValueError, match="left-aligned"):
        model.predict(
            text,
            w,
            non_prefix_mask,
            R_video=np.zeros((*text.shape[:2], 4, 4, 4)),
        )


@pytest.mark.parametrize(
    ("n", "batch_size", "valid_perc"),
    [(5, 4, 0.2), (8, 5, 0.25)],
)
def test_dynamic_batch_norm_handles_singleton_remainders(
    n, batch_size, valid_perc
):
    rng = np.random.default_rng(19)
    t = 2
    model = DynamicTarNet(
        architecture_y=[2, 1],
        architecture_z=[3],
        epochs=1,
        batch_size=batch_size,
        learning_rate=1e-3,
        dropout=0.0,
        bn=True,
        verbose=False,
        device="cpu",
    )
    score = model.fit(
        rng.normal(size=(n, t, 2)).astype(np.float32),
        rng.normal(size=n).astype(np.float32),
        rng.integers(0, 2, size=(n, t)).astype(np.float32),
        np.ones((n, t), dtype=np.float32),
        valid_perc=valid_perc,
        plot_loss=False,
    )

    assert np.isfinite(score)


def test_estimator_uses_one_pipeline_for_vector_and_multimodal_inputs(monkeypatch):
    calls = []
    init_kwargs = []
    masks_seen = []
    treatments_seen = []

    class DummyTarNet:
        def __init__(self, **kwargs):
            self.multimodal = kwargs["multimodal"]
            init_kwargs.append(kwargs.copy())
            calls.append(("init", self.multimodal, kwargs["text_input_dim"]))

        def fit(self, R, Y, W, mask, C=None, valid_perc=0.2, plot_loss=True, *, R_video=None):
            assert np.all(np.isfinite(R))
            assert R_video is None or np.all(np.isfinite(R_video))
            calls.append(
                (
                    "fit",
                    self.multimodal,
                    R.shape,
                    None if R_video is None else R_video.shape,
                    valid_perc,
                )
            )
            masks_seen.append(mask.copy())
            treatments_seen.append(W.copy())

        def predict(self, R, W, mask, C=None, return_rep=False, *, R_video=None):
            assert np.all(np.isfinite(R))
            assert R_video is None or np.all(np.isfinite(R_video))
            masks_seen.append(mask.copy())
            treatments_seen.append(W.copy())
            calls.append(
                (
                    "predict",
                    self.multimodal,
                    R.shape,
                    None if R_video is None else R_video.shape,
                    W.shape,
                )
            )
            n, t = W.shape
            y = torch.zeros((n, 1), dtype=torch.float32)
            h = torch.zeros((n, t, 2), dtype=torch.float32)
            return (y, h, h.reshape(n, -1)) if return_rep else y

    class DummyClassifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5, dtype=np.float32)

    class DummyRegressor:
        def predict(self, x):
            return np.zeros(len(x), dtype=np.float32)

    monkeypatch.setattr(dyn_gpi, "DynamicTarNet", DummyTarNet)
    monkeypatch.setattr(
        dyn_gpi, "_fit_downstream_classifier", lambda **kwargs: DummyClassifier()
    )
    monkeypatch.setattr(
        dyn_gpi, "_fit_downstream_regressor", lambda **kwargs: DummyRegressor()
    )

    rng = np.random.default_rng(11)
    n, t, f = 8, 2, 3
    text = rng.normal(size=(n, t, f))
    video = rng.normal(size=(n, t, 4, 4, 4))
    w_complete = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2, dtype=float)
    y = rng.normal(size=n)
    mask = np.ones((n, t), dtype=float)
    mask[[2, 4, 7], 1] = 0
    w = np.where(mask > 0, w_complete, np.nan)
    text[mask == 0] = np.nan
    video[mask == 0] = np.nan

    generic = dyn_gpi.estimate_k_ipsi(
        text,
        w,
        y,
        np.array([0.5, 2.0]),
        K=2,
        nepoch=7,
        lr=4e-4,
        architecture_y=[7, 1],
        architecture_z=[9, 5],
        valid_perc=0.25,
        step_size=2,
        bn=True,
        patience=3,
        min_delta=1e-3,
        verbose=False,
        device="cpu",
        n_boot=128,
    )
    multimodal = dyn_gpi.estimate_k_ipsi(
        text,
        w,
        y,
        np.array([0.5, 2.0]),
        K=2,
        verbose=False,
        device="cpu",
        n_boot=128,
        R_video=video,
        text_hidden_dims=(2,),
        text_out_dim=2,
        video_channels=(2,),
        video_out_dim=2,
    )

    assert generic["ifvals"].shape == multimodal["ifvals"].shape == (n, 2)
    assert generic["delta_paths"].shape == multimodal["delta_paths"].shape == (2, t)
    assert any(np.any(seen == 0) for seen in masks_seen)
    assert all(np.all(np.isfinite(seen)) for seen in treatments_seen)
    assert np.isnan(w).any()
    np.testing.assert_allclose(generic["ll2"], multimodal["ll2"])
    np.testing.assert_allclose(generic["ul2"], multimodal["ul2"])

    centered = generic["ifvals"] - generic["est"]
    active = generic["sigma"] > 0
    multipliers = np.random.default_rng(42).choice((-1.0, 1.0), size=(128, n))
    process = multipliers @ centered[:, active]
    process /= np.sqrt(generic["n_eff"]) * generic["sigma"][active]
    critical_value = np.quantile(np.max(np.abs(process), axis=1), 0.95)
    np.testing.assert_allclose(
        generic["ll2"], generic["est"] - critical_value * generic["se"]
    )
    np.testing.assert_allclose(
        generic["ul2"], generic["est"] + critical_value * generic["se"]
    )
    assert any(call[0] == "fit" and call[1] is False and call[3] is None for call in calls)
    assert any(call[0] == "fit" and call[1] is True and call[3] is not None for call in calls)
    assert any(
        call[0] == "fit" and call[1] is False and call[4] == pytest.approx(0.25)
        for call in calls
    )
    explicitly_configured = [
        kwargs
        for kwargs in init_kwargs
        if tuple(kwargs["architecture_y"]) == (7, 1)
        and tuple(kwargs["architecture_z"]) == (9, 5)
    ]
    assert explicitly_configured
    assert all(kwargs["epochs"] == 7 for kwargs in explicitly_configured)
    assert all(
        kwargs["learning_rate"] == pytest.approx(4e-4)
        for kwargs in explicitly_configured
    )
    assert all(kwargs["step_size"] == 2 for kwargs in explicitly_configured)
    assert all(kwargs["bn"] is True for kwargs in explicitly_configured)
    assert all(kwargs["patience"] == 3 for kwargs in explicitly_configured)
    assert all(
        kwargs["min_delta"] == pytest.approx(1e-3)
        for kwargs in explicitly_configured
    )

    calls.clear()
    without_bootstrap = dyn_gpi.estimate_k_ipsi(
        text[:5],
        w[:5],
        y[:5],
        np.array([1.0]),
        K=4,
        verbose=False,
        device="cpu",
    )
    assert without_bootstrap["ll2"] is None
    assert without_bootstrap["ul2"] is None
    assert sum(call[0] == "init" for call in calls) == 4
    assert all(call[4][0] > 0 for call in calls if call[0] == "predict")

    zero_variance = dyn_gpi.estimate_k_ipsi(
        np.nan_to_num(text),
        w_complete,
        np.zeros(n),
        np.array([1.0]),
        K=2,
        verbose=False,
        device="cpu",
        n_boot=16,
    )
    np.testing.assert_array_equal(zero_variance["ll2"], zero_variance["est"])
    np.testing.assert_array_equal(zero_variance["ul2"], zero_variance["est"])


def test_estimator_infers_multimodal_mode_from_video():
    signature = inspect.signature(dyn_gpi.estimate_k_ipsi)
    assert list(signature.parameters)[:4] == ["R", "W", "Y", "delta_seq"]
    for name in ("K", "nepoch", "lr", "architecture_y", "architecture_z"):
        assert name in signature.parameters
    for legacy_name in (
        "nsplits",
        "epochs",
        "learning_rate",
        "architecture_fr",
        "head_hidden",
    ):
        assert legacy_name not in signature.parameters
    assert "mask" not in signature.parameters
    assert "multimodal" not in signature.parameters
    assert signature.parameters["n_boot"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["n_boot"].default == 0
    assert signature.parameters["R_video"].kind is inspect.Parameter.KEYWORD_ONLY

    model_signature = inspect.signature(dyn_gpi.DynamicTarNet)
    assert "architecture_y" in model_signature.parameters
    assert "architecture_z" in model_signature.parameters
    assert "include_Si_in_head" in model_signature.parameters
    assert "include_Si_in_rep" in model_signature.parameters
    assert "architecture_fr" not in model_signature.parameters
    assert "head_hidden" not in model_signature.parameters
    assert "include_Ti_in_head" not in model_signature.parameters
    assert "include_Ti_in_rep" not in model_signature.parameters

    base_signature = inspect.signature(dyn_gpi.DynamicTarNetBase)
    assert "sizes_y" in base_signature.parameters
    assert "sizes_z" in base_signature.parameters
    assert "include_Si_in_head" in base_signature.parameters
    assert "include_Si_in_rep" in base_signature.parameters
    assert "architecture_fr" not in base_signature.parameters
    assert "head_hidden" not in base_signature.parameters
    assert "include_Ti_in_head" not in base_signature.parameters
    assert "include_Ti_in_rep" not in base_signature.parameters


@pytest.mark.parametrize("n_boot", [-1, True, 1.5])
def test_estimator_rejects_invalid_n_boot(n_boot):
    with pytest.raises(ValueError, match="n_boot must be a nonnegative integer"):
        dyn_gpi.estimate_k_ipsi(
            np.zeros((2, 1, 1)),
            np.zeros((2, 1)),
            np.zeros(2),
            np.array([1.0]),
            n_boot=n_boot,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("w", "message"),
    [
        (np.array([[0.0, np.nan, 1.0], [1.0, 0.0, np.nan]]), "must be trailing"),
        (np.array([[0.0, np.inf], [1.0, 0.0]]), "finite and binary"),
        (np.array([[0.0, np.nan], [np.nan, np.nan]]), "at least one observed"),
    ],
)
def test_estimator_rejects_invalid_nan_padding(w, message):
    with pytest.raises(ValueError, match=message):
        dyn_gpi.estimate_k_ipsi(
            np.zeros((len(w), w.shape[1], 1)),
            w,
            np.zeros(len(w)),
            np.array([1.0]),
            K=2,
            verbose=False,
        )
