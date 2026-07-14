from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gpi_pack.video as video_module
from gpi_pack.video import CosmosVideoExtractor, extract_videos


class FakeDistribution:
    def __init__(self, mode_value, sample_value):
        self.mode_value = mode_value
        self.sample_value = sample_value
        self.mode_calls = 0
        self.sample_calls = 0

    def mode(self):
        self.mode_calls += 1
        return self.mode_value

    def sample(self, generator=None):
        del generator
        self.sample_calls += 1
        return self.sample_value


class FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(spatial_compression_ratio=4)
        self.decode_calls = 0
        self.encoded_video = None
        self.dist = None

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def eval(self):
        return self

    def encode(self, video):
        self.encoded_video = video.clone()
        batch, _, frames, _, _ = video.shape
        latent = torch.arange(
            batch * 2 * frames * 2 * 3,
            dtype=video.dtype,
            device=video.device,
        ).reshape(batch, 2, frames, 2, 3)
        self.dist = FakeDistribution(latent, latent + 100)
        return SimpleNamespace(latent_dist=self.dist)

    def post_quant_conv(self, latent):
        return latent * 2 + 1

    def decode(self, latent):
        self.decode_calls += 1
        self.last_decoder_input = self.post_quant_conv(latent).clone()
        return SimpleNamespace(sample=self.encoded_video.clone())


def make_frames(num_frames=3, height=5, width=7):
    values = np.arange(num_frames * height * width * 3, dtype=np.uint16) % 256
    return values.astype(np.uint8).reshape(num_frames, height, width, 3)


def make_extractor(**kwargs):
    vae = FakeVAE()
    extractor = CosmosVideoExtractor(
        vae=vae,
        device="cpu",
        dtype="fp32",
        max_frames=None,
        **kwargs,
    )
    return extractor, vae


def test_reconstruct_video_pads_crops_and_temporally_pools():
    frames = make_frames()
    extractor, vae = make_extractor(pad_multiple=4)

    result = extractor.reconstruct_video(frames)

    assert tuple(vae.encoded_video.shape) == (1, 3, 3, 8, 8)
    assert result.input_shape_bcthw == (1, 3, 3, 5, 7)
    assert result.padded_shape_bcthw == (1, 3, 3, 8, 8)
    assert (result.pad_bottom, result.pad_right) == (3, 1)
    assert tuple(result.reconstruction.shape) == (1, 3, 3, 5, 7)
    np.testing.assert_array_equal(
        video_module.bcthw_to_uint8(result.reconstruction), frames
    )

    expected_decoder_input = vae.post_quant_conv(vae.dist.mode_value)
    torch.testing.assert_close(result.decoder_input, expected_decoder_input)
    torch.testing.assert_close(
        result.representation, expected_decoder_input.mean(dim=2)
    )
    assert tuple(result.representation.shape) == (1, 2, 2, 3)


def test_encode_video_uses_deterministic_cosmos_mode():
    extractor, vae = make_extractor(pad_multiple=4)

    result = extractor.encode_video(make_frames())

    assert vae.dist.mode_calls == 1
    assert vae.dist.sample_calls == 0
    torch.testing.assert_close(result.latent, vae.dist.mode_value)
    torch.testing.assert_close(
        result.representation,
        vae.post_quant_conv(vae.dist.mode_value).mean(dim=2),
    )


def test_temporal_pooling_none_preserves_decoder_input():
    extractor, vae = make_extractor(
        pad_multiple=4,
        temporal_pooling="none",
    )

    result = extractor.encode_video(make_frames())

    torch.testing.assert_close(result.representation, result.decoder_input)
    assert tuple(result.representation.shape) == (1, 2, 3, 2, 3)
    torch.testing.assert_close(
        result.decoder_input, vae.post_quant_conv(vae.dist.mode_value)
    )


def test_encode_video_skips_decoder():
    extractor, vae = make_extractor(pad_multiple=4)

    result = extractor.encode_video(make_frames())

    assert result.reconstruction is None
    assert vae.decode_calls == 0


def test_extract_videos_saves_paper_ready_payloads(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.touch()
    frames = make_frames()
    metadata = [
        video_module.segment_meta(index, index * 3, index * 3 + 2, 3.0)
        for index in range(3)
    ]
    monkeypatch.setattr(
        video_module,
        "iter_segments",
        lambda *args, **kwargs: iter((frames, meta) for meta in metadata),
    )
    extractor, vae = make_extractor(pad_multiple=4)

    outputs = extract_videos(
        source,
        tmp_path / "hidden",
        segment_seconds=1,
        segment=(1, 2),
        save_latent=True,
        save_decoder_input=True,
        extractor=extractor,
        verbose=False,
    )

    assert [output.segment_index for output in outputs] == [1, 2]
    assert vae.decode_calls == 0
    for output in outputs:
        payload = torch.load(output.representation_path, weights_only=True)
        assert payload["schema_version"] == 1
        assert payload["video_path"] == str(source)
        assert payload["latent_selection"] == "mode"
        assert payload["temporal_pooling"] == "temporal_mean"
        assert payload["library_versions"]["torch"] == str(torch.__version__)
        assert payload["vae_config"]["spatial_compression_ratio"] == 4
        assert payload["representation_shape"] == (2, 2, 3)
        assert tuple(payload["representation"].shape) == (2, 2, 3)
        assert tuple(payload["latent"].shape) == (2, 3, 2, 3)
        assert tuple(payload["decoder_input"].shape) == (2, 3, 2, 3)
        torch.testing.assert_close(
            payload["pre_decoder_hidden"], payload["decoder_input"]
        )
        assert payload["model_frame_count"] == 3
        assert payload["selected_frame_indices"] == (0, 1, 2)


def test_reconstruction_is_written_only_when_requested(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.touch()
    frames = make_frames()
    meta = video_module.segment_meta(0, 0, 2, 3.0)
    monkeypatch.setattr(
        video_module,
        "iter_segments",
        lambda *args, **kwargs: iter([(frames, meta)]),
    )
    extractor = CosmosVideoExtractor(
        vae=FakeVAE(),
        device="cpu",
        dtype="fp32",
        pad_multiple=4,
        max_frames=2,
    )
    saved = []
    monkeypatch.setattr(
        video_module,
        "save_mp4",
        lambda output_frames, path, fps: saved.append((output_frames, path, fps)),
    )

    outputs = extract_videos(
        source,
        tmp_path / "hidden",
        output_video_dir=tmp_path / "reconstructed",
        segment_seconds=1,
        extractor=extractor,
        verbose=False,
    )

    assert len(saved) == 1
    saved_frames, saved_path, saved_fps = saved[0]
    np.testing.assert_array_equal(saved_frames, frames[[0, 2]])
    assert saved_path == outputs[0].reconstruction_path
    assert saved_fps == pytest.approx(2.0)


def test_find_videos_filters_sorts_and_rejects_bad_input(tmp_path):
    (tmp_path / "b.mov").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.MP4").touch()
    unsupported = tmp_path / "notes.txt"
    unsupported.touch()

    found = video_module.find_videos(tmp_path)

    assert found == sorted([tmp_path / "b.mov", nested / "a.MP4"])
    with pytest.raises(ValueError, match="Unsupported video extension"):
        video_module.find_videos(unsupported)
    with pytest.raises(FileNotFoundError):
        video_module.find_videos(tmp_path / "missing.mp4")


def test_save_mp4_rejects_odd_dimensions_before_loading_codec(tmp_path):
    frames = np.zeros((2, 5, 7, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="even frame dimensions"):
        video_module.save_mp4(frames, tmp_path / "odd.mp4", fps=2)


def test_extract_videos_rejects_same_input_and_output_directory(tmp_path):
    (tmp_path / "input.mp4").touch()
    extractor, _ = make_extractor(pad_multiple=4)

    with pytest.raises(ValueError, match="Input and output directories"):
        extract_videos(
            tmp_path,
            tmp_path,
            segment_seconds=1,
            extractor=extractor,
            verbose=False,
        )


def test_cli_parser_exposes_package_pipeline_options():
    args = video_module.parse_args(
        [
            "--input",
            "input.mp4",
            "--output_dir",
            "outputs",
            "--segment_seconds",
            "5",
            "--segment",
            "1-3",
            "--frame_size",
            "320",
            "480",
            "--max_frames",
            "0",
            "--save_reconstruction",
        ]
    )

    assert args.segment == (1, 3)
    assert args.frame_size == [320, 480]
    assert args.max_frames == 0
    assert args.save_reconstruction is True


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((3, 5, 7), dtype=np.uint8),
        np.zeros((3, 5, 7, 1), dtype=np.uint8),
        np.empty((0, 5, 7, 3), dtype=np.uint8),
    ],
)
def test_invalid_frame_shapes_raise(frames):
    extractor, _ = make_extractor(pad_multiple=4)

    with pytest.raises(ValueError):
        extractor.encode_video(frames)


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="pad_multiple"):
        CosmosVideoExtractor(
            vae=FakeVAE(), device="cpu", dtype="fp32", pad_multiple=0
        )
    with pytest.raises(ValueError, match="segment_seconds"):
        extract_videos([], "unused", segment_seconds=0, extractor=None)
