"""Reconstruct videos with Cosmos and extract paper-ready representations.

The public API mirrors :mod:`gpi_pack.diffusion`: ``CosmosVideoExtractor``
owns the model and provides in-memory encode/decode methods, while
``extract_videos`` handles file discovery, segmentation, and serialization.

For each segment, the representation used by the video paper is the Cosmos
decoder input (``vae.post_quant_conv(latent)``) averaged over its latent-time
axis.  The saved representation therefore has shape ``[C, H, W]``.  The
unpooled decoder input and latent can also be saved for diagnostic use.

Video files are visual-only in this pipeline: audio is neither passed through
Cosmos nor written to reconstructed MP4 files.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_COSMOS_MODEL_ID = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"
DEFAULT_MAX_FRAMES = 121
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VIDEO_PAYLOAD_SCHEMA_VERSION = 1

PathLike = Union[str, Path]
SegmentSpec = Tuple[int, Optional[int]]
FrameArray = Union[np.ndarray, torch.Tensor]

__all__ = [
    "CosmosVideoExtractor",
    "VideoExtractionResult",
    "VideoSegmentOutput",
    "extract_videos",
]


@dataclass(frozen=True)
class VideoExtractionResult:
    """In-memory result for one image or video clip.

    All tensors retain a batch dimension. ``representation`` is ``[B,C,H,W]``
    for the default temporal-mean pooling and ``[B,C,D,H,W]`` when pooling is
    disabled. ``reconstruction`` is populated only by ``reconstruct_video``.
    """

    latent: torch.Tensor
    decoder_input: torch.Tensor
    representation: torch.Tensor
    reconstruction: Optional[torch.Tensor]
    input_shape_bcthw: Tuple[int, ...]
    padded_shape_bcthw: Tuple[int, ...]
    pad_bottom: int
    pad_right: int
    original_num_frames: int
    selected_frame_indices: Tuple[int, ...]
    temporal_pooling: str

    @property
    def pre_decoder_hidden(self) -> torch.Tensor:
        """Backward-compatible name for the unpooled decoder input."""

        return self.decoder_input


@dataclass(frozen=True)
class VideoSegmentOutput:
    """Paths written for one processed video segment."""

    video_path: Path
    segment_index: int
    representation_path: Path
    reconstruction_path: Optional[Path] = None


def parse_segment_spec(value: str) -> SegmentSpec:
    """Parse ``N``, ``START-END``, or ``START-`` into an inclusive range."""

    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("--segment cannot be empty.")

    if "-" not in text:
        try:
            segment = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--segment must be an integer or an inclusive range like 0-5."
            ) from exc
        if segment < 0:
            raise argparse.ArgumentTypeError("--segment indices must be non-negative.")
        return segment, segment

    start_text, end_text = text.split("-", 1)
    if not start_text:
        raise argparse.ArgumentTypeError("--segment range must include a start index.")

    try:
        start = int(start_text)
        end = int(end_text) if end_text else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--segment must be an integer or an inclusive range like 0-5."
        ) from exc

    if start < 0 or (end is not None and end < 0):
        raise argparse.ArgumentTypeError("--segment indices must be non-negative.")
    if end is not None and end < start:
        raise argparse.ArgumentTypeError("--segment range end must be >= start.")
    return start, end


def format_segment_spec(segment_spec: SegmentSpec) -> str:
    """Format an inclusive segment range for display."""

    start, end = segment_spec
    if end is None:
        return f"{start}-"
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _normalize_segment_spec(
    segment: Optional[Union[int, SegmentSpec]],
) -> Optional[SegmentSpec]:
    if segment is None:
        return None
    if isinstance(segment, int):
        if segment < 0:
            raise ValueError("segment indices must be non-negative.")
        return segment, segment
    if not isinstance(segment, tuple) or len(segment) != 2:
        raise TypeError("segment must be an integer or a (start, end) tuple.")
    start, end = segment
    if not isinstance(start, int) or not (isinstance(end, int) or end is None):
        raise TypeError("segment bounds must be integers; the end may also be None.")
    if start < 0 or (end is not None and end < start):
        raise ValueError("segment must satisfy 0 <= start <= end.")
    return start, end


def model_dtype(
    dtype: Union[str, torch.dtype], device: torch.device
) -> torch.dtype:
    """Resolve a public dtype option for the selected device."""

    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        names = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype == "auto":
            resolved = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float32
            )
        elif dtype in names:
            resolved = names[dtype]
        else:
            raise ValueError("dtype must be one of: auto, bf16, fp16, fp32.")

    if device.type == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        raise ValueError(
            "Use fp32/auto on CPU; Cosmos half precision requires a supported GPU."
        )
    return resolved


def _installed_version(package: str) -> Optional[str]:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def find_videos(
    path: PathLike, *, exclude: Sequence[PathLike] = ()
) -> List[Path]:
    """Find supported video files, raising for invalid inputs."""

    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Video input does not exist: {input_path}")

    excluded_roots = [Path(item).expanduser() for item in exclude]
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            supported = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise ValueError(
                f"Unsupported video extension '{input_path.suffix}'. "
                f"Supported: {supported}."
            )
        return [input_path]

    videos = []
    for candidate in input_path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if any(_is_within(candidate, root) for root in excluded_roots):
            continue
        videos.append(candidate)
    return sorted(videos)


def _import_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "Reading video files requires PyAV. Install gpi_pack with the "
            "video extra: pip install 'gpi_pack[video]'."
        ) from exc
    return av


def _import_imageio() -> Any:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Writing reconstructed videos requires imageio and ffmpeg. Install "
            "gpi_pack with the video extra: pip install 'gpi_pack[video]'."
        ) from exc
    return imageio


def fps_for(stream: Any) -> float:
    """Return a finite positive nominal frame rate for a PyAV stream."""

    rate = stream.average_rate or stream.base_rate or stream.guessed_rate or 30.0
    fps = float(rate)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video stream has an invalid frame rate: {fps!r}.")
    return fps


def segment_meta(
    segment_idx: int, start_frame: int, end_frame: int, fps: float
) -> Dict[str, Union[int, float]]:
    """Create nominal frame/time metadata for a decoded segment."""

    return {
        "segment_index": segment_idx,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": end_frame - start_frame + 1,
        "start_time_sec": start_frame / fps,
        "end_time_sec": (end_frame + 1) / fps,
        "duration_sec": (end_frame - start_frame + 1) / fps,
    }


def iter_segments(
    video_path: PathLike, segment_seconds: float, drop_last: bool = False
) -> Iterator[Tuple[np.ndarray, Dict[str, Union[int, float]]]]:
    """Decode a video into consecutive RGB segments.

    Segments use the stream's nominal FPS and therefore contain a fixed number
    of decoded frames. This is exact for constant-frame-rate media and nominal
    for variable-frame-rate media; the chosen FPS and frame bounds are saved in
    every output payload.
    """

    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("segment_seconds must be finite and greater than zero.")

    av = _import_av()
    path = Path(video_path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}.")
        stream = container.streams.video[0]
        fps = fps_for(stream)
        frames_per_segment = max(1, int(round(segment_seconds * fps)))
        frames: List[np.ndarray] = []
        start_frame = 0
        segment_idx = 0

        for frame_idx, frame in enumerate(container.decode(stream)):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) < frames_per_segment:
                continue

            yield np.stack(frames), segment_meta(
                segment_idx, start_frame, frame_idx, fps
            )
            frames.clear()
            start_frame = frame_idx + 1
            segment_idx += 1

        if frames and not drop_last:
            end_frame = start_frame + len(frames) - 1
            yield np.stack(frames), segment_meta(
                segment_idx, start_frame, end_frame, fps
            )


def uniform_frame_indices(num_frames: int, max_frames: Optional[int]) -> np.ndarray:
    """Select at most ``max_frames`` indices while retaining both endpoints."""

    if num_frames <= 0:
        raise ValueError("A clip must contain at least one frame.")
    if max_frames is None:
        return np.arange(num_frames, dtype=np.int64)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")
    if num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, max_frames).round().astype(np.int64)


def frames_to_bcthw(
    frames: FrameArray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert RGB ``[T,H,W,3]`` pixels in ``[0,255]`` to BCTHW in ``[-1,1]``."""

    if isinstance(frames, torch.Tensor):
        frames_array = frames.detach().cpu().numpy()
    else:
        frames_array = np.asarray(frames)
    if frames_array.ndim == 3:
        frames_array = frames_array[None, ...]
    if frames_array.ndim != 4 or frames_array.shape[-1] != 3:
        raise ValueError(
            "frames must have shape [T,H,W,3] (or [H,W,3] for one image); "
            f"got {tuple(frames_array.shape)}."
        )
    if any(size <= 0 for size in frames_array.shape[:3]):
        raise ValueError(
            "frames must have non-empty time, height, and width dimensions."
        )
    if not np.issubdtype(frames_array.dtype, np.number):
        raise TypeError("frames must contain numeric RGB pixel values.")
    if not np.isfinite(frames_array).all():
        raise ValueError("frames contain NaN or infinite pixel values.")
    if frames_array.min() < 0 or frames_array.max() > 255:
        raise ValueError("RGB pixel values must lie in [0, 255].")

    contiguous = np.ascontiguousarray(frames_array)
    video = torch.from_numpy(contiguous).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=torch.float32) / 127.5 - 1.0
    return video.to(dtype=dtype)


def resize_spatial(
    video: torch.Tensor, frame_size: Optional[Tuple[int, int]]
) -> torch.Tensor:
    """Resize a BCTHW tensor to ``(height, width)`` when requested."""

    if frame_size is None:
        return video
    height, width = frame_size
    if height <= 0 or width <= 0:
        raise ValueError("frame_size values must be positive.")
    if tuple(video.shape[-2:]) == (height, width):
        return video

    batch, channels, frames, old_height, old_width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, old_height, old_width
    )
    flat = F.interpolate(
        flat.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).to(dtype=video.dtype)
    return flat.reshape(batch, frames, channels, height, width).permute(
        0, 2, 1, 3, 4
    ).contiguous()


def pad_spatial(
    video: torch.Tensor, multiple: int
) -> Tuple[torch.Tensor, int, int]:
    """Replicate-pad only the bottom/right spatial edges to a model multiple."""

    if multiple <= 0:
        raise ValueError("pad_multiple must be greater than zero.")
    if video.ndim != 5:
        raise ValueError(f"video must be BCTHW; got shape {tuple(video.shape)}.")
    pad_h = (-video.shape[-2]) % multiple
    pad_w = (-video.shape[-1]) % multiple
    if pad_h == 0 and pad_w == 0:
        return video, 0, 0

    batch, channels, frames, height, width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    padded = flat.reshape(batch, frames, channels, height + pad_h, width + pad_w)
    return padded.permute(0, 2, 1, 3, 4).contiguous(), pad_h, pad_w


def crop_spatial(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Remove bottom/right padding from a BCTHW reconstruction."""

    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding values must be non-negative.")
    if pad_h:
        video = video[..., :-pad_h, :]
    if pad_w:
        video = video[..., :, :-pad_w]
    return video


def get_latent(encoder_output: Any) -> torch.Tensor:
    """Return the deterministic Cosmos continuous-tokenizer latent."""

    return encoder_output.latent_dist.mode()


def pool_decoder_input(
    decoder_input: torch.Tensor, temporal_pooling: str
) -> torch.Tensor:
    """Create the downstream representation from the decoder input."""

    if decoder_input.ndim != 5:
        raise ValueError(
            "The Cosmos decoder input must be BCDHW; "
            f"got {tuple(decoder_input.shape)}."
        )
    if temporal_pooling == "temporal_mean":
        return decoder_input.mean(dim=2)
    if temporal_pooling == "none":
        return decoder_input
    raise ValueError("temporal_pooling must be 'temporal_mean' or 'none'.")


@torch.inference_mode()
def reconstruct_and_extract(
    vae: Any, video: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility helper returning latent, decoder input, and reconstruction."""

    latent = get_latent(vae.encode(video))
    decoder_input = vae.post_quant_conv(latent)
    reconstruction = vae.decode(latent).sample
    return latent, decoder_input, reconstruction


def bcthw_to_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert a single reconstructed BCTHW batch to uint8 THWC frames."""

    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(
            "video must have shape [1,3,T,H,W] to write one RGB video; "
            f"got {tuple(video.shape)}."
        )
    pixels = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    pixels = ((pixels + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return pixels.permute(1, 2, 3, 0).numpy()


def save_mp4(frames: np.ndarray, path: PathLike, fps: float) -> None:
    """Write uint8 THWC frames as a visual-only H.264 MP4."""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero.")
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(
            "frames must be a non-empty uint8 array with shape [T,H,W,3]."
        )
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError(
            "H.264 reconstruction requires even frame dimensions; "
            f"got ({height}, {width}). Use an even frame_size."
        )
    imageio = _import_imageio()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


MAX_OUTPUT_NAME_LEN = 200


def output_name(video_path: PathLike) -> str:
    """Create a readable, collision-resistant directory name for one video."""

    path = Path(video_path)
    readable = "_".join(path.with_suffix("").parts[-3:]).replace(" ", "_")
    readable = readable or "video"
    digest = hashlib.sha256(
        str(path.expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:10]
    max_readable = MAX_OUTPUT_NAME_LEN - len(digest) - 1
    return f"{readable[:max_readable]}-{digest}"


class CosmosVideoExtractor:
    """Encode, reconstruct, and extract representations from RGB clips.

    Parameters
    ----------
    model_id:
        Hugging Face model ID containing a Diffusers Cosmos VAE.
    device, dtype:
        Model execution device and precision. ``auto`` uses BF16 only on CUDA
        devices that report BF16 support and otherwise uses FP32.
    frame_size:
        Optional ``(height, width)`` model-input size. The paper reports a
        pooled ``(16,40,60)`` representation, which is consistent with a
        ``(320,480)`` input under 8x8 spatial compression, but it does not
        specify its resize policy. Consequently no resize is applied by
        default.
    max_frames:
        Uniformly sample longer segments down to this many frames. The default
        is the 121-frame context listed for the checkpoint. Set ``None`` to
        disable sampling.
    temporal_pooling:
        ``temporal_mean`` implements the paper's representation. ``none``
        retains the full decoder-input tensor.
    vae:
        Optional injected VAE, primarily useful for offline tests or custom
        compatible models. When omitted, the checkpoint is downloaded/loaded.
    pretrained_kwargs:
        Additional Hugging Face loader options such as ``revision``, ``token``,
        or ``local_files_only``. The requested revision and resolved model
        commit (when available) are recorded in saved payloads.

    Notes
    -----
    The continuous Cosmos tokenizer is a deterministic autoencoder: Diffusers
    exposes its encoder output through an ``IdentityDistribution``, whose
    ``sample`` and ``mode`` are identical. This class therefore always uses the
    mode and does not expose a misleading sampling option.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_COSMOS_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[PathLike] = None,
        dtype: Union[str, torch.dtype] = "auto",
        *,
        frame_size: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
        pad_multiple: Optional[int] = None,
        temporal_pooling: str = "temporal_mean",
        vae: Optional[Any] = None,
        pretrained_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        self.dtype = model_dtype(dtype, self.device)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.frame_size = self._validate_frame_size(frame_size)
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive or None.")
        self.max_frames = max_frames
        if temporal_pooling not in {"temporal_mean", "none"}:
            raise ValueError("temporal_pooling must be 'temporal_mean' or 'none'.")
        self.temporal_pooling = temporal_pooling
        loader_options = dict(pretrained_kwargs or {})
        self.requested_revision = loader_options.get("revision")

        if vae is None:
            vae = self._load_vae(loader_options)
        moved = vae.to(device=self.device, dtype=self.dtype)
        self.vae = moved if moved is not None else vae
        evaluated = self.vae.eval()
        if evaluated is not None:
            self.vae = evaluated

        if pad_multiple is None:
            config = getattr(self.vae, "config", None)
            pad_multiple = int(getattr(config, "spatial_compression_ratio", 8))
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be greater than zero.")
        self.pad_multiple = int(pad_multiple)
        config = getattr(self.vae, "config", None)
        self.model_commit_hash = getattr(config, "_commit_hash", None)
        self.vae_config = {
            name: getattr(config, name, None)
            for name in (
                "latent_channels",
                "spatial_compression_ratio",
                "temporal_compression_ratio",
                "patch_size",
            )
        }
        self.library_versions = {
            "gpi_pack": _installed_version("gpi_pack"),
            "diffusers": _installed_version("diffusers"),
            "torch": str(torch.__version__),
        }

    @staticmethod
    def _validate_frame_size(
        frame_size: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        if frame_size is None:
            return None
        if len(frame_size) != 2:
            raise ValueError("frame_size must be a (height, width) pair.")
        height, width = int(frame_size[0]), int(frame_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame_size values must be positive.")
        return height, width

    def _load_vae(self, pretrained_kwargs: Mapping[str, Any]) -> Any:
        try:
            from diffusers import AutoencoderKLCosmos
        except ImportError as exc:
            raise ImportError(
                "CosmosVideoExtractor requires diffusers>=0.34.0. Install "
                "gpi_pack with its image dependencies."
            ) from exc

        kwargs: Dict[str, Any] = dict(pretrained_kwargs)
        kwargs.setdefault("subfolder", "vae")
        kwargs.setdefault("torch_dtype", self.dtype)
        if self.cache_dir is not None:
            kwargs.setdefault("cache_dir", str(self.cache_dir))
        return AutoencoderKLCosmos.from_pretrained(self.model_id, **kwargs)

    def _preprocess_with_metadata(
        self, frames: FrameArray
    ) -> Tuple[torch.Tensor, int, Tuple[int, ...]]:
        if not isinstance(frames, (np.ndarray, torch.Tensor)):
            frames = np.asarray(frames)
        frame_ndim = int(frames.ndim)
        num_frames = int(frames.shape[0]) if frame_ndim == 4 else 1
        indices = uniform_frame_indices(num_frames, self.max_frames)
        if frame_ndim == 4:
            if isinstance(frames, torch.Tensor):
                tensor_indices = torch.as_tensor(indices, device=frames.device)
                frames = frames.index_select(0, tensor_indices)
            else:
                frames = frames[indices]
        video = frames_to_bcthw(frames, self.device, self.dtype)
        video = resize_spatial(video, self.frame_size)
        return video, num_frames, tuple(int(index) for index in indices)

    def preprocess_video(self, frames: FrameArray) -> torch.Tensor:
        """Convert an RGB image/clip to the normalized model-input tensor."""

        video, _, _ = self._preprocess_with_metadata(frames)
        return video

    @torch.inference_mode()
    def encode_video(
        self,
        frames: FrameArray,
    ) -> VideoExtractionResult:
        """Encode one RGB image/clip without running the decoder."""

        video, original_num_frames, selected_indices = self._preprocess_with_metadata(
            frames
        )
        padded, pad_h, pad_w = pad_spatial(video, self.pad_multiple)
        latent = get_latent(self.vae.encode(padded))
        decoder_input = self.vae.post_quant_conv(latent)
        representation = pool_decoder_input(decoder_input, self.temporal_pooling)
        return VideoExtractionResult(
            latent=latent,
            decoder_input=decoder_input,
            representation=representation,
            reconstruction=None,
            input_shape_bcthw=tuple(video.shape),
            padded_shape_bcthw=tuple(padded.shape),
            pad_bottom=pad_h,
            pad_right=pad_w,
            original_num_frames=original_num_frames,
            selected_frame_indices=selected_indices,
            temporal_pooling=self.temporal_pooling,
        )

    @torch.inference_mode()
    def decode_latents(
        self,
        latent: torch.Tensor,
        *,
        pad_bottom: int = 0,
        pad_right: int = 0,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode latents and remove preprocessing padding."""

        output = self.vae.decode(latent)
        reconstruction = output.sample if hasattr(output, "sample") else output[0]
        reconstruction = crop_spatial(reconstruction, pad_bottom, pad_right)
        if num_frames is not None:
            if num_frames <= 0:
                raise ValueError("num_frames must be positive when provided.")
            reconstruction = reconstruction[:, :, :num_frames]
        return reconstruction

    def reconstruct_video(
        self,
        frames: FrameArray,
    ) -> VideoExtractionResult:
        """Encode and reconstruct one RGB image/clip."""

        result = self.encode_video(frames)
        reconstruction = self.decode_latents(
            result.latent,
            pad_bottom=result.pad_bottom,
            pad_right=result.pad_right,
            num_frames=result.input_shape_bcthw[2],
        )
        return replace(result, reconstruction=reconstruction)

    def transform_video(
        self,
        frames: FrameArray,
    ) -> VideoExtractionResult:
        """Alias for ``reconstruct_video``, matching the image extractor API."""

        return self.reconstruct_video(frames)

    def save_video(self, video: torch.Tensor, path: PathLike, fps: float) -> None:
        """Save one reconstructed BCTHW tensor as an MP4."""

        save_mp4(bcthw_to_uint8(video), path, fps)

    def process_video(
        self,
        video_path: PathLike,
        output_hidden_dir: PathLike,
        *,
        output_video_dir: Optional[PathLike] = None,
        segment_seconds: float,
        segment: Optional[Union[int, SegmentSpec]] = None,
        drop_last: bool = False,
        save_latent: bool = False,
        save_decoder_input: bool = False,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[VideoSegmentOutput]:
        """Process and serialize every selected segment in one video."""

        if not math.isfinite(segment_seconds) or segment_seconds <= 0:
            raise ValueError("segment_seconds must be finite and greater than zero.")
        source = Path(video_path).expanduser()
        segment_spec = _normalize_segment_spec(segment)
        hidden_dir = Path(output_hidden_dir).expanduser() / output_name(source)
        hidden_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_dir = None
        if output_video_dir is not None:
            reconstruction_dir = (
                Path(output_video_dir).expanduser() / output_name(source)
            )
            reconstruction_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Processing {source}")

        outputs: List[VideoSegmentOutput] = []
        for frames, meta in iter_segments(source, segment_seconds, drop_last):
            segment_idx = int(meta["segment_index"])
            if segment_spec is not None:
                start, end = segment_spec
                if segment_idx < start:
                    continue
                if end is not None and segment_idx > end:
                    break

            tag = f"segment_{segment_idx:06d}"
            representation_path = hidden_dir / f"{tag}.pt"
            if representation_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {representation_path}. "
                    "Pass overwrite=True to replace it."
                )
            reconstruction_path = None
            if reconstruction_dir is not None:
                reconstruction_path = reconstruction_dir / f"{tag}_recon.mp4"
                if reconstruction_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Output already exists: {reconstruction_path}. "
                        "Pass overwrite=True to replace it."
                    )

            result = (
                self.reconstruct_video(frames)
                if reconstruction_dir is not None
                else self.encode_video(frames)
            )
            if result.representation.shape[0] != 1:
                raise ValueError("File processing expects one clip per model batch.")

            selected_count = len(result.selected_frame_indices)
            source_count = result.original_num_frames
            source_fps = float(meta["fps"])
            processed_fps = source_fps * selected_count / source_count
            representation = result.representation[0].detach().float().cpu()
            payload: Dict[str, Any] = {
                "schema_version": VIDEO_PAYLOAD_SCHEMA_VERSION,
                "video_path": str(source),
                "model_id": self.model_id,
                "requested_revision": self.requested_revision,
                "model_commit_hash": self.model_commit_hash,
                "vae_config": self.vae_config,
                "library_versions": self.library_versions,
                "model_dtype": str(self.dtype).replace("torch.", "", 1),
                "device_type": self.device.type,
                "segment_seconds": float(segment_seconds),
                "latent_selection": "mode",
                "temporal_pooling": result.temporal_pooling,
                "configured_frame_size": self.frame_size,
                "resize_mode": "bilinear",
                "resize_antialias": True,
                "max_frames": self.max_frames,
                "representation": representation,
                "representation_shape": tuple(representation.shape),
                "representation_dtype": "float32",
                "input_shape_bcthw": result.input_shape_bcthw,
                "padded_shape_bcthw": result.padded_shape_bcthw,
                "padding": {
                    "pad_bottom": result.pad_bottom,
                    "pad_right": result.pad_right,
                },
                "source_segment_frame_count": source_count,
                "source_shape_thwc": tuple(frames.shape),
                "model_frame_count": selected_count,
                "selected_frame_indices": result.selected_frame_indices,
                "frame_sampling": (
                    "uniform" if selected_count < source_count else "none"
                ),
                "processed_fps": processed_fps,
                "audio_reconstructed": False,
                **meta,
            }
            if save_latent:
                payload["latent"] = result.latent[0].detach().float().cpu()
            if save_decoder_input:
                payload["decoder_input"] = (
                    result.decoder_input[0].detach().float().cpu()
                )
                payload["pre_decoder_hidden"] = payload["decoder_input"]
            torch.save(payload, representation_path)

            if reconstruction_dir is not None:
                if result.reconstruction is None:
                    raise RuntimeError(
                        "Reconstruction was requested but was not decoded."
                    )
                assert reconstruction_path is not None
                self.save_video(
                    result.reconstruction, reconstruction_path, processed_fps
                )

            outputs.append(
                VideoSegmentOutput(
                    video_path=source,
                    segment_index=segment_idx,
                    representation_path=representation_path,
                    reconstruction_path=reconstruction_path,
                )
            )
            if verbose:
                print(
                    f"  saved {representation_path.name} | frames "
                    f"{meta['start_frame']}-{meta['end_frame']} | "
                    f"representation {tuple(representation.shape)}"
                )
        return outputs


def _expand_video_inputs(
    videos: Union[PathLike, Sequence[PathLike]], *, exclude: Sequence[PathLike]
) -> List[Path]:
    inputs: Iterable[PathLike]
    if isinstance(videos, (str, Path)):
        inputs = [videos]
    else:
        inputs = videos

    discovered: List[Path] = []
    for item in inputs:
        input_path = Path(item).expanduser()
        if input_path.is_dir() and any(
            input_path.resolve() == Path(root).expanduser().resolve()
            for root in exclude
        ):
            raise ValueError(
                "Input and output directories must be different to avoid "
                "reprocessing generated files."
            )
        effective_exclude = [
            root
            for root in exclude
            if not _is_within(input_path, Path(root).expanduser())
        ]
        discovered.extend(find_videos(item, exclude=effective_exclude))
    unique: Dict[str, Path] = {}
    for path in discovered:
        unique[str(path.expanduser().resolve())] = path
    return sorted(unique.values())


def extract_videos(
    videos: Union[PathLike, Sequence[PathLike]],
    output_hidden_dir: PathLike,
    *,
    segment_seconds: float,
    output_video_dir: Optional[PathLike] = None,
    segment: Optional[Union[int, SegmentSpec]] = None,
    model_id: str = DEFAULT_COSMOS_MODEL_ID,
    device: Optional[Union[str, torch.device]] = None,
    cache_dir: Optional[PathLike] = None,
    dtype: Union[str, torch.dtype] = "auto",
    frame_size: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
    pad_multiple: Optional[int] = None,
    temporal_pooling: str = "temporal_mean",
    drop_last: bool = False,
    save_latent: bool = False,
    save_decoder_input: bool = False,
    overwrite: bool = False,
    extractor: Optional[CosmosVideoExtractor] = None,
    pretrained_kwargs: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
) -> List[VideoSegmentOutput]:
    """Extract pooled Cosmos representations from one or more videos.

    A separate ``segment_*.pt`` payload is written for each clip. By default it
    contains the paper-ready temporal-mean representation as an unbatched
    ``[C,H,W]`` float32 tensor. Supplying ``output_video_dir`` also decodes and
    saves visual-only reconstructions; extraction-only runs skip the decoder.

    ``extractor`` allows callers to reuse one loaded model across calls or inject
    a compatible model for offline testing.
    """

    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("segment_seconds must be finite and greater than zero.")
    hidden_root = Path(output_hidden_dir).expanduser()
    excluded: List[PathLike] = [hidden_root]
    if output_video_dir is not None:
        excluded.append(output_video_dir)
    video_paths = _expand_video_inputs(videos, exclude=excluded)
    if not video_paths:
        raise ValueError("No supported video files were found in the supplied input.")

    hidden_root.mkdir(parents=True, exist_ok=True)
    if output_video_dir is not None:
        Path(output_video_dir).expanduser().mkdir(parents=True, exist_ok=True)

    if extractor is None:
        extractor = CosmosVideoExtractor(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            frame_size=frame_size,
            max_frames=max_frames,
            pad_multiple=pad_multiple,
            temporal_pooling=temporal_pooling,
            pretrained_kwargs=pretrained_kwargs,
        )

    outputs: List[VideoSegmentOutput] = []
    for path in video_paths:
        outputs.extend(
            extractor.process_video(
                path,
                hidden_root,
                output_video_dir=output_video_dir,
                segment_seconds=segment_seconds,
                segment=segment,
                drop_last=drop_last,
                save_latent=save_latent,
                save_decoder_input=save_decoder_input,
                overwrite=overwrite,
                verbose=verbose,
            )
        )
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for ``python -m gpi_pack.video``."""

    parser = argparse.ArgumentParser(
        description="Extract Cosmos video representations and optional reconstructions."
    )
    parser.add_argument(
        "--input", required=True, help="Video file or directory of videos."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for saved .pt files."
    )
    parser.add_argument("--segment_seconds", type=float, required=True)
    parser.add_argument(
        "--segment",
        type=parse_segment_spec,
        default=None,
        help="Optional segment index/range: 0, 0-5, or 5-.",
    )
    parser.add_argument("--model_id", default=DEFAULT_COSMOS_MODEL_ID)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.add_argument(
        "--temporal_pooling",
        choices=["temporal_mean", "none"],
        default="temporal_mean",
    )
    parser.add_argument("--pad_multiple", type=int, default=None)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Uniformly sample longer clips to this many frames; use 0 to disable.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
    )
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--save_latent", action="store_true")
    parser.add_argument("--save_decoder_input", action="store_true")
    parser.add_argument("--save_reconstruction", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line adapter around :func:`extract_videos`."""

    args = parse_args(argv)
    max_frames = None if args.max_frames == 0 else args.max_frames
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative.")
    frame_size = tuple(args.frame_size) if args.frame_size is not None else None
    output_video_dir = args.output_dir if args.save_reconstruction else None

    if args.segment is not None:
        print(f"Segment filter: {format_segment_spec(args.segment)}")
    outputs = extract_videos(
        videos=args.input,
        output_hidden_dir=args.output_dir,
        output_video_dir=output_video_dir,
        segment_seconds=args.segment_seconds,
        segment=args.segment,
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        frame_size=frame_size,
        max_frames=max_frames,
        pad_multiple=args.pad_multiple,
        temporal_pooling=args.temporal_pooling,
        drop_last=args.drop_last,
        save_latent=args.save_latent,
        save_decoder_input=args.save_decoder_input,
        overwrite=args.overwrite,
    )
    print(f"Done. Saved {len(outputs)} segment representation(s).")


if __name__ == "__main__":
    main()
