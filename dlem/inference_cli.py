from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from dlem.dataset_dlem import return_bigwig_regions
from dlem.head import ForkedBasePairTrackHeadBinaryUnet
from dlem.seq_pooler import SequenceFeaturePoolerSimple
from dlem.trainer import LitTrainer
from dlem import util
from dlem.util import convert_diags_to_full_contact

DEFAULT_CHROM = "chr1"
DEFAULT_START = 2_805_000
DEFAULT_RESOLUTION = 10_000
DEFAULT_LAYER_CHANNEL_NUM = 3
DEFAULT_BIN_SIZE = 10_000
DEFAULT_DATA_PATCH_SIZE = 128
DEFAULT_DATA_TRACK_DIM = 1
DEFAULT_SEQ_DIM = 2
DEFAULT_DATA_START = 0
DEFAULT_DATA_STOP = 128
DEFAULT_CHANNEL_PER_ROUTE = 4
DEFAULT_HEAD_LAYER_NUM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DLEM contact map inference and save the full contact map as a .npy file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to store the output .npy file.",
    )
    parser.add_argument(
        "chrom",
        help=f"Chromosome to run inference on (e.g. {DEFAULT_CHROM}).",
    )
    parser.add_argument(
        "start",
        type=int,
        help=f"Start coordinate (inclusive) (e.g. {DEFAULT_START}).",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        required=True,
        metavar="TRACK_BIGWIG",
        help="Epigenetic track bigWig files (one or more).",
    )
    parser.add_argument(
        "--seq-features",
        nargs="+",
        required=True,
        metavar="SEQ_BIGWIG",
        help="Sequence feature bigWig files (one or more).",
    )
    parser.add_argument(
        "--model-checkpoint",
        required=True,
        help="Path to the Lightning checkpoint to load for inference.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Contact map resolution in base pairs. Default: {DEFAULT_RESOLUTION}",
    )
    parser.add_argument(
        "--layer-channel-num",
        type=int,
        default=DEFAULT_LAYER_CHANNEL_NUM,
        help=f"Number of channels for the sequence pooler. Default: {DEFAULT_LAYER_CHANNEL_NUM}",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=DEFAULT_BIN_SIZE,
        help=f"Bin size passed to the sequence pooler. Default: {DEFAULT_BIN_SIZE}",
    )
    parser.add_argument(
        "--data-patch-size",
        type=int,
        default=DEFAULT_DATA_PATCH_SIZE,
        help=f"Patch size used by the model. Default: {DEFAULT_DATA_PATCH_SIZE}",
    )
    parser.add_argument(
        "--data-track-dim",
        type=int,
        default=DEFAULT_DATA_TRACK_DIM,
        help=f"Number of track channels. Default: {DEFAULT_DATA_TRACK_DIM}",
    )
    parser.add_argument(
        "--seq-dim",
        type=int,
        default=DEFAULT_SEQ_DIM,
        help=f"Number of sequence feature channels. Default: {DEFAULT_SEQ_DIM}",
    )
    parser.add_argument(
        "--data-start",
        type=int,
        default=DEFAULT_DATA_START,
        help=f"Start diagonal index used by the model. Default: {DEFAULT_DATA_START}",
    )
    parser.add_argument(
        "--data-stop",
        type=int,
        default=DEFAULT_DATA_STOP,
        help=f"Stop diagonal index used by the model. Default: {DEFAULT_DATA_STOP}",
    )
    parser.add_argument(
        "--channel-per-route",
        type=int,
        default=DEFAULT_CHANNEL_PER_ROUTE,
        help=f"Number of channels per route in the head. Default: {DEFAULT_CHANNEL_PER_ROUTE}",
    )
    parser.add_argument(
        "--head-layer-num",
        type=int,
        default=DEFAULT_HEAD_LAYER_NUM,
        help=f"Number of layers in the head. Default: {DEFAULT_HEAD_LAYER_NUM}",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use (e.g. 'cpu', 'cuda', or 'cuda:0'). Default: cpu",
    )
    parser.add_argument(
        "--bigwig-pooling",
        type=int,
        default=None,
        help="Optional pooling resolution passed to bigWig reader. Default: no pooling.",
    )
    parser.add_argument(
        "--seq-chrom",
        default=None,
        help="Chromosome used for sequence features. Defaults to the value of --chrom.",
    )
    parser.add_argument(
        "--seq-start",
        type=int,
        default=None,
        help="Start coordinate for sequence features. Defaults to --start.",
    )
    parser.add_argument(
        "--seq-stop",
        type=int,
        default=None,
        help="Stop coordinate for sequence features. Defaults to the computed stop value.",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str.lower() == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(target)
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    return device


def _load_bigwig(
    paths: Sequence[str],
    chrom: str,
    start: int,
    stop: int,
    pooling: Optional[int],
    normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> torch.Tensor:
    data = return_bigwig_regions(paths, chrom, start, stop, pooling)
    if normalization is not None:
        data = normalization(data)
    return torch.from_numpy(data).unsqueeze(0).to(torch.float32)


def _prepare_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    seq_pooler = SequenceFeaturePoolerSimple(
        [args.layer_channel_num],
        [args.bin_size],
    )
    head = ForkedBasePairTrackHeadBinaryUnet(
        args.data_patch_size,
        args.data_track_dim,
        args.seq_dim,
        args.data_start,
        args.data_stop,
        util.dlem,
        seq_pooler,
        channel_per_route=args.channel_per_route,
        layer_num=args.head_layer_num,
    )
    lit_module = LitTrainer.load_from_checkpoint(
        args.model_checkpoint,
        model=head,
        loss=torch.nn.MSELoss(),
        map_location=device,
    )
    return lit_module.model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    stop = args.start + args.data_patch_size * args.resolution
    seq_chrom = args.seq_chrom or args.chrom
    seq_start = args.seq_start if args.seq_start is not None else args.start
    seq_stop = args.seq_stop if args.seq_stop is not None else stop

    model = _prepare_model(args, device)

    tracks = _load_bigwig(args.tracks, args.chrom, args.start, stop, args.bigwig_pooling).to(device)
    seq_features = _load_bigwig(args.seq_features, seq_chrom, seq_start, seq_stop, args.bigwig_pooling).to(device)

    init_mass = torch.full(
        (1, args.data_patch_size),
        fill_value=args.data_patch_size,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        diagonals = model.contact_map_prediction(tracks, seq_features, init_mass).cpu().squeeze(0).numpy()

    contact_map = convert_diags_to_full_contact(diagonals, args.data_start, args.data_stop)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.chrom}_{args.start}_{stop}.npy"
    np.save(output_path, contact_map)
    print(f"Saved contact map to {output_path}")


if __name__ == "__main__":
    main()

