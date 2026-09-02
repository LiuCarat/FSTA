"""Command-line arguments belonging to the FSTA-EC baseline."""

from __future__ import annotations


def add_fsta_arguments(parser):
    group = parser.add_argument_group("FSTA-EC baseline")
    group.add_argument("--window-length", type=int, default=80)
    group.add_argument("--stride", type=int, default=40)
    group.add_argument("--epochs", type=int, default=51)
    group.add_argument("--alpha-sp", type=float, default=0.8)
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--log-every", type=int, default=10)
    group.add_argument("--d-model", type=int, default=16)
    group.add_argument("--d-inner-hid", type=int, default=64)
    group.add_argument("--d-k", type=int, default=8)
    group.add_argument("--d-v", type=int, default=8)
    group.add_argument("--n-head", type=int, default=2)
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--n-warmup-steps", type=int, default=4000)
    group.add_argument("--lr-mul", type=float, default=1.2)
    group.add_argument("--weight-decay", type=float, default=0.0)
    group.add_argument("--adam-beta1", type=float, default=0.9)
    group.add_argument("--adam-beta2", type=float, default=0.98)
    group.add_argument("--num-hidden-layers", type=int, default=1)
    group.add_argument("--num-attention-heads", type=int, default=2)
    group.add_argument("--hidden-act", default="gelu")
    group.add_argument("--attention-probs-dropout-prob", type=float, default=0.5)
    group.add_argument("--hidden-dropout-prob", type=float, default=0.5)
    group.add_argument("--initializer-range", type=float, default=0.02)
    group.add_argument("--no-filters", action="store_true")
