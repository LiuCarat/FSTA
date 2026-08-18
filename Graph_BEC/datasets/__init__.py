"""Dataset-specific loaders for Graph_BEC."""

from Graph_BEC.datasets.abide import (
    ABIDERecord,
    load_abide_records,
    load_time_series,
)

__all__ = ["ABIDERecord", "load_abide_records", "load_time_series"]

