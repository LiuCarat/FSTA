from importlib import import_module

__all__ = ["PGRECStatic", "QSRECRefiner", "normative_reference"]


def __getattr__(name):
    modules = {
        "PGRECStatic": ".ablation.mpr_refinement",
        "QSRECRefiner": ".qsr",
        "normative_reference": ".mpr.population_reference",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
