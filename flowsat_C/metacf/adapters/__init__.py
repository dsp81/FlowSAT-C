from metacf.adapters.base import ModelAdapter  # noqa: F401


def make_adapter(cfg: dict, **kw):
    kind = cfg["model"]["adapter"]
    if kind == "dummy":
        from metacf.adapters.dummy import DummyAdapter
        return DummyAdapter(**cfg["model"].get("dummy", {}), **kw)
    if kind == "flowsat_c":
        from metacf.adapters.flowsat_c import FlowSatCAdapter
        return FlowSatCAdapter(cfg["model"], **kw)
    raise ValueError(kind)
