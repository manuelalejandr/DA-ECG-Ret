"""
Central registry mapping method name -> module exposing:
    build_modules(cfg, device) -> dict[str, nn.Module] (must contain
        "embedding_model", an nn.Module with forward(x) -> (B, d) usable
        directly for retrieval / linear-probe evaluation)
    compute_loss(batch, modules, cfg, device) -> (loss: Tensor, logs: dict)

This lets train.py / evaluate.py stay identical regardless of how many
SSL methods are registered.
"""
from ssl_methods import simclr, clocs_lite, ts2vec, tfc, softclt, da_ecg_ret, supervised_ref

METHOD_REGISTRY = {
    "simclr": simclr,
    "clocs_lite": clocs_lite,
    "ts2vec": ts2vec,
    "tfc": tfc,
    "softclt": softclt,
    "da_ecg_ret": da_ecg_ret,
    # supervised REFERENCE encoders (benchmark upper bound; not in SSL_METHODS)
    "supervised_fine": supervised_ref.supervised_fine,
    "supervised_super": supervised_ref.supervised_super,
}


def get_method(name: str):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown SSL method '{name}'. Available: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name]
