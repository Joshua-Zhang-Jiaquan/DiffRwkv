import torch


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_optimizer_type(opt_cfg) -> str:
    opt_type = _cfg_get(opt_cfg, "type", None)
    if opt_type is not None:
        return str(opt_type).lower()

    # Back-compat with older configs using `name: adamw`
    name = _cfg_get(opt_cfg, "name", None)
    if name is None:
        raise ValueError("Missing optimizer.type (or optimizer.name) in config")
    name = str(name).lower()
    if name in {"adam", "adamw"}:
        return "adam"
    if name in {"psgd", "psgdkron"}:
        return "psgd"
    return name


def get_optimizer(config, trainer):
    opt_cfg = _cfg_get(config, "optimizer", {})
    opt_type = _resolve_optimizer_type(opt_cfg)

    params = trainer.parameters()
    if opt_type == "adam":
        beta1 = float(_cfg_get(opt_cfg, "beta1", 0.9))
        beta2 = float(_cfg_get(opt_cfg, "beta2", 0.999))
        betas_cfg = _cfg_get(opt_cfg, "betas", None)
        if betas_cfg is None:
            betas = (beta1, beta2)
        else:
            betas_seq = list(betas_cfg)
            if len(betas_seq) != 2:
                raise ValueError(f"optimizer.betas must have length 2, got {len(betas_seq)}")
            betas = (float(betas_seq[0]), float(betas_seq[1]))
        weight_decay = float(_cfg_get(opt_cfg, "weight_decay", 0.0))
        eps = float(_cfg_get(opt_cfg, "eps", 1e-8))
        lr = float(_cfg_get(opt_cfg, "lr", 1e-4))
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        return optimizer

    if opt_type == "psgd":
        try:
            import heavyball  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Optimizer type 'psgd' requires the 'heavyball' package. "
                "Install it (see requirements.txt) or set optimizer.type/name to 'adamw'."
            ) from exc

        heavyball.utils.set_torch()
        beta = float(_cfg_get(opt_cfg, "beta", 0.9))
        weight_decay = float(_cfg_get(opt_cfg, "weight_decay", 0.0))
        mars = bool(_cfg_get(opt_cfg, "mars", False))
        caution = bool(_cfg_get(opt_cfg, "caution", False))
        optimizer = heavyball.ForeachPSGDKron(
            params,
            beta=beta,
            weight_decay=weight_decay,
            mars=mars,
            caution=caution,
        )
        optimizer.promote = True
        return optimizer

    raise ValueError(f"Unknown optimizer type: {opt_type}")
