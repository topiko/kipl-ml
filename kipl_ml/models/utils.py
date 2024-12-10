def count_parameters(model) -> str:
    np = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return f"{np:.02f} M"
