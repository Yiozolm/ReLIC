from .ReLIC import ReLIC_ELIC, ReLIC_mbt, Unet


models = {}

models.update({
    "elic": ReLIC_ELIC,
    "mbt": ReLIC_mbt,
})


__all__ = [
    "models",
    "Unet",
]