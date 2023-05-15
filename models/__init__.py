from .simsiam import SimSiam, SimSiam_ROMA
from .simclr import SimCLR, SimCLR_ROMA
from .trip_roma import Trip, Trip_ROMA
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr_roma':
        model = SimCLR_ROMA(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'simsiam_roma':
        model = SimSiam_ROMA(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'trip':
        model = Trip(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'trip_roma':
        model = Trip_ROMA(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    else:
        raise NotImplementedError
    return model