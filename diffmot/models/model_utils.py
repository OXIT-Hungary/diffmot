import torch
import torch.nn as nn

import diffmot
from .condition_embedding import History_motion_embedding
from .autoencoder import D2MP


def get_diffmot(cfg) -> nn.Module:

    checkpoint = torch.load(cfg.model.checkpoint)

    encoder = History_motion_embedding()
    model = D2MP(cfg.model, encoder=encoder).to(cfg.model.device)

    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["ddpm"].items()})

    tracker = diffmot.tracker.diffmottracker(config=cfg.tracker, model=model, frame_rate=30)

    return model, tracker
