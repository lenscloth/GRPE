from re import L
import torch

from .fingerprint import MoleculeFingerPrint


def load_pretrained_fingerprint(cuda=False):
    link = "http://192.168.2.130:8000/gsa/pcqm4mv2_pretrained_standard.pt"
    model_state_dict = torch.hub.load_state_dict_from_url(link)

    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model = MoleculeFingerPrint(
        d_model=768,
        dim_feedforward=768,
        nhead=32,
        num_layer=12,
        max_hop=5,
        num_node_type=512 * 9 + 1,
        num_edge_type=30,
    )
    model.load_state_dict(new_state_dict)
    if cuda:
        model = model.cuda()

    return model
