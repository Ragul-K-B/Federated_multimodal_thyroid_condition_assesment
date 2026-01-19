import torch

# -------- ONLY CLASSIFIER PARAMETERS --------
def get_parameters(model):
    """
    Return ONLY classifier parameters for federated learning
    """
    return [
        val.detach().cpu().numpy()
        for val in model.classifier.state_dict().values()
    ]


def set_parameters(model, parameters):
    """
    Load ONLY classifier parameters
    """
    keys = model.classifier.state_dict().keys()
    state_dict = {
        k: torch.tensor(v)
        for k, v in zip(keys, parameters)
    }
    model.classifier.load_state_dict(state_dict, strict=True)
