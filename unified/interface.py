import torch

def run_inference(system, x, modality):
    system_model = system.predict(x, modality)
    probs = torch.softmax(system_model, dim=1)
    pred = torch.argmax(probs, dim=1)
    return pred.item(), probs
