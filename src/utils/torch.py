def torch_is_available():
    import torch
    return torch.cuda.is_available()

def get_torch_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
