import torch

def collate_fn_self_supervised(data):
    images = [d for d in data if d is not None]
    return torch.cat(images).float()

def collate_fn_mae(data):
    images = [d[0] for d in data if d is not None]    
    binarized = [d[1] for d in data if d is not None]

    return torch.cat(images).float(), torch.cat(binarized).float()


def collate_fn_evaluate(data):
    images = [d[0] for d in data]
    labels = [torch.tensor(d[1]) for d in data]
    return torch.cat(images).float(), torch.cat(labels).long()
