import torch
import torch.nn.functional as F


"""
Initialize the device to run PyTorch on. If CUDA is avaliable, set it to the CUDA device, else - CPU.
"""
def initialize_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Current device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"Current device: CPU")
    return device

DEVICE = initialize_device()

"""
Converts the model outputs into a binary mask, given a threshold between 0 and 1.
"""
def binarize(x, threshold=0.5):
    x = torch.where(x > threshold, 1.0, 0.0).type(torch.IntTensor)
    return x

"""
Scales model outputs to resemble probabilities (sum to 1) via softmax scaling.
"""
def probscale(x):
    return F.softmax(x, dim=1)

"""
Computes the percentile limits, given specific class qs.
"""
def get_percentiles(x, q, n_classes = 10):
    x_max, x_argmax = torch.max(x.flatten(start_dim=2), dim=1)
    out = torch.zeros(x.shape[0], x.shape[1]).to(DEVICE)
    for class_id in range(n_classes):
        percentile_input = x_max[x_argmax == class_id]
        percentile_input = percentile_input[percentile_input > 0]
        if len(percentile_input) == 0:
            percentile_input = torch.tensor([1.0]).to(DEVICE)
        out[:, class_id] = torch.quantile(percentile_input, q=q[class_id], dim=0, interpolation='linear')
    return out







