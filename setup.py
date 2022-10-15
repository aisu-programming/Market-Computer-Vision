import subprocess

try:
    import torch
except ModuleNotFoundError:
    subprocess.call("setup-torch.sh")

try:
    import torchvision
except ModuleNotFoundError:
    subprocess.call("setup-torchvision.sh")

subprocess.call("setup-others.sh")
subprocess.call("download-weights.sh")