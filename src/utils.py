# reserved for shared training utilities

import torch

def print_gpu_info():
    if not torch.cuda.is_available():
        print("no cuda gpu found")
        return
    device = torch.device("cuda")
    print("device:", device)
    print("name:", torch.cuda.get_device_name(device))
    print("memory total (gb):", torch.cuda.get_device_properties(device).total_memory / 1e9)

if __name__ == "__main__":
    print_gpu_info()
