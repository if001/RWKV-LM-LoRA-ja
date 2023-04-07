import os
import sys
from typing import Dict
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} [--use-gpu] <lora_alpha> <base_model.pth> <lora_checkpoint.pth> <output.pth>')

if sys.argv[1] == '--use-gpu':
    device='cuda'
    lora_alpha, base_model, lora, output = sys.argv[2:]
else:
    device='cpu'
    lora_alpha, base_model, lora, output = sys.argv[1:]


w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
# merge LoRA-only slim checkpoint into the main weights
w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
for k in w_lora.keys():
    w[k] = w_lora[k]
# merge LoRA weights
keys = set(w.keys())
for k in keys:
    if k.endswith('.weight'):
        prefix = k[:-len('.weight')]
        lora_A = prefix + '.lora_A'
        lora_B = prefix + '.lora_B'
        if lora_A in keys:
            assert lora_B in keys
            print(f'merging {lora_A} and {lora_B} into {k}')
            assert w[lora_B].shape[1] == w[lora_A].shape[0]
            lora_r = w[lora_B].shape[1]
            w[k] = w[k].to(device=device)
            w[lora_A] = w[lora_A].to(device=device)
            w[lora_B] = w[lora_B].to(device=device)
            w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r)
            w[k] = w[k].to(device='cpu')
            del w[lora_A]
            del w[lora_B]
torch.save(w, output)
