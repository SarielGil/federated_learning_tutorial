#!/usr/bin/env python3
"""
Diagnostic script to check model training behavior
"""
import torch
import torch.nn as nn
from model import ConvNet2, LoRAConvNet2

print("="*60)
print("DIAGNOSTIC: Model Parameter Analysis")
print("="*60)

# Test 1: Check ConvNet2 parameters
print("\n1. ConvNet2 (Full Model)")
full_model = ConvNet2()
total_params = sum(p.numel() for p in full_model.parameters())
trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Test 2: Check LoRA model parameters
print("\n2. LoRAConvNet2 (rank=8)")
lora_model = LoRAConvNet2(rank=8)
total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")
print(f"   Trainable ratio: {100*trainable_params/total_params:.2f}%")

# Test 3: List trainable parameters in LoRA
print("\n3. Trainable Parameters in LoRA:")
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(f"   {name}: {param.shape}")

# Test 4: Check if gradients flow
print("\n4. Gradient Flow Test:")
lora_model.train()
dummy_input = torch.randn(2, 3, 224, 224)
dummy_target = torch.tensor([0, 1])
criterion = nn.CrossEntropyLoss()

output = lora_model(dummy_input)
loss = criterion(output, dummy_target)
loss.backward()

has_gradients = False
for name, param in lora_model.named_parameters():
    if param.requires_grad and param.grad is not None:
        if param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"   ✓ {name}: gradient norm = {param.grad.norm().item():.6f}")

if not has_gradients:
    print("   ✗ ERROR: No gradients detected!")
else:
    print("   ✓ Gradients are flowing correctly")

# Test 5: Check forward pass output
print("\n5. Forward Pass Test:")
print(f"   Input shape: {dummy_input.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output values: {output}")
print(f"   Loss: {loss.item():.4f}")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)
