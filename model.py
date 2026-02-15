# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    pytorch model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding = 1), # (64*224*224)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2, padding=1), #(128*112*112)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),                             #(128*112*112)
            # layer 1 complete
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2, padding=1), #(256*56*56)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),                             #(256*56*56)
            #layer 2 complete
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2, padding=1), #(512*28*28)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),                             #(512*28*28)
            # Layer 3 complete
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=2, padding=1), #(512*14*14)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),                             #(512*14*14)
            # layer 4 complete
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=2, padding=1), #(512*7*7)
            nn.BatchNorm2d(512),
            nn.ReLU(), 
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        out = self.conv2(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freezing the original weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # LoRA weights
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scaling = self.alpha / self.rank
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class LoRAConvNet2(nn.Module):
    def __init__(self, rank=4, base_model=None):
        super().__init__()
        self.base_model = base_model if base_model is not None else ConvNet2()
        
        # Wrap the linear layers in the fc sequential block with LoRALinear
        # New indices due to ReLU and Dropout: 0, 3, 6, 8
        self.base_model.fc[0] = LoRALinear(self.base_model.fc[0], rank=rank)
        self.base_model.fc[3] = LoRALinear(self.base_model.fc[3], rank=rank)
        self.base_model.fc[6] = LoRALinear(self.base_model.fc[6], rank=rank)
        self.base_model.fc[8] = LoRALinear(self.base_model.fc[8], rank=rank)

    def forward(self, x):
        return self.base_model(x)

    def get_base_model(self):
        return self.base_model