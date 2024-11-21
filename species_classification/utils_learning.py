'''Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import numpy as np
import torch


def verification(out, pert_out, threshold=0.0):
    '''
    return the ratio of qualified samples.
    '''
    if isinstance(out, torch.Tensor):
        return 1.0 * torch.sum((pert_out - out) < threshold) / out.shape[0]
    else:
        return 1.0 * np.sum((pert_out - out) < threshold) / out.shape[0]


def rule1_fxn(input_tensor, thr):
    device = input_tensor.device
    sig = torch.nn.Sigmoid()
    return sig(1.e3*(-1 * torch.amax(input_tensor, dim=(1,2)) + thr))    # r1 & r2
    #return sig(-1.e3*(-1 * torch.amax(input_tensor, dim=(1,2)) + thr))  # r3
    #return sig(-1.e3*(-1 * input_tensor + thr))                         # r4
