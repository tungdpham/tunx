#!/usr/bin/env python3
"""Replay ResNet50's final block and three SGD updates from an existing FP32 trajectory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch import nn
from run_layer_isolation import save


def load(path, shape):
    a = np.fromfile(path, dtype=np.float32).reshape(shape)
    t = torch.from_numpy(a).cuda()
    if t.ndim == 4:
        t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return t


def compare(reference, actual):
    a = np.fromfile(reference, np.float32).astype(np.float64)
    b = np.fromfile(actual, np.float32).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {reference}, {actual}')
    return dict(max_abs=float(np.max(np.abs(a-b))), relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)), sign_flips=int(np.count_nonzero((a>0)!=(b>0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', default='dump/trajectory_resnet50_42')
    parser.add_argument('--output-dir', default='dump/trajectory_isolation')
    args = parser.parse_args()
    source = Path(args.trajectory).resolve()
    out = Path(args.output_dir).resolve()
    pt = out / 'pytorch'
    pt.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    shape = (8, 7, 7, 2048)
    # Stage 4 block 1 has a projection shortcut; blocks 2 and 3 have identity shortcuts.
    x = load(source/'layer4_block1_bn3.output_step_1.bin', shape)
    x = (x + load(source/'layer4_block1_bn2.output_step_1.bin', shape).relu()).relu()
    x = (x + load(source/'layer4_block2_bn2.output_step_1.bin', shape).relu()).relu()
    x = x.detach().requires_grad_()
    save(pt/'inputs.bin', x)
    dy = load(source/'avgpool.grad_input_step_1.bin', shape)
    save(pt/'grad_output.bin', dy)
    layers = nn.ModuleDict()
    params = {}
    for i, (ic, oc, k) in enumerate([(2048,512,1),(512,512,3),(512,2048,1)]):
        convname = f'layer4_block3_conv{i+1}'
        bnname = f'layer4_block3_bn{i}'
        layers[convname] = nn.Conv2d(ic,oc,k,padding=k//2,bias=False)
        layers[bnname] = nn.BatchNorm2d(oc)
        layers[f'layer4_block3_relu{i+1}'] = nn.ReLU()
    layers.cuda().to(memory_format=torch.channels_last).train()
    for name, layer in layers.items():
        for pname, param in layer.named_parameters():
            filename = f'{name}.{pname}.bin'
            tshape = tuple(param.shape)
            if param.ndim == 4:
                tshape = (tshape[0], tshape[2], tshape[3], tshape[1])
            with torch.no_grad():
                param.copy_(load(source/filename, tshape))
            shutil.copyfile(source/filename, pt/filename)
            params[f'{name}.{pname}'] = param
    # Separate the main-branch input so its retained gradient excludes the shortcut.
    y = x * 1
    layer_inputs = {}
    for name, layer in layers.items():
        layer_inputs[name] = y
        y.retain_grad()
        y = layer(y)
        save(pt/f'{name}.output.bin', y)
    layer_inputs['layer4_block3_relu4'] = y+x
    layer_inputs['layer4_block3_relu4'].retain_grad()
    y = layer_inputs['layer4_block3_relu4'].relu()
    save(pt/'layer4_block3_relu4.output.bin', y)
    save(pt/'outputs.bin', y)
    y.backward(dy)
    for name, value in layer_inputs.items():
        save(pt/f'{name}.input_grad.bin', value.grad)
    for name, param in params.items():
        save(pt/f'{name}.grad.bin', param.grad)
    # Check reconstruction against the saved full-model first convolution.
    reconstruction = compare(source/'layer4_block3_conv1.output_step_1.bin', pt/'layer4_block3_conv1.output.bin')
    print('Reconstructed-input conv1 versus original trajectory:', reconstruction, flush=True)
    # Use flat parameters: SGD is elementwise, so this preserves the raw NHWC file layout.
    shared_params = {name: nn.Parameter(torch.from_numpy(np.fromfile(source/f'{name}.bin',np.float32)).cuda()) for name in params}
    optimizer = torch.optim.SGD(list(shared_params.values()),lr=1e-3,momentum=0.9,weight_decay=1e-4)
    for step in range(1,4):
        folder = pt/f'params_step_{step}'
        folder.mkdir(exist_ok=True)
        for name, param in shared_params.items():
            gradfile = source/f'params_step_{step}'/f'{name}.grad.bin'
            param.grad = torch.from_numpy(np.fromfile(gradfile,np.float32)).cuda()
            shutil.copyfile(gradfile,folder/gradfile.name)
        optimizer.step()
        for name, param in shared_params.items():
            save(folder/f'{name}.updated.bin',param)
    root = Path(__file__).resolve().parents[1]
    results = dict(reconstruction=reconstruction, replay=[], sgd=[])
    for mode in ['replay','sgd']:
        for engine in ['default','cuda','cudnn']:
            dest = out/f'{mode}_{engine}'
            dest.mkdir(exist_ok=True)
            with (dest/'runner.log').open('w') as log:
                subprocess.run([str(root/'build/bin/test_trajectory_isolation'),mode,engine,str(pt),str(dest)],check=True,env=dict(os.environ,NVIDIA_TF32_OVERRIDE='0'),stdout=log,stderr=subprocess.STDOUT)
            files = sorted(pt.glob('*.output.bin')) + sorted(pt.glob('*.input_grad.bin')) + sorted(pt.glob('*.grad.bin')) if mode=='replay' else sorted(pt.glob('params_step_*/*.updated.bin'))
            for ref in files:
                relative = ref.relative_to(pt)
                metrics = compare(ref,dest/relative)
                results[mode].append(dict(engine=engine,tensor=str(relative),**metrics))
                print(mode,engine,relative,metrics,flush=True)
            (out/'summary.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
