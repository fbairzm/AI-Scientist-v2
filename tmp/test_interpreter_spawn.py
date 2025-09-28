#!/usr/bin/env python3
from pathlib import Path
from ai_scientist.treesearch.interpreter import Interpreter
import os
import multiprocessing as mp

p = Path('tmp_test_working')
p.mkdir(exist_ok=True)

code = '''
import sys

import torch
# print('torch version:', getattr(torch, '__version__', None))
# print('torch.version.hip:', getattr(torch.version, 'hip', None))
# cuda_avail = getattr(torch, 'cuda', None) and torch.cuda.is_available()
# print('torch.cuda.is_available():', cuda_avail)
# if cuda_avail:
#     try:
#         a = torch.tensor([1.0]).to('cuda')
#         print('tensor on device:', a.device)
#     except Exception as e:
#         print('failed moving to cuda:', repr(e))
#         print('falling back to cpu')
# else:
# detect ROCm/HIP build
hip_ver = getattr(torch.version, 'hip', None)
if hip_ver:
    print('Detected ROCm/HIP build:', hip_ver)
    try:
        # On ROCm PyTorch builds, the CUDA APIs may still be used; try and catch.
        a = torch.tensor([1.0]).to('cuda')
        print('tensor on device (ROCm using cuda semantics):', a.device)
    except Exception as e:
        print('failed moving to cuda on ROCm:', repr(e))
        print('running on cpu')
else:
    print('no cuda/hip available, running on cpu')

print('sys.executable:', sys.executable)
print('sys.path:', sys.path)
print('sys.platform:', sys.platform)
print('sys.version:', sys.version) 
'''

def main():
    
    itp = Interpreter(p, timeout=20)
    res = itp.run(code, reset_session=True)
    #print('TERM_OUT:\n', '\n'.join(res.term_out))
    print('EXC_TYPE:', res.exc_type)
    print('EXC_STACK:', res.exc_stack) 
    print('EXEC_TIME:', res.exec_time)
    print('MEMORY_USED:', res.term_out)


if __name__ == '__main__':
    main()
