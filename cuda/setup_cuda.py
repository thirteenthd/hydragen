import os, glob, shutil

root = '/home/thirteenthd/cuda-toolkit'
shutil.rmtree(root, ignore_errors=True)
os.makedirs(f'{root}/bin', exist_ok=True)
os.makedirs(f'{root}/include', exist_ok=True)
os.makedirs(f'{root}/lib64', exist_ok=True)
os.symlink('lib64', f'{root}/lib')
os.symlink('/usr/local/cuda-13.1/nvvm', f'{root}/nvvm')

# Link bin
for f in glob.glob('/usr/local/cuda-13.1/bin/*'):
    os.symlink(f, f'{root}/bin/{os.path.basename(f)}')

# Link include from 13.1
for f in glob.glob('/usr/local/cuda-13.1/include/*'):
    os.symlink(f, f'{root}/include/{os.path.basename(f)}')

# Add missing cu13 headers (e.g. nvrtc.h) that don't collide
cu13_inc = '/home/thirteenthd/vllm-env/lib/python3.12/site-packages/nvidia/cu13/include'
if os.path.isdir(cu13_inc):
    for f in glob.glob(f'{cu13_inc}/*'):
        dest = f'{root}/include/{os.path.basename(f)}'
        if not os.path.exists(dest):
            os.symlink(f, dest)

# Add missing nvtx headers that don't collide
nvtx_inc = '/home/thirteenthd/vllm-env/lib/python3.12/site-packages/nvidia/nvtx/include'
if os.path.isdir(nvtx_inc):
    for f in glob.glob(f'{nvtx_inc}/*'):
        dest = f'{root}/include/{os.path.basename(f)}'
        if not os.path.exists(dest):
            os.symlink(f, dest)

# Link lib64 from 13.1
for f in glob.glob('/usr/local/cuda-13.1/lib64/*'):
    os.symlink(f, f'{root}/lib64/{os.path.basename(f)}')

# Link cu13 libs
cu13_lib = '/home/thirteenthd/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib'
if os.path.isdir(cu13_lib):
    for f in glob.glob(f'{cu13_lib}/*'):
        dest = f'{root}/lib64/{os.path.basename(f)}'
        if not os.path.exists(dest):
            os.symlink(f, dest)

# Link nvtx libs
nvtx_lib = '/home/thirteenthd/vllm-env/lib/python3.12/site-packages/nvidia/nvtx/lib'
if os.path.isdir(nvtx_lib):
    for f in glob.glob(f'{nvtx_lib}/*'):
        dest = f'{root}/lib64/{os.path.basename(f)}'
        if not os.path.exists(dest):
            os.symlink(f, dest)

# Create unversioned .so links
for f in glob.glob(f'{root}/lib64/*.so.*'):
    base = f.split('.so.')[0] + '.so'
    if not os.path.exists(base):
        os.symlink(os.path.basename(f), base)

print('Success! Checking cuda.h and nvrtc.h:')
with open(f'{root}/include/cuda.h') as fh:
    for line in fh:
        if '#define CUDA_VERSION' in line:
            print(line.strip())
print('nvrtc.h exists:', os.path.exists(f'{root}/include/nvrtc.h'))
print('libnvrtc.so exists:', os.path.exists(f'{root}/lib64/libnvrtc.so'))
print('libnvToolsExt.so exists:', os.path.exists(f'{root}/lib64/libnvToolsExt.so'))
