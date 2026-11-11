"""Explicit CPU/CUDA/MPS device selection and honest memory accounting."""
from __future__ import annotations

import platform
import resource
import sys

import torch


def select_device(requested='auto'):
    requested = str(requested)
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Check the NVIDIA driver and PyTorch CUDA wheel, or use --device cpu.')
    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('MPS was requested but is unavailable on this machine.')
    if device.type not in ('cuda', 'cpu', 'mps'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N, mps')
    if device.type == 'cuda':
        # Retain FP32 arithmetic for comparison with the supplied experiment.
        # Changing TF32/AMP is an explicit new speed/accuracy setting.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def sync(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def reset_peak_memory(device):
    device = torch.device(device)
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device):
    """CUDA peaks are per reset; RSS is process-lifetime high water, not VRAM."""
    device = torch.device(device)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024)
    out = dict(device=str(device), process_peak_rss_bytes=rss_bytes,
               process_peak_rss_scope='process lifetime high-water mark; CPU RAM, not GPU memory')
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        out.update(gpu_name=prop.name, gpu_total_bytes=int(prop.total_memory),
                   gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                   gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                   gpu_current_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                   gpu_current_reserved_bytes=int(torch.cuda.memory_reserved(device)),
                   gpu_peak_scope='PyTorch allocations since last reset, including model weights already resident; excludes driver/other processes')
    elif device.type == 'mps':
        out.update(mps_current_allocated_bytes=int(torch.mps.current_allocated_memory()),
                   mps_driver_allocated_bytes=int(torch.mps.driver_allocated_memory()),
                   mps_memory_scope='current allocations only; no CUDA VRAM peak inferred')
    return out


def environment_info(device='auto'):
    device = select_device(device)
    out = dict(platform=platform.platform(), python=platform.python_version(),
               torch=torch.__version__, torch_cuda_build=torch.version.cuda,
               device=str(device), cuda_available=torch.cuda.is_available(),
               cpu_torch_threads=torch.get_num_threads())
    if device.type == 'cuda':
        out.update(gpu_name=torch.cuda.get_device_name(device),
                   compute_capability=list(torch.cuda.get_device_capability(device)),
                   compiled_architectures=torch.cuda.get_arch_list())
    return out
