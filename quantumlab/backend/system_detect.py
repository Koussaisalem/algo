"""
System detection utilities for runtime and hardware specifications.
Auto-detects GPU, CPU, RAM, storage, and environment type.
"""

import os
import platform
import psutil
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

def detect_runtime_type() -> str:
    """Detect the runtime environment type."""
    # Check for common cloud/container indicators
    if os.path.exists('/.dockerenv'):
        return 'docker'
    elif os.path.exists('/dev/.cgroup'):
        return 'container'
    elif os.path.exists('/workspace') or os.getenv('CODESPACES'):
        return 'codespace'
    elif os.getenv('AWS_EXECUTION_ENV') or os.path.exists('/var/lib/cloud'):
        return 'aws'
    elif os.path.exists('/var/lib/gce'):
        return 'gcp'
    elif os.path.exists('/var/lib/azure'):
        return 'azure'
    elif os.getenv('KUBERNETES_SERVICE_HOST'):
        return 'kubernetes'
    else:
        return 'local'

def detect_gpus() -> List[Dict[str, Any]]:
    """Detect available GPUs (NVIDIA, AMD, Intel)."""
    gpus = []
    
    # Try NVIDIA GPUs first
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'type': 'NVIDIA',
                        'memory_total_mb': int(parts[2]),
                        'memory_free_mb': int(parts[3]),
                        'memory_used_mb': int(parts[4]),
                        'utilization_percent': int(parts[5]),
                        'temperature_c': int(parts[6]) if parts[6] != 'N/A' else None
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # Try AMD ROCm
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            gpus.append({
                'type': 'AMD',
                'name': result.stdout.strip(),
                'driver': 'ROCm'
            })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    return gpus

def detect_cpu() -> Dict[str, Any]:
    """Detect CPU specifications."""
    try:
        cpu_info = {
            'name': platform.processor() or 'Unknown',
            'architecture': platform.machine(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'frequency_max_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'usage_percent': psutil.cpu_percent(interval=1),
        }
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['name'] = line.split(':')[1].strip()
                            break
            except:
                pass
        
        return cpu_info
    except Exception as e:
        return {'error': str(e)}

def detect_memory() -> Dict[str, Any]:
    """Detect RAM specifications."""
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'used_gb': round(mem.used / (1024**3), 2),
            'percent_used': mem.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2),
            'swap_percent': swap.percent
        }
    except Exception as e:
        return {'error': str(e)}

def detect_storage() -> List[Dict[str, Any]]:
    """Detect storage devices and usage."""
    storage_devices = []
    
    try:
        partitions = psutil.disk_partitions()
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                storage_devices.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'percent_used': usage.percent
                })
            except PermissionError:
                continue
    except Exception as e:
        storage_devices.append({'error': str(e)})
    
    return storage_devices

def detect_network() -> Dict[str, Any]:
    """Detect network interfaces and statistics."""
    try:
        net_io = psutil.net_io_counters()
        interfaces = psutil.net_if_addrs()
        
        return {
            'bytes_sent_gb': round(net_io.bytes_sent / (1024**3), 2),
            'bytes_recv_gb': round(net_io.bytes_recv / (1024**3), 2),
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'interfaces': list(interfaces.keys())
        }
    except Exception as e:
        return {'error': str(e)}

def detect_python_environment() -> Dict[str, Any]:
    """Detect Python and ML framework versions."""
    env_info = {
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
    }
    
    # Check for ML frameworks
    frameworks = {}
    try:
        import torch
        frameworks['pytorch'] = torch.__version__
        frameworks['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            frameworks['cuda_version'] = torch.version.cuda
            frameworks['cudnn_version'] = torch.backends.cudnn.version()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        frameworks['tensorflow'] = tf.__version__
    except ImportError:
        pass
    
    try:
        import numpy as np
        frameworks['numpy'] = np.__version__
    except ImportError:
        pass
    
    env_info['frameworks'] = frameworks
    return env_info

def get_system_specs() -> Dict[str, Any]:
    """Get complete system specifications."""
    return {
        'runtime': {
            'type': detect_runtime_type(),
            'os': platform.system(),
            'os_version': platform.version(),
            'os_release': platform.release(),
            'hostname': platform.node(),
        },
        'cpu': detect_cpu(),
        'memory': detect_memory(),
        'gpus': detect_gpus(),
        'storage': detect_storage(),
        'network': detect_network(),
        'environment': detect_python_environment(),
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }

def get_recommendations(specs: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on detected specs."""
    recommendations = []
    
    # Check GPU availability
    if not specs['gpus']:
        recommendations.append('⚠️ No GPU detected. Training will be slow. Consider using cloud GPU instances.')
    elif len(specs['gpus']) > 1:
        recommendations.append(f'✅ {len(specs["gpus"])} GPUs detected. Use distributed training for faster results.')
    
    # Check memory
    mem = specs['memory']
    if mem.get('total_gb', 0) < 8:
        recommendations.append('⚠️ Low RAM detected (<8GB). Reduce batch size to avoid out-of-memory errors.')
    elif mem.get('percent_used', 0) > 80:
        recommendations.append('⚠️ High memory usage detected. Close unnecessary applications.')
    
    # Check storage
    for storage in specs['storage']:
        if storage.get('percent_used', 0) > 90:
            recommendations.append(f'⚠️ Low disk space on {storage["mountpoint"]}. Clean up to avoid issues.')
    
    # Check Python environment
    env = specs['environment']
    if not env.get('frameworks', {}).get('pytorch'):
        recommendations.append('⚠️ PyTorch not detected. Install it for model training.')
    
    if not recommendations:
        recommendations.append('✅ System configuration looks good!')
    
    return recommendations
