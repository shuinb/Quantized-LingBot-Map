"""Sample one child process's GPU memory, including CUDA context and libraries.

Example: python -m qmap memory --gpu 0 --output memory.json -- python -m qmap infer ...
The child inherits CUDA_VISIBLE_DEVICES; --gpu is the physical NVML index.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from qmap.common import memory_sampler


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qmap memory', description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=.25)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.gpu < 0 or not 0 < args.interval <= 10:
        parser.error('Require a child command, nonnegative physical GPU index, and sampling interval in (0,10]')
    sample, sampler = memory_sampler(args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Memory output already exists')
    report = dict(status='running', command=command, physical_gpu_index=args.gpu,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), sampler=sampler,
                  sample_interval_seconds=args.interval, samples=[],
                  scope='Child process GPU memory from launch to exit, including context/libraries and allocator reservations. Sampled peak can miss short transients; pair with PyTorch peak counters.')
    begin = time.perf_counter()
    with subprocess.Popen(command) as child:
        report['pid'] = child.pid
        args.output.write_text(json.dumps(report, indent=2))
        try:
            while child.poll() is None:
                used_mib = sample(child.pid)
                if used_mib is not None:
                    report['samples'].append([time.perf_counter() - begin, used_mib])
                time.sleep(args.interval)
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            report.update(status='complete' if child.returncode == 0 else 'failed',
                          returncode=child.returncode, seconds=time.perf_counter() - begin,
                          sample_count=len(report['samples']))
            peak = max((row[1] for row in report['samples']), default=None)
            report['sampled_peak_process_mib'] = peak
            report['sampled_peak_process_bytes'] = None if peak is None else int(peak * 2**20)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != 'samples'}), flush=True)
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
