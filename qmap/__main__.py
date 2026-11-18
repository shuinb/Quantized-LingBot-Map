"""Small command-line interface for the three-method comparison."""
import argparse
import importlib
import sys


COMMANDS = {
    'calibrate': 'Build calibrated W4A4 or DSFQ from independent calibration clips',
    'infer': 'Run BF16, calibrated W4A4, or DSFQ on an image sequence',
    'evaluate': 'Compare completed runs against BF16 and optional TUM ground truth',
    'visualize': 'Render trajectories and RGB point clouds from saved predictions',
    'gallery': 'Create a DSFQ RGB/depth/point-map showcase and optional orbit GIF',
    'depth-gallery': 'Compare the same RGB frames and relative depths across methods',
    'prepare-tum': 'Extract and associate an official TUM RGB-D archive',
    'memory': 'Measure one inference process with NVML',
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='python -m qmap', description=__doc__)
    parser.add_argument('command', choices=COMMANDS, help='Use COMMAND --help for its options')
    parser.epilog = '\n'.join(f'{name}: {description}' for name, description in COMMANDS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    module = importlib.import_module(f'qmap.{command.replace("-", "_")}')
    return module.main(argv[1:]) or 0


if __name__ == '__main__':
    raise SystemExit(main())
