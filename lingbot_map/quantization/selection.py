"""Capture anchor, local-history and post-eviction calibration states."""
import numpy as np


def stream_steps(frames, scale_frames, window, retained_anchors, interval=1):
    if min(frames, scale_frames, window, retained_anchors, interval) < 1 or frames < scale_frames:
        raise ValueError('Positive stream sizes required, with frames >= scale_frames')
    records = []
    appended = scale_frames
    for step in range(frames - scale_frames + 1):
        if step and (step - 1) % interval == 0:
            appended += 1
        phase = 'anchor' if not step else ('memory' if appended > retained_anchors + window else 'local')
        records.append({'step': step, 'frame_end': scale_frames + step,
                        'appended_frames': appended, 'phase': phase})
    return records


def balanced_capture_steps(records, samples):
    if samples < 3:
        raise ValueError('At least three samples per clip are needed for state coverage')
    groups = [[record['step'] for record in records if record['phase'] == phase]
              for phase in ('anchor', 'local', 'memory')]
    if any(not group for group in groups):
        raise ValueError('Missing calibration state phase')
    selected = {groups[0][0], groups[1][-1], groups[2][-1]}
    remaining = [record['step'] for record in records if record['step'] not in selected]
    needed = min(samples - len(selected), len(remaining))
    if needed:
        selected.update(remaining[index] for index in np.linspace(0, len(remaining) - 1, needed, dtype=int))
    return sorted(selected)
