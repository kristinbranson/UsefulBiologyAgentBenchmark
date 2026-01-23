#!/usr/bin/env python
"""
Crop experiment directories: videos and mat files to a specified split frame.

Reads experiment directories from CSV, calculates split frames from indicator data,
and crops videos/mat files to the split frame. Also anonymizes Metadata.xml.

Usage:
    conda activate APT2507

    # Show help
    python crop_expdirs.py --help

    # Dry run - just compute split frames and update CSV
    python crop_expdirs.py --dry-run

    # Crop only mat files (trx.mat and movie-track.mat)
    python crop_expdirs.py --mat

    # Crop only videos
    python crop_expdirs.py --video

    # Crop everything (video + mat + metadata)
    python crop_expdirs.py --all

    # Force redo mat files even if they exist
    python crop_expdirs.py --mat --force

    # Crop specific mat files
    python crop_expdirs.py --trx --track

    # Anonymize metadata only
    python crop_expdirs.py --metadata

    # Use different CSV file
    python crop_expdirs.py --csv myfile.csv --mat
"""

import argparse
import os
import re
import shutil
import numpy as np
import scipy.io as sio
from multiprocessing import Pool, cpu_count

from movies import Movie
from crop_ufmf import cropufmf
from matio import loadmat, savemat
import TrkFile


# =============================================================================
# Configuration defaults
# =============================================================================

DEFAULT_CSV = "expdirlist20260115_anonymized.csv"
DEFAULT_SPLIT_PULSE = 10
DEFAULT_WORKERS = 8


# =============================================================================
# Utility functions
# =============================================================================

def get_frame_count(movie_path):
    """Get frame count from a movie file."""
    try:
        mov = Movie(movie_path)
        nframes = mov.get_n_frames()
        mov.close()
        return nframes
    except Exception as e:
        return f"ERROR: {e}"


def get_indicator_data(expdir, split_after=None):
    """
    Read indicatordata.mat and return summary info.

    Args:
        expdir: experiment directory path
        split_after: pulse number (1-indexed) to calculate midpoint after.
                     If 10, calculates midpoint between end of pulse 10 and start of pulse 11.

    Returns dict with pulse timing info and optionally split_frame.
    """
    indicator_path = os.path.join(expdir, "indicatordata.mat")

    if not os.path.exists(indicator_path):
        return None

    try:
        mat = sio.loadmat(indicator_path)
        led = mat['indicatorLED'][0, 0]

        startframes = led['startframe'][0]
        endframes = led['endframe'][0]
        indicatordigital = led['indicatordigital'][0]

        result = {
            'rect': mat['rect'][0],
            'n_pulses': len(startframes),
            'startframes': startframes,
            'endframes': endframes,
            'pulse_duration': int(np.median(endframes - startframes)),
            'starttimes': led['starttimes'][0],
            'endtimes': led['endtimes'][0],
            'n_on_frames': int(np.sum(indicatordigital == 1)),
            'n_off_frames': int(np.sum(indicatordigital == 0)),
            'first_pulse_frame': int(startframes[0]),
            'last_pulse_end_frame': int(endframes[-1]),
        }

        # Calculate midpoint between end of pulse N and start of pulse N+1
        if split_after is not None and split_after < len(startframes):
            idx = split_after - 1  # Convert 1-indexed to 0-indexed
            end_pulse_n = int(endframes[idx])
            start_pulse_n1 = int(startframes[idx + 1])
            split_frame = (end_pulse_n + start_pulse_n1) // 2
            result['end_pulse_n'] = end_pulse_n
            result['start_pulse_n1'] = start_pulse_n1
            result['split_frame'] = split_frame

        return result
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# Video cropping
# =============================================================================

def crop_video(movie_path, out_movie, split_frame):
    """
    Crop a video to the specified frame range.

    Args:
        movie_path: input movie file path
        out_movie: output movie file path
        split_frame: last frame to include (0-indexed)

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        os.makedirs(os.path.dirname(out_movie), exist_ok=True)
        cropufmf(
            inmoviefile=movie_path,
            cropframes=[0, split_frame],
            outmoviefiles=[out_movie],
        )
        return True, 'cropped'
    except Exception as e:
        return False, f'error: {e}'


# =============================================================================
# Mat file cropping
# =============================================================================

def _normalize_trx_to_dict(trx_data):
    """
    Normalize trx data to dict-of-lists format.

    Handles both:
    - dict format (from matio.loadmat/h5py)
    - array of mat_struct objects (from scipy.io.loadmat with struct_as_record=False)

    Returns:
        tuple of (field_names, trx_dict) where trx_dict is {field: [val_fly0, val_fly1, ...]}
    """
    if isinstance(trx_data, dict):
        # Already dict format
        field_names = list(trx_data.keys())
        return field_names, trx_data
    else:
        # scipy format: (1, nflies) array of mat_struct objects - flatten to 1D
        trx_arr = trx_data.flatten()
        if len(trx_arr) == 0:
            return [], {}
        field_names = trx_arr[0]._fieldnames
        trx_dict = {field: [getattr(fly, field) for fly in trx_arr] for field in field_names}
        return field_names, trx_dict


def _crop_trx_dict(trx_dict, field_names, split_frame):
    """
    Crop trx data (in dict-of-lists format) to split_frame.
    Removes flies that don't exist before split_frame.

    Args:
        trx_dict: dict of {field: [val_fly0, val_fly1, ...]}
        field_names: list of field names
        split_frame: last frame to include (0-indexed)

    Returns:
        cropped trx_dict
    """
    per_frame_fields = {'x', 'y', 'a', 'b', 'theta', 'x_mm', 'y_mm', 'a_mm', 'b_mm',
                        'theta_mm', 'timestamps', 'dt', 'xwingl', 'ywingl', 'xwingr', 'ywingr', 'sex'}

    # Get number of flies
    nflies = len(trx_dict[field_names[0]]) if field_names else 0

    # First pass: identify which flies to keep (firstframe <= split_frame)
    keep_indices = []
    for i in range(nflies):
        firstframe = int(np.array(trx_dict['firstframe'][i]).flatten()[0])
        # Keep fly if it exists before or at split_frame (1-indexed, so split_frame+1)
        if firstframe <= split_frame + 1:
            keep_indices.append(i)

    # Second pass: crop kept flies and build new trx_dict
    new_trx_dict = {field: [] for field in field_names}

    for i in keep_indices:
        firstframe = int(np.array(trx_dict['firstframe'][i]).flatten()[0])
        endframe = int(np.array(trx_dict['endframe'][i]).flatten()[0])

        # Calculate new endframe (can't exceed split_frame+1, 1-indexed)
        new_endframe = min(endframe, split_frame + 1)
        new_nframes = new_endframe - firstframe + 1

        for field in field_names:
            val = trx_dict[field][i]

            if field == 'endframe':
                new_trx_dict[field].append(np.array([[float(new_endframe)]]))
            elif field == 'nframes':
                new_trx_dict[field].append(np.array([[float(new_nframes)]]))
            elif field in per_frame_fields:
                # Skip if val is a string (single-char sex) or None
                if isinstance(val, str) or val is None:
                    new_trx_dict[field].append(val)
                    continue
                arr = np.array(val)
                if arr.size == 0:
                    new_trx_dict[field].append(arr)
                    continue
                # dt has nframes-1 elements (time differences between frames)
                crop_len = new_nframes - 1 if field == 'dt' else new_nframes
                if arr.ndim == 1:
                    new_trx_dict[field].append(arr[:crop_len])
                elif arr.shape[0] == 1:
                    new_trx_dict[field].append(arr[:, :crop_len])
                elif arr.shape[1] == 1:
                    new_trx_dict[field].append(arr[:crop_len, :])
                else:
                    new_trx_dict[field].append(arr[:, :crop_len])
            else:
                # Scalar or other fields - keep as-is
                new_trx_dict[field].append(val)

    return new_trx_dict


def crop_trx_mat(infile, outfile, split_frame):
    """
    Crop trx.mat file to only include frames up to split_frame.

    Args:
        infile: input trx.mat path
        outfile: output trx.mat path
        split_frame: last frame to include (0-indexed)

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        data, dtype = loadmat(infile)

        # Crop top-level timestamps
        if 'timestamps' in data:
            ts = np.array(data['timestamps'])
            if ts.shape[0] > ts.shape[1]:
                data['timestamps'] = ts[:split_frame+1, :]
            else:
                data['timestamps'] = ts[:, :split_frame+1]

        # Normalize and crop trx data
        field_names, trx_dict = _normalize_trx_to_dict(data['trx'])
        data['trx'] = _crop_trx_dict(trx_dict, field_names, split_frame)

        # Remove matlab metadata keys that cause issues
        for key in ['__header__', '__version__', '__globals__']:
            data.pop(key, None)

        # savemat handles struct array conversion
        # transpose_arrays=False because data is already in correct (row vector) orientation
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        savemat(outfile, data, matfiletype='hdf5storage', transpose_arrays=False)
        return True, 'cropped'
    except Exception as e:
        return False, f'error: {e}'


def crop_track_mat(infile, outfile, split_frame):
    """
    Crop movie-track.mat file to only include frames up to split_frame.

    Args:
        infile: input movie-track.mat path
        outfile: output movie-track.mat path
        split_frame: last frame to include (0-indexed)

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        data, dtype = loadmat(infile)
        trk = data['trk']

        # trk['data'] loaded from HDF5 has shape (n_features, n_frames, n_flies)
        # Crop frames (axis 1)
        if 'data' in trk:
            trk_data = np.array(trk['data'])
            trk['data'] = trk_data[:, :split_frame+1, :]

        # savemat handles struct array conversion
        # transpose_arrays=False because data is already in correct orientation
        savemat(outfile, data, matfiletype='hdf5storage', transpose_arrays=False)
        return True, 'cropped'
    except Exception as e:
        return False, f'error: {e}'


def crop_protocol_mat(infile, outfile, step_idx=0):
    """
    Crop protocol.mat to keep only one step.

    Args:
        infile: input protocol.mat path
        outfile: output protocol.mat path
        step_idx: which step to keep (0-indexed, default 0 = Step 1)

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        import scipy.io as sio

        # Load with struct_as_record=False to get mat_struct objects
        data = sio.loadmat(infile, struct_as_record=False, squeeze_me=False)
        protocol = data['protocol'][0, 0]  # Extract the struct

        # Fields that have one value per step (shape (3, 1) typically)
        step_fields = [
            'stepNum', 'duration', 'delayTime',
            'Rintensity', 'RpulseWidth', 'RpulsePeriod', 'RpulseNum', 'RoffTime', 'Riteration',
            'Gintensity', 'GpulseWidth', 'GpulsePeriod', 'GpulseNum', 'GoffTime', 'Giteration',
            'Bintensity', 'BpulseWidth', 'BpulsePeriod', 'BpulseNum', 'BoffTime', 'Biteration',
        ]

        # Create new protocol dict with cropped fields
        new_protocol = {}
        for field in step_fields:
            if hasattr(protocol, field):
                arr = getattr(protocol, field)
                # Arrays are shape (n_steps, 1), keep only the specified step
                if arr.ndim == 1:
                    cropped = arr[step_idx:step_idx+1]
                elif arr.shape[0] >= arr.shape[1]:
                    # Shape is (n_steps, 1) - crop along axis 0
                    cropped = arr[step_idx:step_idx+1, :]
                else:
                    # Shape is (1, n_steps) - crop along axis 1
                    cropped = arr[:, step_idx:step_idx+1]
                # Convert to double for MATLAB compatibility
                new_protocol[field] = cropped.astype(np.float64)

        # ProtocolData: shape (3, 21) -> (1, 21)
        if hasattr(protocol, 'ProtocolData'):
            pd = protocol.ProtocolData
            new_protocol['ProtocolData'] = pd[step_idx:step_idx+1, :].astype(np.float64)

        # ProtocolHeader: keep as-is (column labels)
        if hasattr(protocol, 'ProtocolHeader'):
            new_protocol['ProtocolHeader'] = protocol.ProtocolHeader

        # Save using scipy (v5 format, same as original)
        os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)
        sio.savemat(outfile, {'protocol': new_protocol}, appendmat=False)
        return True, 'cropped'
    except Exception as e:
        return False, f'error: {e}'


def copy_mat_file(src, dst):
    """
    Copy a mat file.

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
            shutil.copy2(src, dst)
            return True, 'copied'
        else:
            return False, 'not found'
    except Exception as e:
        return False, f'error: {e}'


def crop_registered_trx_mat(infile, outfile, split_frame):
    """
    Crop registered_trx.mat file to only include frames up to split_frame.
    Uses the same logic as crop_trx_mat since both have the same trx structure.

    Args:
        infile: input registered_trx.mat path
        outfile: output registered_trx.mat path
        split_frame: last frame to include (0-indexed)

    Returns:
        tuple of (success: bool, message: str)
    """
    return crop_trx_mat(infile, outfile, split_frame)


def crop_registration_mat(infile, outfile, split_frame):
    """
    Copy registrationdata.mat and modify end_frame and seconds_crop_end.

    Args:
        infile: input registrationdata.mat path
        outfile: output registrationdata.mat path
        split_frame: new end_frame value

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        import scipy.io as sio

        data = sio.loadmat(infile)

        # Update end_frame to split_frame
        if 'end_frame' in data:
            data['end_frame'] = np.array([[float(split_frame)]])

        # Set seconds_crop_end to 0
        if 'seconds_crop_end' in data:
            data['seconds_crop_end'] = np.array([[0.0]])

        # Save
        os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)
        sio.savemat(outfile, data, appendmat=False)
        return True, 'cropped'
    except Exception as e:
        return False, f'error: {e}'


def crop_apt_trk(infile, outfile, split_frame):
    """
    Crop apt.trk file to only include frames up to split_frame.

    Uses the TrkFile.Trk class to load, crop, and save the tracking data.
    Preserves the original file format (tracklet/sparse/full).

    Args:
        infile: input apt.trk path
        outfile: output apt.trk path
        split_frame: last frame to include (0-indexed)

    Returns:
        tuple of (success: bool, message: str)
    """
    trk = TrkFile.Trk(trkfile=infile)
    for fly in range(trk.ntargets):
        p = trk.gettarget(fly)
        p[:,:,split_frame:] = trk.defaultval
        trk.settarget(p,fly)
    trk.consolidate()
    trk.save(outfile)

    return True, 'cropped'

# =============================================================================
# Metadata anonymization
# =============================================================================

def get_anon_line_name(anon_expdir):
    """
    Extract the anonymized line name from the anonymized experiment directory path.

    E.g., '.../VNC2_Line0_RigA_20230524T111117' -> 'Line0'
    """
    basename = os.path.basename(anon_expdir)
    match = re.search(r'_(Line\d+)_', basename)
    if match:
        return match.group(1)
    return 'LineX'


def anonymize_metadata(infile, outfile, anon_line_name):
    """
    Create an anonymized version of Metadata.xml.

    Anonymizes:
    - line name -> anon_line_name (e.g., 'Line0')
    - experimenter -> 'anonymous'
    - cross_barcode -> '000000'
    - robot_stock_copy -> '0'
    - handler_* -> 'anonymous'
    - wish_list -> '0'

    Args:
        infile: input Metadata.xml path
        outfile: output Metadata.xml path
        anon_line_name: anonymized line name (e.g., 'Line0')

    Returns:
        tuple of (success: bool, message: str)
    """
    try:
        if not os.path.exists(infile):
            return False, 'not found'

        with open(infile, 'r') as f:
            content = f.read()

        # Anonymize line name
        content = re.sub(r'line="[^"]*"', f'line="{anon_line_name}"', content)

        # Anonymize experimenter
        content = re.sub(r'experimenter="[^"]*"', 'experimenter="anonymous"', content)

        # Anonymize cross_barcode
        content = re.sub(r'cross_barcode="[^"]*"', 'cross_barcode="000000"', content)

        # Anonymize robot_stock_copy
        content = re.sub(r'robot_stock_copy="[^"]*"', 'robot_stock_copy="0"', content)

        # Anonymize wish_list
        content = re.sub(r'wish_list="[^"]*"', 'wish_list="0"', content)

        # Anonymize handler_* fields
        content = re.sub(r'handler_cross="[^"]*"', 'handler_cross="anonymous"', content)
        content = re.sub(r'handler_sorting="[^"]*"', 'handler_sorting="anonymous"', content)
        content = re.sub(r'handler_starvation="[^"]*"', 'handler_starvation="anonymous"', content)

        os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)
        with open(outfile, 'w') as f:
            f.write(content)

        return True, 'anonymized'
    except Exception as e:
        return False, f'error: {e}'


# =============================================================================
# Main processing function
# =============================================================================

def process_experiment(args):
    """
    Process a single experiment. Designed to be called by multiprocessing.Pool.

    Args:
        args: dict with keys:
            - idx, total: for progress display
            - orig_expdir, anon_expdir: directories
            - split_frame: frame to crop to
            - movie_path: path to input movie
            - do_video, do_trx, do_track, do_copy, do_metadata: what to process
            - force: whether to redo existing files

    Returns:
        dict with 'success', 'exp_name', 'results'
    """
    idx = args['idx']
    total = args['total']
    orig_expdir = args['orig_expdir']
    anon_expdir = args['anon_expdir']
    split_frame = args['split_frame']
    movie_path = args['movie_path']
    do_video = args.get('do_video', False)
    do_trx = args.get('do_trx', False)
    do_track = args.get('do_track', False)
    do_copy = args.get('do_copy', False)
    do_registration = args.get('do_registration', False)
    do_protocol = args.get('do_protocol', False)
    do_metadata = args.get('do_metadata', False)
    do_apt = args.get('do_apt', False)
    force = args.get('force', False)

    exp_name = os.path.basename(orig_expdir)
    results = {}

    # Determine what needs to be done
    tasks = []
    if do_video:
        tasks.append('video')
    if do_trx:
        tasks.append('trx')
    if do_track:
        tasks.append('track')
    if do_copy:
        tasks.append('copy')
    if do_registration:
        tasks.append('registration')
    if do_protocol:
        tasks.append('protocol')
    if do_metadata:
        tasks.append('metadata')
    if do_apt:
        tasks.append('apt')

    if not tasks:
        return {'success': True, 'exp_name': exp_name, 'results': {}, 'skipped': True}

    print(f"[{idx+1}/{total}] Processing {exp_name} ({', '.join(tasks)})")

    try:
        os.makedirs(anon_expdir, exist_ok=True)

        # Video cropping
        if do_video:
            out_movie = os.path.join(anon_expdir, "movie.ufmf")
            if force or not os.path.exists(out_movie):
                success, msg = crop_video(movie_path, out_movie, split_frame)
                results['video'] = msg
            else:
                results['video'] = 'exists'

        # trx.mat cropping
        if do_trx:
            trx_src = os.path.join(orig_expdir, 'movie_JAABA', 'trx.mat')
            trx_dst = os.path.join(anon_expdir, 'movie_JAABA', 'trx.mat')
            if os.path.exists(trx_src):
                if force or not os.path.exists(trx_dst):
                    success, msg = crop_trx_mat(trx_src, trx_dst, split_frame)
                    results['trx'] = msg
                else:
                    results['trx'] = 'exists'
            else:
                results['trx'] = 'not found'

        # movie-track.mat cropping
        if do_track:
            track_src = os.path.join(orig_expdir, 'movie-track.mat')
            track_dst = os.path.join(anon_expdir, 'movie-track.mat')
            if os.path.exists(track_src):
                if force or not os.path.exists(track_dst):
                    success, msg = crop_track_mat(track_src, track_dst, split_frame)
                    results['track'] = msg
                else:
                    results['track'] = 'exists'
            else:
                results['track'] = 'not found'

        # Copy other mat files
        if do_copy:
            copy_files = [
                ('movie-bg.mat', 'movie-bg.mat'),
                ('flytracker-calibration.mat', 'flytracker-calibration.mat'),
            ]
            for src_name, dst_name in copy_files:
                src = os.path.join(orig_expdir, src_name)
                dst = os.path.join(anon_expdir, dst_name)
                if force or not os.path.exists(dst):
                    success, msg = copy_mat_file(src, dst)
                    results[src_name] = msg
                else:
                    results[src_name] = 'exists'

        # Copy and modify registrationdata.mat and registered_trx.mat
        if do_registration:
            # registrationdata.mat
            reg_src = os.path.join(orig_expdir, 'registrationdata.mat')
            reg_dst = os.path.join(anon_expdir, 'registrationdata.mat')
            if os.path.exists(reg_src):
                if force or not os.path.exists(reg_dst):
                    success, msg = crop_registration_mat(reg_src, reg_dst, split_frame)
                    results['registrationdata'] = msg
                else:
                    results['registrationdata'] = 'exists'
            else:
                results['registrationdata'] = 'not found'

            # registered_trx.mat
            regtrx_src = os.path.join(orig_expdir, 'registered_trx.mat')
            regtrx_dst = os.path.join(anon_expdir, 'registered_trx.mat')
            if os.path.exists(regtrx_src):
                if force or not os.path.exists(regtrx_dst):
                    success, msg = crop_registered_trx_mat(regtrx_src, regtrx_dst, split_frame)
                    results['registered_trx'] = msg
                else:
                    results['registered_trx'] = 'exists'
            else:
                results['registered_trx'] = 'not found'

        # Crop protocol.mat to keep only step 1
        if do_protocol:
            proto_src = os.path.join(orig_expdir, 'protocol.mat')
            proto_dst = os.path.join(anon_expdir, 'protocol.mat')
            if os.path.exists(proto_src):
                if force or not os.path.exists(proto_dst):
                    success, msg = crop_protocol_mat(proto_src, proto_dst, step_idx=0)
                    results['protocol'] = msg
                else:
                    results['protocol'] = 'exists'
            else:
                results['protocol'] = 'not found'

        # Anonymize Metadata.xml
        if do_metadata:
            meta_src = os.path.join(orig_expdir, 'Metadata.xml')
            meta_dst = os.path.join(anon_expdir, 'Metadata.xml')
            if force or not os.path.exists(meta_dst):
                anon_line_name = get_anon_line_name(anon_expdir)
                success, msg = anonymize_metadata(meta_src, meta_dst, anon_line_name)
                results['metadata'] = msg
            else:
                results['metadata'] = 'exists'

        # Crop apt.trk
        if do_apt:
            apt_src = os.path.join(orig_expdir, 'apt.trk')
            apt_dst = os.path.join(anon_expdir, 'apt.trk')
            if os.path.exists(apt_src):
                if force or not os.path.exists(apt_dst):
                    success, msg = crop_apt_trk(apt_src, apt_dst, split_frame)
                    results['apt'] = msg
                else:
                    results['apt'] = 'exists'
            else:
                results['apt'] = 'not found'

        print(f"[{idx+1}/{total}] Finished {exp_name}: {results}")
        return {'success': True, 'exp_name': exp_name, 'results': results, 'skipped': False}

    except Exception as e:
        print(f"[{idx+1}/{total}] ERROR {exp_name}: {e}")
        return {'success': False, 'exp_name': exp_name, 'results': results, 'error': str(e), 'skipped': False}


# =============================================================================
# CSV handling
# =============================================================================

def read_csv(csv_file):
    """Read CSV file and return list of (orig_expdir, anon_expdir, split_frame or None)."""
    entries = []
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            orig = parts[0]
            anon = parts[1]
            split = int(parts[2]) if len(parts) > 2 else None
            entries.append((orig, anon, split))
    return entries


def write_csv(csv_file, entries):
    """Write entries to CSV file."""
    with open(csv_file, 'w') as f:
        for orig, anon, split in entries:
            if split is not None:
                f.write(f"{orig},{anon},{split}\n")
            else:
                f.write(f"{orig},{anon}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Crop experiment directories (videos and mat files) to split frame.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crop_expdirs.py --dry-run          # Just compute split frames, update CSV
  python crop_expdirs.py --mat              # Crop all mat files (trx + track + copy)
  python crop_expdirs.py --mat --force      # Redo all mat files even if they exist
  python crop_expdirs.py --trx --track      # Crop only trx.mat and movie-track.mat
  python crop_expdirs.py --video            # Crop only videos
  python crop_expdirs.py --all              # Crop everything
  python crop_expdirs.py --all --force      # Redo everything
        """
    )

    # Input/output options
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Input CSV file (default: {DEFAULT_CSV})')
    parser.add_argument('--split-pulse', type=int, default=DEFAULT_SPLIT_PULSE,
                        help=f'Pulse number to split after (default: {DEFAULT_SPLIT_PULSE})')
    parser.add_argument('--workers', '-j', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')

    # What to do
    parser.add_argument('--dry-run', action='store_true',
                        help='Only compute split frames and update CSV, no cropping')
    parser.add_argument('--all', action='store_true',
                        help='Crop everything (video + all mat files)')
    parser.add_argument('--mat', action='store_true',
                        help='Crop all mat files (trx + track + copy)')
    parser.add_argument('--video', action='store_true',
                        help='Crop videos')
    parser.add_argument('--trx', action='store_true',
                        help='Crop trx.mat files')
    parser.add_argument('--track', action='store_true',
                        help='Crop movie-track.mat files')
    parser.add_argument('--copy', action='store_true',
                        help='Copy other mat files (movie-bg.mat, calibration)')
    parser.add_argument('--registration', action='store_true',
                        help='Crop registrationdata.mat (update end_frame)')
    parser.add_argument('--protocol', action='store_true',
                        help='Crop protocol.mat to keep only step 1')
    parser.add_argument('--metadata', action='store_true',
                        help='Anonymize and copy Metadata.xml')
    parser.add_argument('--apt', action='store_true',
                        help='Crop apt.trk files')

    # Behavior options
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force redo even if output files exist')
    parser.add_argument('--no-update-csv', action='store_true',
                        help='Do not update CSV with split frames')

    args = parser.parse_args()

    # Resolve what to do
    if args.all:
        do_video = do_trx = do_track = do_copy = do_registration = do_protocol = do_metadata = do_apt = True
    elif args.mat:
        do_video = False
        do_trx = do_track = do_copy = do_registration = do_protocol = do_metadata = do_apt = True
    else:
        do_video = args.video
        do_trx = args.trx
        do_track = args.track
        do_copy = args.copy
        do_registration = args.registration
        do_protocol = args.protocol
        do_metadata = args.metadata
        do_apt = args.apt

    # If nothing specified and not dry-run, show help
    if not args.dry_run and not any([do_video, do_trx, do_track, do_copy, do_registration, do_protocol, do_metadata, do_apt]):
        parser.print_help()
        print("\nError: Specify what to process (--all, --mat, --video, --trx, --track, --copy, --registration, --protocol, --metadata) or --dry-run")
        return 1

    # Read CSV
    print(f"Reading {args.csv}...")
    entries = read_csv(args.csv)
    print(f"  Found {len(entries)} experiments")

    # Compute split frames if needed
    n = args.split_pulse
    print(f"\n{'Experiment':<55} {'Frames':>8} {f'End{n}':>7} {f'Start{n+1}':>8} {f'Mid{n}-{n+1}':>9}")
    print("-" * 92)

    updated_entries = []
    for orig, anon, existing_split in entries:
        exp_name = os.path.basename(orig)
        movie_path = os.path.join(orig, "movie.ufmf")

        nframes = get_frame_count(movie_path)
        indicator = get_indicator_data(orig, split_after=args.split_pulse)

        if isinstance(nframes, int) and indicator and 'error' not in indicator:
            split_frame = indicator['split_frame']
            print(f"{exp_name:<55} {nframes:>8} {indicator['end_pulse_n']:>7} "
                  f"{indicator['start_pulse_n1']:>8} {split_frame:>9}")
            updated_entries.append((orig, anon, split_frame))
        elif isinstance(nframes, int) and existing_split is not None:
            print(f"{exp_name:<55} {nframes:>8} (using existing split: {existing_split})")
            updated_entries.append((orig, anon, existing_split))
        else:
            err = indicator.get('error', 'NO INDICATOR DATA') if indicator else 'NO INDICATOR DATA'
            print(f"{exp_name:<55} {nframes if isinstance(nframes, int) else 'ERROR':>8} {err}")
            continue

    print("-" * 92)
    print(f"Valid experiments: {len(updated_entries)}")

    # Update CSV
    if not args.no_update_csv:
        print(f"\nUpdating {args.csv}...")
        write_csv(args.csv, updated_entries)

    # Stop here if dry run
    if args.dry_run:
        print("\nDry run complete.")
        return 0

    # Prepare processing arguments
    process_args = []
    for i, (orig, anon, split) in enumerate(updated_entries):
        process_args.append({
            'idx': i,
            'total': len(updated_entries),
            'orig_expdir': orig,
            'anon_expdir': anon,
            'split_frame': split,
            'movie_path': os.path.join(orig, 'movie.ufmf'),
            'do_video': do_video,
            'do_trx': do_trx,
            'do_track': do_track,
            'do_copy': do_copy,
            'do_registration': do_registration,
            'do_protocol': do_protocol,
            'do_metadata': do_metadata,
            'do_apt': do_apt,
            'force': args.force,
        })

    # Process experiments
    tasks = []
    if do_video:
        tasks.append('video')
    if do_trx:
        tasks.append('trx')
    if do_track:
        tasks.append('track')
    if do_copy:
        tasks.append('copy')
    if do_registration:
        tasks.append('registration')
    if do_protocol:
        tasks.append('protocol')
    if do_metadata:
        tasks.append('metadata')
    if do_apt:
        tasks.append('apt')

    print(f"\nProcessing {len(process_args)} experiments ({', '.join(tasks)}) "
          f"using {args.workers} workers{'(force)' if args.force else ''}...")

    if args.workers <= 1:
        results = []
        for pa in process_args:
            res = process_experiment(pa)
            results.append(res)
    else:
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_experiment, process_args)

    # Summary
    successes = sum(1 for r in results if r['success'])
    skipped = sum(1 for r in results if r.get('skipped', False))
    failures = [r for r in results if not r['success']]

    print(f"\n{'='*60}")
    print(f"Complete: {successes}/{len(results)} succeeded "
          f"({skipped} skipped, {successes-skipped} processed)")

    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f['exp_name']}: {f.get('error', 'unknown')}")

    return 0 if not failures else 1


if __name__ == "__main__":
    exit(main())
