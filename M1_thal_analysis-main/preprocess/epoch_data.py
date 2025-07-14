from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import scipy.signal as sig
import xarray as xr
from tqdm import tqdm
import gc

# Column of event table used to align trials
event_time_column = 'tone_onset'

# Columns with event metadata (will be attached to trials)
event_meta_columns = ['unlock', 'tone_onset', 'tone_offset']

# Beginning and end of epoch relative to its central event
trial_win = None # will be computed based on task event
# trial_win = (-4.0, 2.0) # or it can be set explicitly

margin = 0.25 # margin around trial window to analyze (in s)

# Sampling rates
fs_lfp = 500
fs_joystick = 1000

sess_used = None
#sess_used = ['230509_2775_1454VAL']


def resample_and_resize(x, k, n=None):
    # Resample by the factor of 1/k
    if x.ndim != 1:
        raise ValueError('Input should have one dimension')
    x = x.astype(np.float64)
    if k != 1:
        if k.is_integer():
            x = sig.decimate(x, int(k))
        else:
            print('Non-integer rescaling factor')
            x = sig.resample(x, len(x) / k)
    if n is not None:
        # Make the size equal to n (truncate or pad win nan's)
        nx = len(x)
        if n <= nx:
            x = x[:n]
        else:
            x = np.pad(x, (0, n - nx), mode='constant', constant_values=np.nan)
    return x


def _epoch_data(X, fs, ev_times, dims_in, coords_in, trial_win, 
               ev_meta=None, time_dim=1):
    """ Create epoched xr.DataArray from ndarray X. """
    
    dt = 1 / fs
    nsamples = X.shape[time_dim]
    t = np.arange(nsamples) * dt
    
    # Associate an xarray with the data matrix
    X = xr.DataArray(X, dims=dims_in, coords=coords_in)
    
    # Calculate relative epoch time samples
    t_trial = np.arange(trial_win[0], trial_win[1] + dt, dt)
    
    # Allocate an xarray for epoched data
    ntrials = len(ev_times)
    dims_ep = ('trial', *dims_in)
    coords_ep = {'trial': np.arange(ntrials)} | coords_in
    time_dim = coords_in['time'][0]  # dimension associated with time coordinate
    coords_ep['time'] = (time_dim, t_trial)
    coords_ep['sample'] = (time_dim, np.arange(len(t_trial)))
    Xep = xr.DataArray(np.nan, dims=dims_ep, coords=coords_ep)

    Xep.attrs['fs'] = fs

    # Epoch the data
    for n, t0 in enumerate(ev_times):
        t0 = t[np.argmin(np.abs(t - t0))]
        t_win = np.round((t0 + trial_win) / dt)
        Xep.loc[dict(trial=n)].values[...] = X.sel(sample=slice(*t_win)).values
    
    # Add metadata for each trial from event table
    if ev_meta is not None:
        for col in ev_meta.columns:
            Xep.coords[col] = ('trial', ev_meta[col])
    return Xep


def load_lfp(dir, region):
    assert region in ['m1', 'th'], f"Invalid region {region}. Should be either 'm1' or 'th'."

    if isinstance(dir, str):
        dir = Path(dir)

    fpath_lfp = dir / f'{region}_LFPs.pkl'
    with open(fpath_lfp, 'rb') as fid:
        lfp_data = pkl.load(fid)
    return lfp_data


def epoch_data(dirpath_sess_in, region, task_events, trial_win):
    assert region in ['m1', 'th'], f"Invalid region {region}. Should be either 'm1' or 'th'."

    if trial_win is None:
        from preprocess.create_task_events import infer_trial_window
        trial_win = infer_trial_window(task_events, margin=margin)

    # Load joystick trajectories
    fpath_sig = dirpath_sess_in / 'tasksignals.pkl'
    with open(fpath_sig, 'rb') as fid:
        sig_data = pkl.load(fid)

    # Load LFPs
    print('Load...')
    try:
        lfp_data = load_lfp(dirpath_sess_in, region)
    except FileNotFoundError as e:
        print(f"No LFP data for {region} in {dirpath_sess_in}.\nProcessing trajectories only..\n")
        lfp_data = None
    
    if lfp_data:
        # LFP time bins
        if lfp_data['sampling rate'] != fs_lfp:
            raise ValueError('Unexpected LFP sampling rate')
        
        nsamples = lfp_data[f'{region}_lfps'].shape[0]
    else:
        nsamples = None
    
    # Resample and resise trajectories to match LFPs
    k = sig_data['sampling rate'] / fs_lfp
    x_sigs = resample_and_resize(sig_data['x'], k, nsamples)
    y_sigs = resample_and_resize(sig_data['y'], k, nsamples)
    xy_sigs = np.vstack((x_sigs, y_sigs))

    if nsamples is None:
        nsamples = xy_sigs.shape[1]

    dt = 1 / fs_lfp
    t = np.arange(nsamples) * dt

    # Discard trials that don't fit into the data
    t_ev = task_events[event_time_column]
    mask = ((t_ev + trial_win[0]) > t[0]) & ((t_ev + trial_win[1]) < t[-1])
    task_events = task_events.loc[mask.values, :]
    t_ev = task_events[event_time_column]

    # Coordinates
    time_coords = {'time': ('sample', t),
                'sample': ('sample', np.arange(nsamples))}
    axis_coords = {'axis': ('axis', ['x', 'y'])}
    
    # Epoch LFPs and joystick signals
    print('Epoch...')

    Xep_joy = _epoch_data(xy_sigs, fs_lfp, t_ev,
                ('axis', 'sample'), axis_coords | time_coords,
                trial_win, task_events[event_meta_columns], time_dim=1)
    if lfp_data:
        chan_depths = lfp_data[f'{region}_lfp_depths_from_surface']
        nchans = len(chan_depths)
        chan_coords = {'chan': ('chan', np.arange(nchans)),
                    'depth': ('chan', chan_depths)}
    
        Xep_lfp = _epoch_data(lfp_data[f'{region}_lfps'].T, fs_lfp, t_ev,
                    ('chan', 'sample'), chan_coords | time_coords,
                    trial_win, task_events[event_meta_columns], time_dim=1)

        Xep = xr.Dataset({'LFP': Xep_lfp, 'Joystick': Xep_joy})
    else:
        Xep = xr.Dataset({'Joystick': Xep_joy})

    # Make 'time' the indexed coordinate instead of 'sample'
    Xep = Xep.swap_dims({'sample': 'time'})

    Xep.attrs['central_event'] = event_time_column

    return Xep

def process_spikes(folder, region, task_events, drop_intertrial_spikes=True):
    assert region in ['m1', 'th'], f"Invalid region {region}. Should be either 'm1' or 'th'."

    from preprocess.create_task_events import infer_trial_window
    trial_window = infer_trial_window(task_events, margin=margin)

    import pickle as pkl
    import pandas as pd

    # spikes in M1
    with open (folder / f'{region}spts.pkl', 'rb') as f:
        spikes_raw = pkl.load(f)

    import time
    st = time.time()

    depths = spikes_raw.pop('depths_from_surface', None)
    wave_widths = spikes_raw.pop('spike_widths', None)

    spikes = []
    cell_gids = []
    for cell, cell_spikes in spikes_raw.items():
        # print(f'{cell} --- {cell_spikes.shape}')
        assert cell_spikes.shape[1] == 1, f"Unexpected spikes array shape {cell_spikes.shape} for cell {cell}. Second dimension should be 1?"
        cell_spikes = cell_spikes.flatten()

        cell_gids.extend([cell] *  len(cell_spikes))
        spikes.extend(cell_spikes)

    data = {
        'cell_id': cell_gids,
        'spike_time': spikes,
        'trial': -1,
        'stage': -1,
    }
    df = pd.DataFrame(data)

    # iterate over trials and assign trials and task phases to spikes
    for index, e in task_events.iterrows():
        sp_times = df['spike_time']
        # assign trial
        trial_start, trial_end = e[event_time_column] + trial_window

        # print(f'{trial_start} -- {trial_end}')
        # print(f'{e.unlock} -- {e.tone_offset}')

        mask = (sp_times >= trial_start) & (sp_times < trial_end)
        df.loc[mask, 'trial'] = index

        # assign task phase
        this_trial_spikes = df.loc[mask, 'spike_time']
        # 'before', 'pre', 'tone', 'after'
        task_phases = np.array([trial_start, e.unlock, e.tone_onset, e.tone_offset, trial_end])
        ph = np.digitize(this_trial_spikes, task_phases, right=False)
        df.loc[mask, 'stage'] = ph-1

    df.attrs['task_stage_names'] = ['before', 'unlock', 'tone', 'after']
    df.attrs['trial_window'] = trial_window
    df.attrs['cell_depths'] = depths
    df.attrs['spike_widths'] = wave_widths
    df.attrs['central_event'] = event_time_column

    if drop_intertrial_spikes:
        # drop spikes not attributed to any of trials
        df = df.loc[df['trial'] >= 0]

    # df.sort_values('spike_time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # compress int columns that are apparently not containing large values
    df.cell_id = df.cell_id.astype('uint16')
    df.trial = df.trial.astype('uint16')
    df.stage = df.stage.astype('uint8')

    print(f'Spikes processed in {(time.time() - st)*1e3} ms (excluding file loading time)')

    return df

def save_epoched(path, region, spikes=None, data=None):

    # save epoched data (LFP, joystick)
    if data is not None:
        fpath_out = path / f'{region}_lfps_epoched.nc'
        data.to_netcdf(fpath_out, engine='h5netcdf', invalid_netcdf=True)

    if spikes is not None:
    # save epoched spikes
        spikes.to_csv(path / f'{region}_epoched_spikes.csv', index=False)

        # save attrs
        import json
        # Convert ndarray attributes to lists for JSON serialization
        attrs = {key: value.tolist() 
                if isinstance(value, np.ndarray) 
                else value for key, value in spikes.attrs.items()}
        with open(path / f'{region}_epoched_spikes_attrs.json', 'w') as jsonfile:
            json.dump(attrs, jsonfile)


def load_epoched_data(path, region):
    return xr.open_dataset(path / f'{region}_lfps_epoched.nc')


def load_epoched_spikes(path, region):
    import json
    spikes = pd.read_csv(path / f'{region}_epoched_spikes.csv')
    with open(path / f'{region}_epoched_spikes_attrs.json', 'r') as jsonfile:
        attrs = json.load(jsonfile)

    # Convert lists back to numpy arrays only if they are lists
    for key, value in attrs.items():
        if isinstance(value, list):
            spikes.attrs[key] = np.array(value)
        else:
            spikes.attrs[key] = value

    return spikes


if __name__ == '__main__':
    from preprocess.create_task_events import load_task_events

    # Root folder containing sessions as subfolders
    dirpath_data = Path('./data')
    
    # List subfolders
    folders = [f.name for f in dirpath_data.iterdir() if f.is_dir()]
    
    for folder in tqdm(folders):
        try:
            task_events = load_task_events(dirpath_data / folder)
        except FileNotFoundError as e:
            print(f"Error parsing {folder}: {e}.\nSkipping to the next one..\n")
            continue

        path = dirpath_data / folder
        for region in ['m1', 'th']:
            try:
                data = epoch_data(path, region, task_events, trial_win)
                spikes = process_spikes(path, region, task_events)

                save_epoched(path, region, spikes, data)

            except FileNotFoundError as e:
                print(f"Error parsing {region} data in {folder}: {e}.\nSkipping to the next one..\n")

        gc.collect()