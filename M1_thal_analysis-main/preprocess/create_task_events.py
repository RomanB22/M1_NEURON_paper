import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.task_sigs import get_tonetimes, get_locktimes, get_rewardtimes, get_xy


def first_larger_than(x, th):
    ind = np.argwhere(x > th)
    if len(ind):
        return x[ind[0, 0]]
    else:
        return np.nan
    
def last_smaller_than(x, th):
    ind = np.argwhere(x < th)
    if len(ind):
        return x[ind[-1, 0]]
    else:
        return np.nan

def signals_to_events(fpath_sigs):
    toneonsettimes, toneoffsettimes = get_tonetimes(fpath_sigs)
    locktimes, unlocktimes = get_locktimes(fpath_sigs)
    rewardtimes = get_rewardtimes(fpath_sigs)
    x,y = get_xy(fpath_sigs)    
    events = {}
    events['x'] = x
    events['y'] = y
    events['toneonsettimes'] = toneonsettimes
    events['toneoffsettimes'] = toneoffsettimes
    events['locktimes'] = locktimes
    events['unlocktimes'] = unlocktimes
    events['rewardtimes'] = rewardtimes
    return events

def create_trial_ev_table(ev_data):
    # Group events by trials into a table
    trials = []
    for n, t_on in enumerate(ev_data['toneonsettimes']):
        t_off = ev_data['toneoffsettimes'][n]
        t_unlock = last_smaller_than(ev_data['unlocktimes'], t_on)
        t_lock = first_larger_than(ev_data['locktimes'], t_on)
        t_reward = first_larger_than(ev_data['rewardtimes'], t_on)
        t_on_next = first_larger_than(ev_data['toneonsettimes'], t_on)
        if t_on_next < t_reward:
            t_reward = np.nan
        trial = {'tone_onset': t_on, 'tone_offset': t_off,
                 'unlock': t_unlock, 'lock': t_lock, 'reward': t_reward}
        trials.append(trial)
    ev_tbl = pd.DataFrame(trials)
    
    # Calculate additional info about each trial
    ev_tbl['tone_len'] = ev_tbl['tone_offset'] - ev_tbl['tone_onset']
    ev_tbl['unlock_len'] = ev_tbl['tone_onset'] - ev_tbl['unlock']
    ev_tbl['ISI'] = np.nan * np.ones(len(ev_tbl))
    t_on = ev_tbl['tone_onset'].values
    ev_tbl.loc[1:, 'ISI'] = t_on[1:] - t_on[:-1]
    
    # Discard bad trials
    ev_tbl = ev_tbl[~np.isnan(ev_tbl['reward'])]
    return ev_tbl   


def infer_trial_window(events, margin=0):

    st = -(events.tone_onset - events.unlock).max() - margin
    end = (events.tone_offset - events.tone_onset).max() + margin
    return (np.round(st, 2), np.round(end, 2))


def save_task_events(dir, events):
    if isinstance(dir, str):
        dir = Path(dir)

    fpath_ev = dir / 'task_events.csv'
    events.to_csv(fpath_ev)


def load_task_events(dir):

    if isinstance(dir, str):
        dir = Path(dir)

    fpath_ev = dir / 'task_events.csv'
    events = pd.read_csv(fpath_ev)

    return events


if __name__ == '__main__':

    # Root folder containing sessions as subfolders
    dirpath_data = Path('./data')
    
    # List subfolders
    folders = [f.name for f in dirpath_data.iterdir() if f.is_dir()]
    
    for folder in tqdm(folders):
        try:
            print(f"Processing {folder}...")
            # Path to the file with signals
            dirpath_sess = dirpath_data / folder
            fpath_sigs = dirpath_sess / 'tasksignals.pkl'
            
            # Convert signals to events
            events = signals_to_events(fpath_sigs)
            
            # Group events by trials into a table, discard 'bad' trials
            ev_tbl = create_trial_ev_table(events)
            
            # Save the table
            save_task_events(dirpath_sess, ev_tbl)
        except Exception as e:
            print(f"Error parsing {folder}: {e}. \nSkipping to the next one..\n")