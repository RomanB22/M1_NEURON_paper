import numpy as np
from scipy import stats

def get_tonetimes(file):
    mat = np.load(file, allow_pickle=True)
    tasksr = mat['sampling rate']

    tone = (abs(stats.zscore(np.diff(mat['tone']))))
    tone = tone - min(tone)
    tonethresh = 1

    onsettimes = np.array([])
    offsettimes = np.array([0])

    if tone[0] >= tonethresh:
        onsettimes = np.array([0])
        
    crossings_up = (tone[:-1] <= tonethresh) & (tone[1:] > tonethresh)
    crossings_down = (tone[:-1] >= tonethresh) & (tone[1:] < tonethresh)
    
    onsettimes = np.concatenate(
        (onsettimes, (np.where(crossings_up)[0] + 1) / tasksr))
    offsettimes = np.concatenate(
        (offsettimes, (np.where(crossings_down)[0] + 2) / tasksr))
    
    if tone[-1] >= tonethresh:
        offsettimes = np.concatenate(
            (offsettimes, [(len(tone) - 1) / tasksr]))

    toneonsettimes = np.array([])
    toneoffsettimes = np.array([])
    for t in onsettimes:
        #pofft = offsettimes[offsettimes < t]
        pofft = offsettimes[offsettimes <= t]
        if len(pofft) > 0 and t - pofft[-1] > 0.1:
            toneonsettimes = np.append(toneonsettimes, t)
            toneoffsettimes = np.append(toneoffsettimes, pofft[-1])
    toneoffsettimes = np.append(toneoffsettimes, offsettimes[-1])
    toneoffsettimes = toneoffsettimes[toneoffsettimes > toneonsettimes[0]]
    toneonsettimes = toneonsettimes[toneonsettimes < toneoffsettimes[-1]]

    c = 0
    incpos = np.array([])
    for t in toneonsettimes:
        stot = toneoffsettimes[toneoffsettimes > t][0]
        if stot - t > 0.5:
            incpos = np.append(incpos, int(c))
        c = c + 1
    toneonsettimes = toneonsettimes[incpos.astype(int)]
    toneoffsettimes = toneoffsettimes[incpos.astype(int)]
    
    return toneonsettimes, toneoffsettimes

def get_locktimes(file):
    mat = np.load(file, allow_pickle=True)
    tasksr = mat['sampling rate']

    lock = mat['joysticklock']
    lockthresh = max(lock) / 2

    locktimes = np.array([])
    unlocktimes = np.array([])
    for n in range(0, len(lock) - 1):
        if lock[n] <= lockthresh and lock[n + 1] > lockthresh:
            locktimes = np.append(locktimes, n / tasksr)
        if lock[n] > lockthresh and lock[n + 1] <= lockthresh:
            unlocktimes = np.append(unlocktimes, n / tasksr)

    return locktimes, unlocktimes

def get_xy(file):
    mat = np.load(file, allow_pickle=True)
    return mat['x'], mat['y']

def get_rewardtimes(file):
    mat = np.load(file, allow_pickle=True)
    reward = mat['reward']
    tasksr = mat['sampling rate']
    rewardthresh = max(reward)/2
    rewardtimes = np.array([])
    for n in range(0, len(reward)-1):
        if reward[n]<=rewardthresh and reward[n+1]>rewardthresh:
            rewardtimes = np.append(rewardtimes, n/tasksr)

    return rewardtimes
