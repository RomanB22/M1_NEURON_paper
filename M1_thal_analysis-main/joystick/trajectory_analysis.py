import numpy as np

def detect_bouts(array, threshold, min_interval):
    import matplotlib.pyplot as plt

    binarized = array > threshold

    cross_from_below = (binarized[:-1] == False) & (binarized[1:] == True)
    cross_from_above = (binarized[:-1] == True) & (binarized[1:] == False)

    crossings = np.zeros(len(array))[:-1]
    crossings[cross_from_below] = 2
    crossings[cross_from_above] = 1

    cross_times = np.where(crossings > 0)[0]

    bouts = []
    broken = False

    for iter, cross_time in enumerate(cross_times):
        if crossings[cross_time] == 1: # crossed from above
            bout_start = cross_times[iter-1]
            bouts.append((bout_start, cross_time))
        else: # crossed from below
            if len(bouts) > 0:
                interbump_interval = cross_time - cross_times[iter-1]

                if interbump_interval > min_interval:
                    # bout sequence extinct
                    broken = True
                    # print(f'broken at {interbump_interval}')
                    break
    # print(bouts)

    if len(bouts) > 0:
        start = bouts[0][0]
        end = bouts[-1][1]
        # if not broken:
        #     print(f'last at {len(array) - end}')
        return start, end
    else:
        # print('No bouts identified')
        return None

def detect_ballistic_phase(multitrial, time_pre, smoothing_window):
    
    nn = 0 # only one task for now
    n_trials = multitrial[nn]['Ntrials']

    max_length = 500
    resting_velocities = np.zeros((n_trials, time_pre))
    ballistic_velocities = np.zeros((n_trials, max_length))

    for trialId in range(n_trials):
        this_trial = multitrial[nn]['Trial'+str(trialId+1)]
        trajectory = np.array(this_trial['Trajectory'])
        velocity = np.diff(trajectory)
        absVelocity = np.sqrt(velocity[0]**2 + velocity[1]**2)

        resting_velocities[trialId] = absVelocity[:time_pre]

        kernel = np.ones(smoothing_window) / smoothing_window
        ballistic_velocities[trialId] = np.convolve(absVelocity, kernel, 'same')[time_pre: time_pre + max_length]

    
    mean = resting_velocities.mean()
    std = resting_velocities.std(axis=1).mean()
    threshold = mean + 1.5*std

    all = []
    for i, array in enumerate(ballistic_velocities):
        ballistic = detect_bouts(array, threshold, min_interval=100)
        if ballistic:
            ballistic = ballistic[0] - smoothing_window / 2, ballistic[1] + smoothing_window / 2
        all.append(ballistic)
        # print(f"{i}.\t{ballistic}")
    return all