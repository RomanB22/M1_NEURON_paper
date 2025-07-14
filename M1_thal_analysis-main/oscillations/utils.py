from math import fmod, pi, cos, sin
import numpy as np
import scipy
import matplotlib.pyplot as plt

def AccomodateSpectra(f_signal,S_signal,f_norm,S_norm,Ntrials):
    if len(f_signal) > len(f_norm):
        # Normalization signal goes to sampling as in signal
        x = f_norm
        y = S_norm
        xs = f_signal
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)

        f_signal_norm = xs
        # compute mean for normalization
        S_norm_mean = np.mean(ys,axis=0)
        S_signal_norm1 = np.array([S_signal[ntrial] / S_norm_mean for ntrial in range(Ntrials)])
        S_signal_norm2 = np.array([(S_signal[ntrial]-S_norm_mean)/S_norm_mean for ntrial in range(Ntrials)])
    else:
        # Signals become sampled as in the normalization
        x = f_signal
        y = S_signal
        xs = f_norm
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)

        f_signal_norm = xs
        # compute mean for normalization
        S_norm_mean = np.mean(S_norm,axis=0)
        S_signal_norm1 = np.array([ys[ntrial] / S_norm_mean for ntrial in range(Ntrials)])
        S_signal_norm2 = np.array([(ys[ntrial]-S_norm_mean)/S_norm_mean for ntrial in range(Ntrials)])

    return f_signal_norm, S_signal_norm1, S_signal_norm2

def AccomodateSpectra_fft(f_signal,S_signal,Phase_signal,f_norm,S_norm,Phase_norm,Ntrials):
    if len(f_signal) > len(f_norm):
        # Normalization signal goes to sampling as in signal
        x = f_norm
        y = S_norm
        xs = f_signal
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)

        f_signal_norm = xs
        # compute mean for normalization
        S_norm_mean = np.mean(ys,axis=0)
        S_signal_norm1 = np.array([S_signal[ntrial] / S_norm_mean for ntrial in range(Ntrials)])
        S_signal_norm2 = np.array([(S_signal[ntrial]-S_norm_mean)/S_norm_mean for ntrial in range(Ntrials)])

        x = f_norm
        y = Phase_norm
        xs = f_signal
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)
    
        Phase_norm_mean = np.mean(ys,axis=0)
        Phase_signal_norm = np.array([Phase_signal[ntrial] - Phase_norm_mean for ntrial in range(Ntrials)])

    else:
        # Signals become sampled as in the normalization
        x = f_signal
        y = S_signal
        xs = f_norm
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)

        f_signal_norm = xs
        # compute mean for normalization
        S_norm_mean = np.mean(S_norm,axis=0)
        S_signal_norm1 = np.array([ys[ntrial] / S_norm_mean for ntrial in range(Ntrials)])
        S_signal_norm2 = np.array([(ys[ntrial]-S_norm_mean)/S_norm_mean for ntrial in range(Ntrials)])

        x = f_signal
        y = Phase_signal
        xs = f_norm
        cs = scipy.interpolate.interp1d(x, y, kind='linear', axis=1, fill_value="extrapolate")
        ys = cs(xs)

        Phase_norm_mean = np.mean(Phase_norm,axis=0)
        Phase_signal_norm = np.array([ys[ntrial] - Phase_norm_mean for ntrial in range(Ntrials)])

    return f_signal_norm, S_signal_norm1, S_signal_norm2, Phase_signal_norm


def circular_hist(ax, x, bins=32, density=True, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def circ_r(alpha):
    import numpy as np
    
    r = np.sum([np.exp(1j*val) for val in alpha]);
    
    theta = np.arctan2(np.imag(r),np.real(r)) * 180/pi
    r = np.abs(r)/len(alpha)

    return r, theta


def circ_rtest(alpha):
    # Computes Rayleigh test for non-uniformity of circular data.
    # H0: the population is uniformly distributed around the circle
    # HA: the populatoin is not distributed uniformly around the circle
    # Assumption: the distribution has maximally one mode and the data is sampled from a von Mises distribution!
    # Input:
    #   alpha	sample of angles in radians
    # Output:
    #   pval1,2  p-value of Rayleigh's test
    #   z     value of the z-statistic

    r,theta =  circ_r(alpha)
    n = len(alpha)

    # compute Rayleigh's R (equ. 27.1)
    R = n*r

    # compute Rayleigh's z (equ. 27.2)
    z = R*R / n

    # compute p value using approxation in Zar, p. 617
    pval1 = np.exp(np.sqrt(1+4*n+4*(n*n-R*R))-(1+2*n));
    pval2 = np.exp(-z) * (1 + (2*z - z*z) / (4*n) - (24*z - 132*z*z + 76*z*z*z - 9*z*z*z*z) / (288*n*n))

    return r, theta, pval1, pval2, z

def pairwise_phase_consist(alpha):
    """ Compute Pairwise Phase Consistency (PPC) according to Vinck et al. 2010

    Args:
        alpha (list, nd.arraay): phases in radians

    Returns:
        float: sample estimate of pairwise phase consistency
    """
    n = len(alpha)
    if n < 2:
        # print('PPC not possible since num spikes < 2')
        return -2

    if n < 14: # it was found empirically that for N<14 it's faster to use straightforward for-loops
        sum = 0
        for j in range(n-1):
            for k in range(j+1, n):
                # Eq. 14 from Vinck et al. 2010
                sum += cos(alpha[j])*cos(alpha[k]) + sin(alpha[j])*sin(alpha[k])
    else:
        # for larger data, build all possible pairs:
        phi = np.tile(alpha, (n, 1))
        psi = phi.T
        # unique pairs - create upper triangular mask excluding main diagonal
        mask = np.triu(np.ones((n, n)), 1)

        # compute ppc
        all = mask * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        sum = all.sum()
    num_pairs = n*(n-1)/2
    return sum / num_pairs

def surrogate_of(original, fs):
    freqs = scipy.fft.fftfreq(len(original), 1/fs)
    trans = scipy.fft.fft(original)

    magnitudes = np.abs(trans)
    phases = np.angle(trans)

    np.random.shuffle(phases)

    trans_shuffled = magnitudes * np.exp(1j * phases)

    # Make sure the signal is Hermitian symmetric (thanks, ChatGPT), so that ifft will be real
    trans_shuffled[1:len(original)//2] = np.conj(trans_shuffled[len(original)//2+1:][::-1])

    surrogate = scipy.fft.ifft(trans_shuffled).real

    # keep original std
    ratio = original.std() / surrogate.std()
    surrogate *= ratio
    return surrogate

def freq_reponse_test():
    from scipy.signal import freqz

    fs=500
    # filtOrder = 3
    nyquist = fs/2.0
    # filtFreq = [12,25]

    # Plot the frequency response for a few different orders.
    
    plt.figure(1)
    plt.clf()

    start = 2
    step = 4.5
    windwidth = 4

    ys = []
    freqs = [[start+i*step, start+i*step + windwidth] for i in range(14)]
    for i, filtFreq in enumerate(freqs):
        Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]

        for filtOrder in [4]:
            b, a = scipy.signal.butter(filtOrder, Wn, btype='bandpass')
            # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            n = 2000
            w, h = freqz(b, a, worN=n)

            col = plt.get_cmap('tab10').colors[i%10]
            x = (fs * 0.5 / np.pi) * w[:int(n/3)]
            y = abs(h[:int(n/3)])
            ys.append(y)
            plt.plot(x, y, label=f"{filtFreq[0]}-{filtFreq[1]} Hz", c=col)
        plt.plot([filtFreq[0], filtFreq[0]], [0,1], '--', c=col)
        plt.plot([filtFreq[1], filtFreq[1]], [0,1], '--', c=col)
    avgGain = np.array(ys).mean(axis=0)
    plt.plot(x, avgGain)

    # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    # plt.grid(True)
    plt.locator_params(axis='x', nbins=10)
    plt.legend(loc='best')
    print()

