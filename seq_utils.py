import numpy as np 

import scipy
import scipy.signal 
from scipy.signal import butter, filtfilt, hilbert, sosfiltfilt

import matplotlib.pyplot as plt

def generate_seq(f_fund, std_gauss, dt_offset, total_length=None, N_neurons=None, dt=0.01,
  pad_zeros = 0.0, end_pad = 0., beg_pad = 0.):
  ''' 
  method to generate a sequence of neural activity composed of individual 
  neuron fluctuations at f_fund (frequency), convolved with a gaussian (std_gauss),
  with each neuron at an offset with respect to one another (dictated by frac_offset). And 
  enough neurons are included to make the total length of the sequence equal to total_length. 

  Input units: 
    f_fund: frequency (Hz)
    std_gauss: standard deviation of gaussian (sec)
    dt_offset: time offset b/w rows; 
    total_lenth: seconds OR
    N_neurons: integer unit
    dt: seconds
  '''
  ## N_units
  if total_length is None:
    N_units = N_neurons
  else:
    ### Now use this to tile 
    ind_offs = ( frac_offset / f_fund ) / dt # frac cycles to seconds to timesteps 
    N_units = int(np.floor((total_length / dt ) / ind_offs))

  ### Now use this to tile 
  ind_offs = ( dt_offset ) / dt # frac cycles to seconds to timesteps 
  
  ### Get the number of indices in the std_gauss ###
  if std_gauss < 0:
    ind_tot = int(((1./f_fund)/dt) + (N_units*ind_offs))

  else:
    std_ind = int(np.ceil(std_gauss/dt))
    ### Make the gaussian envelope ###
    gaussian_env = scipy.signal.gaussian(2*std_ind, std_ind, sym=True)

    ### Make the sine wave, centered at the middle of the gaussian; 
    sine_t = np.linspace(-std_gauss, std_gauss, len(gaussian_env))
  
    ### Get sine wave ###
    sine = np.cos(2*np.pi*f_fund*sine_t)
    
    ### Get unit length ###
    unitL = len(sine)
    unitL_half = int(0.5*unitL)

    ### Multiple to get a burst unit: 
    unit_mod = sine*gaussian_env 

    ### Total length of time; 
    ind_tot = int(unitL_half + N_units*ind_offs + unitL_half)

  ### Initialize the data: 
  Data = np.zeros((ind_tot, N_units))
  T_tot = ind_tot*dt

  for i_n in range(N_units):
    if std_gauss < 0:
      ts_ = np.linspace(i_n*dt_offset, T_tot + i_n*dt_offset, ind_tot)
      Data[:, i_n] = np.sin(2*np.pi*f_fund*ts_)
    
    else:
      ix_start = int(unitL_half + i_n*ind_offs)
      Data[ix_start - unitL_half:ix_start + unitL_half, i_n] = unit_mod.copy()
  
  if pad_zeros > 0:
    pz = int(pad_zeros/dt)
    pad = np.zeros((pz, N_units))
    Data = np.vstack((pad, Data, pad))
  if end_pad > 0:
    pz = int(end_pad/dt)
    pad = np.zeros((pz, N_units))
    Data = np.vstack((Data, pad))
  if beg_pad > 0:
    pz = int(beg_pad/dt)
    pad = np.zeros((pz, N_units))
    Data = np.vstack((pad, Data))
  return Data


def fit_density(frequencies):

  bandwidths = 10 ** np.linspace(-1, 1, 100)
  grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths})
  grid.fit(frequencies[:, None]);
  #Now we can find the choice of bandwidth which maximizes the score 
  #(which in this case defaults to the log-likelihood):

  bw = grid.best_params_['bandwidth']

  kde = KernelDensity(bandwidth=bw, kernel='gaussian')
  kde.fit(frequencies[:, None])

  # score_samples returns the log of the probability density
  x_f = np.linspace(-10, 10, 10000)
  logprob = kde.score_samples(x_f[:, None])

  return x_f, np.exp(logprob)

def sweep_fund_freq_and_offset(fund_freqs, dt_offset, N_neurons, gauss_std = 0.5, dt = 0.01,
                               noise_level = 0.1, pad_zeros = 0.5, nreps = 20):
    '''
    method to interate through diff frequencies/frac_offsets of underlying single
    unit modulation
    
    plots the non-negative frequency eigenvalues in a power spectra-like plot
    '''

    ### Data -- sequence example ###
    f_dat, ax_dat = plt.subplots(figsize=(15, 9), ncols = len(fund_freqs), nrows = len(dt_offset))
    
    ### Results -- eigenspec ###
    f, ax = plt.subplots(figsize=(15, 9), ncols = len(fund_freqs), nrows = len(dt_offset))
    
    for i_ff, freq in enumerate(fund_freqs):

      for i_t, dt_off in enumerate(dt_offset):
        
        ### Generate sequence data ##
        frac = dt_off*freq
        data = generate_seq(freq, gauss_std, frac, N_neurons=N_neurons, dt = dt,
                                      pad_zeros = pad_zeros)
        nT, nD = data.shape
        
        ### Trial data used later ###
        R2 = []; EVs = []; 
        for ntrl in range(nreps):
            trl_data = data + noise_level*np.random.randn(nT, nD)
            data_trl_t = trl_data[1:, :].copy()
            data_trl_tm1 = trl_data[:-1, :].copy()

            ### Estimate A ###
            ### Compute the A matrix ###
            Aest = np.linalg.lstsq(data_trl_tm1, data_trl_t, rcond=None)[0] 
            Aest = Aest.T
            
            ### Get non-negative angle EVs; 
            ev, _ = np.linalg.eig(Aest)
            an = np.angle(ev)
            EVs.append(ev)

            ##### Test Data ####
            trl_data2 = data + noise_level*np.random.randn(nT, nD)
            data_trl_t2 = trl_data2[1:, :].copy()
            data_trl_tm12 = trl_data2[:-1, :].copy()
            R2.append(lds_utils.get_population_R2(data_trl_t2, np.dot(Aest, data_trl_tm12.T).T))
        
            ### Eigenvalues; 
            lds_utils.eigenspec(Aest, axi=ax[i_t, i_ff], xlim=(-.5, 10.), ylim = (0., 0.8), skip_legend = True)
        
        ### Plot noisy data
        ax_dat[i_t, i_ff].pcolormesh(np.arange(nT+1)*dt, np.arange(nD+1), trl_data.T, cmap='binary', vmin=-1, vmax=1)
        
        ### Plot distribution of eigenvalue frequencies ###
        EVs = np.hstack((EVs))
        fr = np.angle(EVs)/(2*np.pi*dt)
        x_f, freq_density = seq_utils.fit_density(fr)
        ax2 = ax[i_t, i_ff].twinx()
        ax2.plot(x_f, freq_density, 'r--')

        ## Find peaks: 
        peaks, _ = scipy.signal.find_peaks(freq_density, width = 100, prominence=.02)
        ax2.plot(x_f[peaks], freq_density[peaks], 'r*')
        
        freq_dens_nneg = freq_density[x_f >= 0]
        x_f_nneg = x_f[x_f>= 0]
        ix_max = np.argmax(freq_dens_nneg)
        ax2.plot(x_f_nneg[ix_max], freq_dens_nneg[ix_max], 'r*')
        ax2.set_xlim([-.5, 10])
        ax2.set_ylim([0., 1.])
        
        ### Label axes 
        if i_ff == 0:
          ax[i_t, i_ff].set_ylabel('Offs %.2f sec'%dt_off)
          ax_dat[i_t, i_ff].set_ylabel('Offs %.2f sec'%dt_off)
        
        if i_t == len(dt_offset)-1:
          ax[i_t, i_ff].set_xlabel('Freq %.1f Hz'%freq)
          ax_dat[i_t, i_ff].set_xlabel('Freq %.1f Hz'%freq)

        R2 = np.hstack((R2))
        ax[i_t, i_ff].set_title('R2 = %.2f +/- %.2f' %(np.mean(R2), np.std(R2)))
    f.tight_layout()
    f_dat.tight_layout()

def simple_fft(signal, Fs):
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = 1.0 / Fs
    yf = scipy.fftpack.fft(signal)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    yf_sub = 2.0/N * np.abs(yf[:N//2])
    return xf, yf_sub

### Smoothed spikes ###
def get_smoothed_spks(spks, binsize, smoothsize):
    ### Assumes a time x neuron x trl format
    SPKS_smooth = np.zeros_like(spks)
    std = smoothsize/binsize
    window = scipy.signal.gaussian(21., std)
    window = window / float(np.sum(window))
    smooth = [];
    for trl in range(SPKS_smooth.shape[2]):
        for n in range(SPKS_smooth.shape[1]):
            SPKS_smooth[:, n, trl] = np.convolve(window, spks[:, n, trl], mode='same')
    return SPKS_smooth
