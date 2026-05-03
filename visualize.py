# ==========================================
# visualize.py
# Contains plotting functions for the simulation
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(f, E_tx):
    """
    Plots the frequency spectrum for signal.
    """
    plt.ion()
    plt.figure('Frequency Spectrum')
    Spec_Tx = np.fft.fftshift(np.abs(np.fft.fft(E_tx, axis=0)), axes=0)
    f_GHz = f / 1e9
    
    plt.plot(f_GHz, Spec_Tx[:, 0]); plt.title(' Spectrum ')
    plt.grid(True); plt.xlabel('Freq (GHz)'); plt.ylabel('Magnitude'); plt.xlim([np.min(f_GHz), np.max(f_GHz)])
    
    plt.tight_layout()
    plt.draw() # Draw without blocking

def plot_wav_spec(t, f, E_tx, E_rx):
    """
    Plots the time-domain waveforms and frequency spectra 
    for both TX and RX signals.
    """
    
    plt.ion()
    # Time-Domain Waveforms
    plt.figure('Time-Domain Waveforms', figsize=(10, 8))
    
    plt.subplot(4, 1, 1); plt.plot(t, np.abs(E_tx[:, 0])); plt.title('Tx Waveform (X-Pol)')
    plt.grid(True); plt.xlabel('Time (s)'); plt.ylabel('|E|')
    
    plt.subplot(4, 1, 2); plt.plot(t, np.abs(E_tx[:, 1])); plt.title('Tx Waveform (Y-Pol)')
    plt.grid(True); plt.xlabel('Time (s)'); plt.ylabel('|E|')
    
    plt.subplot(4, 1, 3); plt.plot(t, np.abs(E_rx[:, 0])); plt.title('Rx Waveform (X-Pol)')
    plt.grid(True); plt.xlabel('Time (s)'); plt.ylabel('|E|')
    
    plt.subplot(4, 1, 4); plt.plot(t, np.abs(E_rx[:, 1])); plt.title('Rx Waveform (Y-Pol)')
    plt.grid(True); plt.xlabel('Time (s)'); plt.ylabel('|E|')
    
    plt.tight_layout()

    # Frequency Spectra
    plt.figure('Frequency Spectra', figsize=(10, 8))
    Spec_Tx = np.fft.fftshift(np.abs(np.fft.fft(E_tx, axis=0)), axes=0)
    Spec_Rx = np.fft.fftshift(np.abs(np.fft.fft(E_rx, axis=0)), axes=0)
    f_GHz = f / 1e9
    
    plt.subplot(4, 1, 1); plt.plot(f_GHz, Spec_Tx[:, 0]); plt.title('Tx Spectrum (X-Pol)')
    plt.grid(True); plt.xlabel('Freq (GHz)'); plt.ylabel('Magnitude'); plt.xlim([np.min(f_GHz), np.max(f_GHz)])
    
    plt.subplot(4, 1, 2); plt.plot(f_GHz, Spec_Tx[:, 1]); plt.title('Tx Spectrum (Y-Pol)')
    plt.grid(True); plt.xlabel('Freq (GHz)'); plt.ylabel('Magnitude'); plt.xlim([np.min(f_GHz), np.max(f_GHz)])
    
    plt.subplot(4, 1, 3); plt.plot(f_GHz, Spec_Rx[:, 0]); plt.title('Rx Spectrum (X-Pol)')
    plt.grid(True); plt.xlabel('Freq (GHz)'); plt.ylabel('Magnitude'); plt.xlim([np.min(f_GHz), np.max(f_GHz)])
    
    plt.subplot(4, 1, 4); plt.plot(f_GHz, Spec_Rx[:, 1]); plt.title('Rx Spectrum (Y-Pol)')
    plt.grid(True); plt.xlabel('Freq (GHz)'); plt.ylabel('Magnitude'); plt.xlim([np.min(f_GHz), np.max(f_GHz)])
    
    plt.tight_layout()
    plt.draw() # Draw without blocking

def setup_constellation(Nch):
    """Initializes figures for constellation diagrams."""
    plots_per_fig = Nch
    nrows = int(np.floor(np.sqrt(plots_per_fig)))
    ncols = int(np.ceil(plots_per_fig / nrows))
    
    fig_x = plt.figure('X-Polarization Constellations', figsize=(12, 8))
    plt.suptitle('X-Polarization')
    
    fig_y = plt.figure('Y-Polarization Constellations', figsize=(12, 8))
    plt.suptitle('Y-Polarization')
    
    return fig_x, fig_y, nrows, ncols

def single_ch_constellation(fig_x, fig_y, nrows, ncols, m, y_plot):
    """Plots the constellation for a single channel."""
    plt.figure(fig_x.number)
    plt.subplot(nrows, ncols, m + 1)
    plt.scatter(np.real(y_plot[:, 0]), np.imag(y_plot[:, 0]), s=8, alpha=0.6)
    plt.axis('square')
    plt.grid(True)
    plt.title(f'Ch {m+1} X')
    
    plt.figure(fig_y.number)
    plt.subplot(nrows, ncols, m + 1)
    plt.scatter(np.real(y_plot[:, 1]), np.imag(y_plot[:, 1]), s=8, alpha=0.6)
    plt.axis('square')
    plt.grid(True)
    plt.title(f'Ch {m+1} Y')