import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import imageio
import re
import scipy.io.wavfile as wavfile
import json
from scipy.signal import resample, spectrogram

def load_metadata(directory):
    meta_path = os.path.join(directory, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"Loaded metadata: {meta}")
        return meta
    else:
        print("Warning: metadata.json not found, using default units.")
        return None

def make_video_from_slices(directory, output='simulation.mp4', fps=20):
    meta = load_metadata(directory)
    dx = meta["dx"] if meta else 1.0
    size_x = meta["size_x"] if meta else None
    size_y = meta["size_y"] if meta else None
    def extract_step(filename):
        match = re.search(r'slice_step_(\d+)\.csv', os.path.basename(filename))
        return int(match.group(1)) if match else -1
    files = sorted(
        glob.glob(os.path.join(directory, 'slice_step_*.csv')),
        key=extract_step
    )
    images = []
    vmin, vmax = None, None
    # First pass to determine global min/max for consistent color scale
    for fname in files:
        data = np.loadtxt(fname, delimiter=',')
        if vmin is None or np.min(data) < vmin:
            vmin = np.min(data)
        if vmax is None or np.max(data) > vmax:
            vmax = np.max(data)
    for fname in files:
        data = np.loadtxt(fname, dtype=np.float32, delimiter=',')
        norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Default
        norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
        plt.imshow(
            data,
            cmap='viridis',
            norm=norm,
            extent=[0, (size_x or data.shape[1]) * dx, 0, (size_y or data.shape[0]) * dx],
            origin='lower',
            aspect='equal'  # Ensure axes have the same scale
        )
        plt.xlabel(f"x [m] (dx={dx})")
        plt.ylabel(f"y [m] (dx={dx})")
        plt.title("2D Wavefield Slice")
        plt.axis('on')
        img_path = fname.replace('.csv', '.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        images.append(imageio.imread(img_path))
    imageio.mimsave(output, images, fps=fps)
    print(f"Video saved as {output}")

def point_samples_csv_to_wav(csv_path, output_dir, sample_rate=44100, normalize=True):
    meta = load_metadata(os.path.dirname(csv_path))
    dt = meta["dt"] if meta else None
    orig_samples = np.loadtxt(csv_path, delimiter=',')
    # Resample to 44100 Hz if dt is available and not already at 44100 Hz
    if dt:
        sim_rate = 1.0 / dt
        if abs(sim_rate - sample_rate) > 1:  # Only resample if rates differ
            num_target = int(len(orig_samples) * sample_rate / sim_rate)
            print(f"Resampling from {sim_rate:.2f} Hz to {sample_rate} Hz (samples: {len(orig_samples)} -> {num_target})")
            samples = resample(orig_samples, num_target)
    if normalize:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
    samples_int16 = np.int16(samples * 32767)
    wav_path = os.path.join(output_dir, "point_sample.wav")
    wavfile.write(wav_path, sample_rate, samples_int16)
    print(f"WAV file saved as {wav_path}")

    # Plot waveform and save as PNG
    plt.figure(figsize=(10, 4))
    t = np.arange(len(samples)) / sample_rate
    plt.plot(t, samples, linewidth=0.8)
    plt.title("Point Sample Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "point_sample.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Waveform plot saved as {png_path}")

    # Plot spectrogram and save as PNG
    plt.figure(figsize=(10, 4))
    # Use a larger NFFT for better low-frequency resolution
    fft_window = min(1024 * 32, 2**int(np.log2(len(orig_samples) / 4)))  # Use a large window for FFT
    print(f"Using FFT window size: {fft_window} (max: {len(orig_samples)})")
    plt.specgram(
        orig_samples, 
        Fs=sim_rate, 
        NFFT=fft_window,        # Larger window for higher frequency resolution
        noverlap=fft_window // 2,    # 75% overlap for smoother spectrogram
        cmap='viridis',
    )
    plt.title("Point Sample Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, 15000)
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    spec_png_path = os.path.join(output_dir, "point_sample_spectrogram.png")
    plt.savefig(spec_png_path)
    plt.close()
    print(f"Spectrogram plot saved as {spec_png_path}")

    fft_window = min(1024 * 32, len(orig_samples))  # Use a large window for FFT

    # Analyze strong frequencies using a large window
    f, t_spec, Sxx = spectrogram(
        orig_samples, fs=sim_rate, nperseg=fft_window, noverlap=fft_window/2, scaling='spectrum', mode='magnitude'
    )

    # Bin frequencies into 200 Hz windows
    freq_bins = np.arange(0, f[-1] + 100, 100)
    bin_indices = np.digitize(f, freq_bins) - 1  # bin_indices[i] gives the bin for f[i]

    print("Top 10 strongest frequencies (per 200Hz bin) for each time window (with dB values):")
    for i in range(Sxx.shape[1]):
        # For each time window, sum magnitudes in each 200Hz bin
        bin_strengths = {}
        for bin_idx in range(len(freq_bins)):
            mask = bin_indices == bin_idx
            if np.any(mask):
                bin_strengths[bin_idx] = np.sum(Sxx[mask, i])
        # Find the 10 bins with the highest summed magnitude
        top_bins = sorted(bin_strengths.items(), key=lambda x: x[1], reverse=True)[:10]
        # For each top bin, find the frequency with the max magnitude in that bin
        top_freqs = []
        for bin_idx, _ in top_bins:
            mask = bin_indices == bin_idx
            if np.any(mask):
                idx_in_bin = np.argmax(Sxx[mask, i])
                freq_in_bin = f[mask][idx_in_bin]
                mag = Sxx[mask, i][idx_in_bin]
                db_val = 20 * np.log10(mag + 1e-12)  # add epsilon to avoid log(0)
                top_freqs.append((freq_in_bin, db_val))
        # Sort by frequency for display
        top_freqs_sorted = sorted(top_freqs, key=lambda x: x[0])
        freqs_str = ", ".join([f"{freq:.1f} Hz ({db:.1f} dB)" for freq, db in top_freqs_sorted])
        print(f"t={t_spec[i]:.3f}s: {freqs_str}")

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 3:
        print("Usage: python make_media.py <snapshot_directory> <output_directory> [output_video.mp4]")
        exit(1)
    snapshot_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    # Always convert point_samples.csv to wav
    csv_path = os.path.join(snapshot_dir, "point_samples.csv")
    point_samples_csv_to_wav(csv_path, output_dir)
    
    output_video = sys.argv[3] if len(sys.argv) > 3 else "simulation.mp4"
    output_video_path = os.path.join(output_dir, output_video)
    make_video_from_slices(snapshot_dir, output_video_path)

