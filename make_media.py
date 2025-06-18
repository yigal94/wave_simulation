import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import imageio
import re
import scipy.io.wavfile as wavfile

def make_video_from_slices(directory, output='simulation.mp4', fps=20):
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
        # Change to symlog scale
        norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
        plt.imshow(data, cmap='viridis', norm=norm)
        plt.axis('off')
        img_path = fname.replace('.csv', '.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        images.append(imageio.imread(img_path))
    imageio.mimsave(output, images, fps=fps)
    print(f"Video saved as {output}")

def point_samples_csv_to_wav(csv_path, output_dir, sample_rate=44100, normalize=True):
    samples = np.loadtxt(csv_path, delimiter=',')
    if normalize:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
    # Convert to 16-bit PCM
    samples_int16 = np.int16(samples * 32767)
    wav_path = os.path.join(output_dir, "point_sample.wav")
    wavfile.write(wav_path, sample_rate, samples_int16)
    print(f"WAV file saved as {wav_path}")

    # Plot waveform and save as PNG
    plt.figure(figsize=(10, 4))
    plt.plot(samples, linewidth=0.8)
    plt.title("Point Sample Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "point_sample.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Waveform plot saved as {png_path}")

    # Plot spectrogram and save as PNG
    plt.figure(figsize=(10, 4))
    plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Point Sample Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    spec_png_path = os.path.join(output_dir, "point_sample_spectrogram.png")
    plt.savefig(spec_png_path)
    plt.close()
    print(f"Spectrogram plot saved as {spec_png_path}")

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 3:
        print("Usage: python make_media.py <snapshot_directory> <output_directory> [output_video.mp4]")
        exit(1)
    snapshot_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    output_video = sys.argv[3] if len(sys.argv) > 3 else "simulation.mp4"
    output_video_path = os.path.join(output_dir, output_video)
    make_video_from_slices(snapshot_dir, output_video_path)
    # Always convert point_samples.csv to wav
    csv_path = os.path.join(snapshot_dir, "point_samples.csv")
    point_samples_csv_to_wav(csv_path, output_dir)
