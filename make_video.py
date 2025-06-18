import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import imageio
import re

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python make_video.py <snapshot_directory> [output_video.mp4]")
        exit(1)
    directory = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "simulation.mp4"
    make_video_from_slices(directory, output)
