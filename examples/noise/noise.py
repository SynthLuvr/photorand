from typing import Final

import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rawpy


def extract_noise_map(
	raw_path: str,
	output_path: str,
	patch_size: int = 256
) -> None:
	"""
	Isolates and amplifies the stochastic noise floor of a RAW sensor frame.

	This function bypasses standard image processing to visualize the
	physical entropy (shot noise and thermal jitter) captured by the
	hardware's pixel wells.
	"""
	with rawpy.imread(raw_path) as raw:
		# 1. Access the unprocessed Bayer pattern (the 'raw' values)
		# We cast to float32 to perform high-precision amplification
		# of the Least Significant Bits (LSBs).
		raw_data: npt.NDArray[np.float32] = raw.raw_image.astype(np.float32)

		# 2. Grab a patch from the center to avoid lens-edge artifacts (vignetting)
		h: int
		w: int
		h, w = raw_data.shape

		start_h: Final[int] = h // 2
		start_w: Final[int] = w // 2

		# We sample a localized grid to demonstrate the lack of spatial correlation
		# in raw sensor noise.
		patch: npt.NDArray[np.float32] = raw_data[
			start_h : start_h + patch_size,
			start_w : start_w + patch_size
		]

		# 3. Amplify the physical chaos
		# Subtract the minimum and scale to 0-255. This makes the invisible
		# quantum fluctuations visible to the human eye.
		patch_min: float = float(np.min(patch))
		patch_max: float = float(np.max(patch))

		if patch_max > patch_min:
			normalized_patch: npt.NDArray[np.float32] = (
				(patch - patch_min) / (patch_max - patch_min) * 255
			)
		else:
			normalized_patch = np.zeros_like(patch)

		# 4. Save as a grayscale 'Reality Map'
		# The resulting PNG is a direct visualization of the entropy source
		# used to seed the ChaCha20 expander.
		imageio.imsave(output_path, normalized_patch.astype(np.uint8))
		print(f"Physical entropy map saved to: {output_path}")

# Usage: Visualizing the 'Seed' of your TRNG
extract_noise_map('src/tests/data/DSC03088.ARW', 'examples/noise/entropy_visualized.png')
extract_noise_map('src/tests/data/DSC03089.ARW', 'examples/noise/entropy_visualized_2.png')

def create_spectral_entropy_diff(
	path_a: str,
	path_b: str,
	output_path: str
) -> None:
	"""
	Generates a multi-color spectral heatmap of the delta between two
	entropy captures.

	This visualizes the 'Stochastic Signature'—the unique, non-repeating
	physical noise floor of the camera sensor.
	"""
	# 1. Load the two noise maps
	map_a: npt.NDArray[np.uint8] = iio.imread(path_a)
	map_b: npt.NDArray[np.uint8] = iio.imread(path_b)

	# 2. Calculate the absolute difference in 16-bit
	diff: npt.NDArray[np.int16] = np.abs(map_a.astype(np.int16) - map_b.astype(np.int16))

	# 3. Amplify the delta
	d_min: Final[float] = float(np.min(diff))
	d_max: Final[float] = float(np.max(diff))

	if d_max > d_min:
		normalized_diff: npt.NDArray[np.float32] = (diff - d_min) / (d_max - d_min)
	else:
		normalized_diff = np.zeros_like(diff, dtype=np.float32)

	# 4. Apply a professional colormap (Magma or Viridis)
	# 'magma' goes from black -> purple -> orange -> white
	cm = plt.get_cmap('magma')
	spectral_map: npt.NDArray[np.float32] = cm(normalized_diff)

	# 5. Convert from float (0-1) to uint8 (0-255) for PNG saving
	# The cmap returns RGBA, so we take the first 3 channels (RGB)
	rgb_output: npt.NDArray[np.uint8] = (spectral_map[:, :, :3] * 255).astype(np.uint8)

	# 6. Save the final Heatmap
	iio.imwrite(output_path, rgb_output)
	print(f"Spectral entropy heatmap saved to: {output_path}")

# Usage
create_spectral_entropy_diff(
	'examples/noise/entropy_visualized.png',
	'examples/noise/entropy_visualized_2.png',
	'examples/noise/entropy_heatmap.png'
)