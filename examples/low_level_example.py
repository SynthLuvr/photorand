import os

from src.low_level import sample_entropy_grid
from src.low_level.csprng import expand_entropy_chacha20
from src.low_level.hash import hash_entropy_pool
from src.low_level.ingest import ingest_raw_image


def main():
	# 1. Path to a sample RAW image (using project test data)
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	image_path = os.path.join(project_root, "src", "tests", "data", "DSC02111.ARW")

	if not os.path.exists(image_path):
		print(f"Error: Could not find sample image at {image_path}")
		return

	print("--- Low Level Example ---")
	print(f"Using image: {image_path}\n")

	# 2. Ingest: Read pure sensor data
	print("[1/4] Ingesting RAW image sensor data...")
	raw_data = ingest_raw_image(image_path)
	print(f"      - Shape: {raw_data.shape}")
	print(f"      - Type: {raw_data.dtype}")

	# 3. Sample: Harvest noisy LSBs from a grid
	print("\n[2/4] Sampling noisy bits (LSB extraction)...")
	entropy_pool = sample_entropy_grid(raw_data)
	print(f"      - Harvested: {len(entropy_pool)} bytes of raw entropy")

	# 4. Hash: Compress/Blend into a cryptographically secure seed
	print("\n[3/4] Blending entropy with SHA3-512...")
	trng_seed = hash_entropy_pool(entropy_pool)
	print(f"      - Generated Seed (hex): {trng_seed.hex()}")
	print(f"      - Seed length: {len(trng_seed)} bytes")

	# 5. CSPRNG: Expand physical seed into arbitrary random stream
	print("\n[4/4] Expanding seed using ChaCha20 CSPRNG...")
	random_bytes = expand_entropy_chacha20(trng_seed, 32)
	print(f"      - 32 Random Bytes (hex): {random_bytes.hex()}")

if __name__ == "__main__":
	main()
