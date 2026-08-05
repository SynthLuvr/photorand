import os

from src import PhotoRandEngine, PhotoRandSeed


def main():
	# Path to a sample RAW image (using project test data)
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	image_path = os.path.join(project_root, "src", "tests", "data", "DSC02111.ARW")

	if not os.path.exists(image_path):
		print(f"Error: Could not find sample image at {image_path}")
		return

	print("--- High Level Example ---")
	print(f"Using image: {image_path}\n")

	# 1. Using PhotoRandSeed (The TRNG interface)
	print("Step 1: Creating PhotoRandSeed (TRNG)...")
	pr_seed = PhotoRandSeed(image_path)
	print(f"      - Hex Seed: {pr_seed.to_hex_string()[:64]}...")
	print(f"      - Random Integer [1-100]: {pr_seed.to_int_range(1, 100)}")

	# 2. Using PhotoRandEngine (The CSPRNG engine)
	print("\nStep 2: Initializing PhotoRandEngine (CSPRNG)...")
	# You can pass the seed object or just the image path
	engine = PhotoRandEngine(pr_seed)

	print("\nStep 3: Generating various random types...")
	print(f"      - Random Bytes (16): {engine.next_bytes(16).hex()}")
	print(f"      - Random Integer: {engine.next_int()}")
	print(f"      - Random Range [1000, 2000]: {engine.next_int_range(1000, 2000)}")
	print(f"      - Random Alphanumeric: {engine.next_string(20, charset='alphanumeric')}")
	print(f"      - Random Boolean: {engine.next_bool()}")

	# 3. Deterministic Mode
	print("\nStep 4: Deterministic Mode (Reproducible sequence)...")
	engine_det1 = PhotoRandEngine(image_path, salt=False)
	seq1 = [engine_det1.next_int_range(0, 1000) for _ in range(5)]

	engine_det2 = PhotoRandEngine(image_path, salt=False)
	seq2 = [engine_det2.next_int_range(0, 1000) for _ in range(5)]

	print(f"      - Sequence 1: {seq1}")
	print(f"      - Sequence 2: {seq2}")
	print(f"      - Matches? {'Yes' if seq1 == seq2 else 'No'}")

if __name__ == "__main__":
	main()
