from PIL import Image
import os

bad_files = []
base_dir = "C:/Users/nagas/deepfake-detection/data/processed"

# Only check files inside subdirectories
for subdir in ['real', 'fake']:
    folder_path = os.path.join(base_dir, subdir)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verifies image integrity
        except Exception as e:
            bad_files.append(file_path)
            print(f"Bad file: {file_path} — {e}")

print(f"\nTotal bad files: {len(bad_files)}")
