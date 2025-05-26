import os
from PIL import Image

def deep_clean_images(directory):
    removed = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            
            # Skip hidden/system files
            if file.startswith('.') or file == 'Thumbs.db':
                print(f"Removing system file: {path}")
                os.remove(path)
                continue
                
            # Verify image integrity
            try:
                with Image.open(path) as img:
                    img.verify()  # Check file structure
                    img.load()    # Force pixel data loading
                    
                # Additional format check for JPEG
                if file.lower().endswith(('.jpg', '.jpeg')):
                    with open(path, 'rb') as f:
                        if b'JFIF' not in f.peek(10):
                            raise Exception("Invalid JPEG header")
                            
            except Exception as e:
                print(f"Removing {path}: {str(e)}")
                os.remove(path)
                removed.append(path)
                
    return removed
