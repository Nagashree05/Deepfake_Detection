import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits():
    # Path configuration
    real_dir = "data/processed/real"
    fake_dir = "data/processed/fake"
    splits_dir = "data/splits"
    
    # Create splits directory if not exists
    os.makedirs(splits_dir, exist_ok=True)

    # Collect real and fake image paths
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Create DataFrame
    df = pd.DataFrame({
        'filepath': real_images + fake_images,
        'label': [0]*len(real_images) + [1]*len(fake_images)
    })

    # Stratified split (70% train, 15% val, 15% test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        stratify=df['label'], 
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['label'], 
        random_state=42,
    )

    # Save splits
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # Class weights for imbalance handling (optional)
    class_weights = {
        0: len(fake_images)/len(real_images),
        1: 1.0
    }
    print(f"Class weights: {class_weights}")

if __name__ == "__main__":
    create_splits()
