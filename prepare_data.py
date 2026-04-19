import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

def create_dataset_csvs(dataset_base_path):
    # The dataset has 3 folders: train, val, test
    # We will grab train and val, combine them, and re-split them 80/20.
    
    data = []
    
    # 1. Loop through all folders to collect paths and labels
    for folder in ['train', 'val', 'test']:
        folder_path = os.path.join(dataset_base_path, folder)
        
        # Get NORMAL images (Label 0)
        normal_cases = glob(os.path.join(folder_path, 'NORMAL', '*.jpeg'))
        for img_path in normal_cases:
            data.append({'image_path': img_path, 'label': 0, 'original_split': folder})
            
        # Get PNEUMONIA images (Label 1)
        pneumonia_cases = glob(os.path.join(folder_path, 'PNEUMONIA', '*.jpeg'))
        for img_path in pneumonia_cases:
            data.append({'image_path': img_path, 'label': 1, 'original_split': folder})

    df = pd.DataFrame(data)

    # 2. Separate the test set (we keep this completely unseen)
    test_df = df[df['original_split'] == 'test'].copy()
    
    # 3. Combine original train and val, then do a fresh 80/20 split
    train_val_df = df[df['original_split'] != 'test'].copy()
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=train_val_df['label'] # Ensures equal ratio of Pneumonia/Normal in both splits
    )

    # 4. Save to CSVs
    train_df[['image_path', 'label']].to_csv('train.csv', index=False)
    val_df[['image_path', 'label']].to_csv('val.csv', index=False)
    test_df[['image_path', 'label']].to_csv('test.csv', index=False)

    print(f"Created CSVs successfully!")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

if __name__ == "__main__":
    # This assumes your extracted Kaggle folder is named "chest_xray" 
    # and is in the exact same folder as this script.
    local_dataset_path = 'chest_xray' 
    
    print("Scanning folders and creating CSVs...")
    create_dataset_csvs(local_dataset_path)

