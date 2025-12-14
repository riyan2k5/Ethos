import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

# --- Configuration ---
INPUT_FILE = '../data/spotify_data_reduced.csv'
TARGET_COLUMN = 'genre'

def train_and_evaluate_model(input_path: str, target_col: str):
    """
    Loads the dataset, trains a Random Forest Classifier, and prints
    detailed performance metrics by genre.
    """
    print(f"--- Starting Model Training Pipeline ---")
    
    try:
        # 1. Load the Data
        print(f"Loading data from '{input_path}'...")
        df = pd.read_csv(input_path)
        
        # 2. Feature Selection
        # We need to drop the target column AND non-numeric identifiers (Metadata)
        # Check if 'track_id' exists and drop it if so, along with names
        metadata_cols = ['artist_name', 'track_name', 'track_id', target_col]
        
        # distinct valid features are columns NOT in metadata_cols
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Features used for training ({len(feature_cols)}):")
        print(feature_cols)
        
        # 3. Train/Test Split (80/20)
        print("\nSplitting data (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Note: stratify=y ensures each genre is represented proportionally in train and test sets
        
        # 4. Initialize and Train the Model
        print("Training Random Forest Classifier... (This may take a moment)")
        start_time = time.time()
        
        # Using 100 trees (n_estimators=100) is a standard starting point
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        
        # 5. Predictions
        print("Generating predictions on test set...")
        y_pred = clf.predict(X_test)
        
        # 6. Evaluation
        overall_acc = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print(f"✅ MODEL RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {overall_acc:.2%}")
        print("-" * 60)
        print("\nDetailed Report by Genre (Precision, Recall, F1-Score):\n")
        
        # This generates the table you asked for
        print(classification_report(y_test, y_pred))
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nERROR: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    train_and_evaluate_model(INPUT_FILE, TARGET_COLUMN)