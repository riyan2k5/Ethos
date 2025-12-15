"""
Pipeline script to train all ML models in sequence.
This script orchestrates the training of all models:
- Genre Classification Model
- Clustering Model
- Energy Regression Model
- Popularity Regression Model
- Similar Songs Model
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import training functions from each model script
from ml.train_genre_model import train_and_evaluate_model as train_genre_model
from ml.train_clustering_model import train_clustering_model
from ml.train_energy_regression_model import train_energy_regression_model
from ml.train_popularity_regression_model import train_popularity_regression_model
from ml.train_similar_songs_model import train_similar_songs_model

# Configuration
TABLE_NAME = "spotify_songs"
N_CLUSTERS = 2  # For clustering model


class ModelTrainingPipeline:
    """Pipeline to train all ML models sequentially."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
    
    def run(self) -> bool:
        """
        Run the complete training pipeline for all models.
        
        Returns:
            bool: True if all models trained successfully, False otherwise
        """
        self.start_time = time.time()
        
        print("=" * 80)
        print("ML MODEL TRAINING PIPELINE")
        print("=" * 80)
        print(f"Training all models from database table: {TABLE_NAME}")
        print()
        
        # Define all training tasks
        training_tasks = [
            {
                "name": "Genre Classification Model",
                "function": lambda: train_genre_model(TABLE_NAME, "genre"),
                "description": "Random Forest Classifier for genre prediction"
            },
            {
                "name": "Clustering Model",
                "function": lambda: train_clustering_model(TABLE_NAME, N_CLUSTERS),
                "description": "KMeans clustering for genre groups"
            },
            {
                "name": "Energy Regression Model",
                "function": lambda: train_energy_regression_model(TABLE_NAME, "energy"),
                "description": "Random Forest Regressor for energy prediction"
            },
            {
                "name": "Popularity Regression Model",
                "function": lambda: train_popularity_regression_model(TABLE_NAME, "popularity"),
                "description": "Random Forest Regressor for popularity prediction"
            },
            {
                "name": "Similar Songs Model",
                "function": lambda: train_similar_songs_model(TABLE_NAME),
                "description": "KNN model for finding similar songs"
            }
        ]
        
        # Train each model
        success_count = 0
        failure_count = 0
        
        for idx, task in enumerate(training_tasks, 1):
            print("\n" + "=" * 80)
            print(f"MODEL {idx}/{len(training_tasks)}: {task['name']}")
            print("=" * 80)
            print(f"Description: {task['description']}")
            print()
            
            task_start_time = time.time()
            success = False
            error_message = None
            
            try:
                # Run the training function
                task['function']()
                success = True
                success_count += 1
                print(f"\n✅ {task['name']} completed successfully")
            except ConnectionError as e:
                error_message = f"Database connection error: {str(e)}"
                failure_count += 1
                print(f"\n❌ {task['name']} failed: {error_message}")
            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                failure_count += 1
                print(f"\n❌ {task['name']} failed: {error_message}")
                import traceback
                traceback.print_exc()
            
            task_end_time = time.time()
            task_duration = task_end_time - task_start_time
            
            # Store result
            self.results.append({
                "name": task['name'],
                "success": success,
                "duration": task_duration,
                "error": error_message
            })
            
            print(f"Duration: {task_duration:.2f} seconds")
        
        # Calculate total time
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Print summary
        self._print_summary(success_count, failure_count, total_duration)
        
        # Return True if all models succeeded
        return failure_count == 0
    
    def _print_summary(self, success_count: int, failure_count: int, total_duration: float):
        """Print a summary of the training pipeline execution."""
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Total models: {len(self.results)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failure_count}")
        print(f"⏱️  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print()
        
        print("Detailed Results:")
        print("-" * 80)
        for result in self.results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{status:12} | {result['name']:40} | {result['duration']:8.2f}s")
            if result['error']:
                print(f"             | Error: {result['error']}")
        
        print("=" * 80)
        
        if failure_count == 0:
            print("\n🎉 All models trained successfully!")
        else:
            print(f"\n⚠️  {failure_count} model(s) failed. Check the errors above.")
        
        print()


def main():
    """Main entry point for the training pipeline."""
    pipeline = ModelTrainingPipeline()
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

