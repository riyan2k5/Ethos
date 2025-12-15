"""
Example usage of the Prefect ML pipeline.
This demonstrates different ways to run the pipeline.
"""
from workflows.ml_pipeline import (
    ml_training_pipeline,
    ingest_data_task,
    preprocess_data_task,
    engineer_features_task,
    train_genre_model_task,
    train_clustering_model_task,
    train_energy_regression_model_task,
    train_popularity_regression_model_task,
    train_similar_songs_model_task
)


def example_full_pipeline():
    """Example: Run the complete pipeline."""
    print("Running complete ML training pipeline...")
    
    result = ml_training_pipeline(
        table_name="spotify_songs",
        n_clusters=2
    )
    
    print("\nPipeline Results:")
    if result.get('model_training'):
        print("\nModel Training Results:")
        for model_name, model_result in result['model_training'].items():
            status = model_result.get('status', 'unknown')
            print(f"  {model_name}: {status}")


def example_individual_tasks():
    """Example: Run individual tasks manually."""
    from workflows.ml_pipeline import (
        train_genre_model_task,
        train_clustering_model_task,
        train_energy_regression_model_task,
        train_popularity_regression_model_task,
        train_similar_songs_model_task
    )
    
    print("Running individual tasks...")
    
    # Train models individually
    genre_result = train_genre_model_task("spotify_songs", "genre")
    print(f"Genre Model: {genre_result.get('status')}")
    
    clustering_result = train_clustering_model_task("spotify_songs", 2)
    print(f"Clustering Model: {clustering_result.get('status')}")
    
    energy_result = train_energy_regression_model_task("spotify_songs", "energy")
    print(f"Energy Model: {energy_result.get('status')}")
    
    popularity_result = train_popularity_regression_model_task("spotify_songs", "popularity")
    print(f"Popularity Model: {popularity_result.get('status')}")
    
    similar_songs_result = train_similar_songs_model_task("spotify_songs")
    print(f"Similar Songs Model: {similar_songs_result.get('status')}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
        if example_type == "full":
            example_full_pipeline()
        elif example_type == "individual":
            example_individual_tasks()
        else:
            print(f"Unknown example type: {example_type}")
            print("Usage: python example_usage.py [full|individual]")
    else:
        print("Usage: python example_usage.py [full|individual]")
        print("\nExamples:")
        print("  python example_usage.py full        # Run complete pipeline")
        print("  python example_usage.py individual  # Run individual tasks")

