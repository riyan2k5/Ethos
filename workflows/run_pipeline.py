"""
Simple script to run the Prefect ML pipeline.
This can be used to execute the pipeline locally or deploy it to Prefect Cloud/Server.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  .env file not found at {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Environment variables must be set manually.")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

from workflows.ml_pipeline import ml_training_pipeline


def main():
    """Main entry point for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML Training Pipeline with Prefect")
    parser.add_argument(
        "--table-name",
        type=str,
        default="spotify_songs",
        help="Name of the database table (default: spotify_songs)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of clusters for clustering model (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    print("=" * 80)
    print("Starting ML Training Pipeline")
    print("=" * 80)
    print(f"Table Name: {args.table_name}")
    print(f"N Clusters: {args.n_clusters}")
    print("Note: Data is expected to be already cleaned and in the database")
    print("=" * 80)
    print()
    
    try:
        result = ml_training_pipeline(
            table_name=args.table_name,
            n_clusters=args.n_clusters
        )
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        
        # Print summary
        if result.get("model_training"):
            print("\nModel Training Results:")
            for model_name, model_result in result["model_training"].items():
                status = model_result.get("status", "unknown")
                print(f"  {model_name}: {status}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

