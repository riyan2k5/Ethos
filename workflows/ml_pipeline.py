"""
Prefect workflow pipeline for ML model training.
Orchestrates model training, evaluation, and model saving.
Data is expected to already be cleaned and loaded in the database.
Models handle their own feature engineering internally.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
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

from prefect import flow, task, get_run_logger
from prefect.tasks import exponential_backoff
from datetime import datetime, timedelta

# Import existing training functions
from src.ml.train_genre_model import train_and_evaluate_model
from src.ml.train_clustering_model import train_clustering_model
from src.ml.train_energy_regression_model import train_energy_regression_model
from src.ml.train_popularity_regression_model import train_popularity_regression_model
from src.ml.train_similar_songs_model import train_similar_songs_model

# Import notification service
from workflows.notifications import NotificationService


# Configuration
TABLE_NAME = "spotify_songs"
N_CLUSTERS = 2


@task(
    name="train_genre_model",
    log_prints=True,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
def train_genre_model_task(table_name: str, target_col: str = "genre") -> Dict[str, Any]:
    """
    Task to train genre classification model.
    
    Args:
        table_name: Name of the database table
        target_col: Name of the target column
        
    Returns:
        Dictionary with training results
    """
    # Ensure .env is loaded in task context
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except:
        pass
    
    logger = get_run_logger()
    logger.info(f"Starting genre classification model training")
    
    try:
        train_and_evaluate_model(table_name, target_col)
        
        logger.info("✅ Genre classification model training complete")
        
        return {
            "status": "success",
            "model_type": "genre_classification",
            "table_name": table_name,
            "target_col": target_col
        }
        
    except Exception as e:
        logger.error(f"❌ Genre model training failed: {e}")
        raise


@task(
    name="train_clustering_model",
    log_prints=True,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
def train_clustering_model_task(table_name: str, n_clusters: int = 2) -> Dict[str, Any]:
    """
    Task to train clustering model.
    
    Args:
        table_name: Name of the database table
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with training results
    """
    # Ensure .env is loaded in task context
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except:
        pass
    
    logger = get_run_logger()
    logger.info(f"Starting clustering model training with {n_clusters} clusters")
    
    try:
        train_clustering_model(table_name, n_clusters)
        
        logger.info("✅ Clustering model training complete")
        
        return {
            "status": "success",
            "model_type": "clustering",
            "table_name": table_name,
            "n_clusters": n_clusters
        }
        
    except Exception as e:
        logger.error(f"❌ Clustering model training failed: {e}")
        raise


@task(
    name="train_energy_regression_model",
    log_prints=True,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
def train_energy_regression_model_task(table_name: str, target_col: str = "energy") -> Dict[str, Any]:
    """
    Task to train energy regression model.
    
    Args:
        table_name: Name of the database table
        target_col: Name of the target column
        
    Returns:
        Dictionary with training results
    """
    # Ensure .env is loaded in task context
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except:
        pass
    
    logger = get_run_logger()
    logger.info(f"Starting energy regression model training")
    
    try:
        train_energy_regression_model(table_name, target_col)
        
        logger.info("✅ Energy regression model training complete")
        
        return {
            "status": "success",
            "model_type": "energy_regression",
            "table_name": table_name,
            "target_col": target_col
        }
        
    except Exception as e:
        logger.error(f"❌ Energy regression model training failed: {e}")
        raise


@task(
    name="train_popularity_regression_model",
    log_prints=True,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
def train_popularity_regression_model_task(table_name: str, target_col: str = "popularity") -> Dict[str, Any]:
    """
    Task to train popularity regression model.
    
    Args:
        table_name: Name of the database table
        target_col: Name of the target column
        
    Returns:
        Dictionary with training results
    """
    # Ensure .env is loaded in task context
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except:
        pass
    
    logger = get_run_logger()
    logger.info(f"Starting popularity regression model training")
    
    try:
        train_popularity_regression_model(table_name, target_col)
        
        logger.info("✅ Popularity regression model training complete")
        
        return {
            "status": "success",
            "model_type": "popularity_regression",
            "table_name": table_name,
            "target_col": target_col
        }
        
    except Exception as e:
        logger.error(f"❌ Popularity regression model training failed: {e}")
        raise


@task(
    name="train_similar_songs_model",
    log_prints=True,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
def train_similar_songs_model_task(table_name: str) -> Dict[str, Any]:
    """
    Task to train similar songs model.
    
    Args:
        table_name: Name of the database table
        
    Returns:
        Dictionary with training results
    """
    # Ensure .env is loaded in task context
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except:
        pass
    
    logger = get_run_logger()
    logger.info(f"Starting similar songs model training")
    
    try:
        train_similar_songs_model(table_name)
        
        logger.info("✅ Similar songs model training complete")
        
        return {
            "status": "success",
            "model_type": "similar_songs",
            "table_name": table_name
        }
        
    except Exception as e:
        logger.error(f"❌ Similar songs model training failed: {e}")
        raise


@task(
    name="verify_database_connection",
    log_prints=True,
    retries=2,
    retry_delay_seconds=30
)
def verify_database_connection_task() -> Dict[str, Any]:
    """
    Task to verify database connection before starting model training.
    
    Returns:
        Dictionary with connection verification results
    """
    logger = get_run_logger()
    logger.info("Verifying database connection...")
    
    try:
        # Ensure .env is loaded in task context
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=True)
        except:
            pass
        
        from src.api.db_connection import test_connection
        
        if test_connection():
            logger.info("✅ Database connection verified successfully")
            return {
                "status": "success",
                "message": "Database connection verified"
            }
        else:
            raise Exception("Database connection test failed")
            
    except Exception as e:
        logger.error(f"❌ Database connection verification failed: {e}")
        logger.error("Please ensure:")
        logger.error("  1. DATABASE_URL environment variable is set")
        logger.error("  2. Database is accessible")
        logger.error("  3. .env file exists in project root with DATABASE_URL")
        raise


@flow(name="ml_training_pipeline", log_prints=True)
def ml_training_pipeline(
    table_name: str = TABLE_NAME,
    n_clusters: int = N_CLUSTERS,
    enable_notifications: bool = True
) -> Dict[str, Any]:
    """
    Main Prefect flow orchestrating ML model training.
    
    Assumes data is already cleaned and loaded in the database.
    Models handle their own feature engineering internally.
    
    Args:
        table_name: Name of the database table containing cleaned data
        n_clusters: Number of clusters for clustering model
        enable_notifications: Whether to send success/failure notifications
        
    Returns:
        Dictionary with pipeline execution results
    """
    start_time = datetime.now()
    logger = get_run_logger()
    notification_service = NotificationService() if enable_notifications else None
    
    logger.info("=" * 80)
    logger.info("ML TRAINING PIPELINE - STARTING")
    logger.info("=" * 80)
    logger.info(f"Table: {table_name}")
    logger.info("Note: Data is expected to be already cleaned and in the database")
    logger.info("Models will handle feature engineering internally")
    
    if notification_service:
        if notification_service.enabled:
            logger.info("Discord notifications enabled")
        else:
            logger.info("Notifications enabled but Discord webhook not configured")
    else:
        logger.info("Notifications disabled")
    
    logger.info("=" * 80)
    
    results = {
        "database_verification": None,
        "model_training": {},
        "notification_sent": None
    }
    
    try:
        # Verify database connection first
        logger.info("\n" + "=" * 80)
        logger.info("VERIFYING DATABASE CONNECTION")
        logger.info("=" * 80)
        db_check = verify_database_connection_task()
        results["database_verification"] = db_check
        
        # Model Training (all models can run in parallel)
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING")
        logger.info("=" * 80)
        
        # Train all models (they can run in parallel)
        # Wrap each in try-except for individual error handling
        training_errors = []
        
        try:
            genre_result = train_genre_model_task(table_name, "genre")
            results["model_training"]["genre_classification"] = genre_result
        except Exception as e:
            error_msg = f"Genre model training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            training_errors.append(error_msg)
            results["model_training"]["genre_classification"] = {
                "status": "failed",
                "error": str(e)
            }
        
        try:
            clustering_result = train_clustering_model_task(table_name, n_clusters)
            results["model_training"]["clustering"] = clustering_result
        except Exception as e:
            error_msg = f"Clustering model training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            training_errors.append(error_msg)
            results["model_training"]["clustering"] = {
                "status": "failed",
                "error": str(e)
            }
        
        try:
            energy_result = train_energy_regression_model_task(table_name, "energy")
            results["model_training"]["energy_regression"] = energy_result
        except Exception as e:
            error_msg = f"Energy regression model training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            training_errors.append(error_msg)
            results["model_training"]["energy_regression"] = {
                "status": "failed",
                "error": str(e)
            }
        
        try:
            popularity_result = train_popularity_regression_model_task(table_name, "popularity")
            results["model_training"]["popularity_regression"] = popularity_result
        except Exception as e:
            error_msg = f"Popularity regression model training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            training_errors.append(error_msg)
            results["model_training"]["popularity_regression"] = {
                "status": "failed",
                "error": str(e)
            }
        
        try:
            similar_songs_result = train_similar_songs_model_task(table_name)
            results["model_training"]["similar_songs"] = similar_songs_result
        except Exception as e:
            error_msg = f"Similar songs model training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            training_errors.append(error_msg)
            results["model_training"]["similar_songs"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        successful_models = [
            name for name, result in results["model_training"].items()
            if result.get("status") == "success"
        ]
        failed_models = [
            name for name, result in results["model_training"].items()
            if result.get("status") == "failed"
        ]
        
        logger.info(f"✅ Successful models: {len(successful_models)}/{len(results['model_training'])}")
        if successful_models:
            logger.info(f"   {', '.join(successful_models)}")
        
        if failed_models:
            logger.warning(f"❌ Failed models: {len(failed_models)}/{len(results['model_training'])}")
            logger.warning(f"   {', '.join(failed_models)}")
        
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        logger.info("=" * 80)
        
        # Send notifications
        if notification_service:
            try:
                if failed_models:
                    # Partial failure - send warning
                    details = {
                        "successful": len(successful_models),
                        "failed": len(failed_models),
                        "failed_models": ", ".join(failed_models),
                        "duration": f"{duration:.2f}s"
                    }
                    notification_sent = notification_service.notify_warning(
                        "ML Training Pipeline",
                        f"Pipeline completed with {len(failed_models)} model(s) failing",
                        details
                    )
                    results["notification_sent"] = notification_sent
                else:
                    # Complete success
                    details = {
                        "models_trained": len(successful_models),
                        "models": ", ".join(successful_models),
                        "duration": f"{duration:.2f}s"
                    }
                    notification_sent = notification_service.notify_success(
                        "ML Training Pipeline",
                        duration,
                        details
                    )
                    results["notification_sent"] = notification_sent
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
                results["notification_sent"] = False
        
        if failed_models:
            logger.info("ML TRAINING PIPELINE - COMPLETED WITH ERRORS")
            logger.info("=" * 80)
        else:
            logger.info("ML TRAINING PIPELINE - COMPLETE")
            logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        error_msg = str(e)
        
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {error_msg}")
        logger.error(f"Duration before failure: {duration:.2f} seconds")
        logger.error("=" * 80)
        
        # Send failure notification
        if notification_service:
            try:
                details = {
                    "duration": f"{duration:.2f}s",
                    "error_type": type(e).__name__
                }
                notification_sent = notification_service.notify_failure(
                    "ML Training Pipeline",
                    error_msg,
                    details
                )
                results["notification_sent"] = notification_sent
            except Exception as notify_error:
                logger.warning(f"Failed to send failure notification: {notify_error}")
                results["notification_sent"] = False
        
        # Re-raise to mark flow as failed
        raise


if __name__ == "__main__":
    # Run the pipeline
    result = ml_training_pipeline(
        table_name=TABLE_NAME,
        n_clusters=N_CLUSTERS
    )
    print("\nPipeline Results:")
    print(result)

