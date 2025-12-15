# Prefect ML Pipeline

This directory contains the Prefect workflow pipeline for orchestrating the ML model training process.

## Overview

The pipeline orchestrates model training for all 5 ML models:
- Genre Classification Model
- Clustering Model
- Energy Regression Model
- Popularity Regression Model
- Similar Songs Model

**Note**: Data is expected to already be cleaned and loaded in the database. Models handle their own feature engineering internally.

## Installation

Make sure Prefect is installed:

```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

Run the pipeline directly:

```bash
python workflows/run_pipeline.py
```

With custom options:

```bash
python workflows/run_pipeline.py \
    --csv-path data/spotify_data_reduced.csv \
    --table-name spotify_songs \
    --truncate \
    --n-clusters 2
```

### Using Prefect CLI

You can also run the flow using Prefect CLI:

```bash
# Run the flow
prefect deployment run ml-training-pipeline/ml-training-pipeline

# Or run directly
python -m workflows.ml_pipeline
```

### Programmatic Usage

```python
from workflows.ml_pipeline import ml_training_pipeline

# Run the complete pipeline
result = ml_training_pipeline(
    csv_path="data/spotify_data_reduced.csv",
    table_name="spotify_songs",
    truncate_table=False,
    n_clusters=2
)

print(result)
```

## Pipeline Structure

### Tasks

- `train_genre_model_task`: Trains genre classification model
- `train_clustering_model_task`: Trains clustering model
- `train_energy_regression_model_task`: Trains energy regression model
- `train_popularity_regression_model_task`: Trains popularity regression model
- `train_similar_songs_model_task`: Trains similar songs model

### Main Flow

`ml_training_pipeline`: Orchestrates all model training tasks (can run in parallel).

## Configuration

The pipeline uses the following default configuration:

- `TABLE_NAME`: `spotify_songs`
- `N_CLUSTERS`: `2`

These can be overridden when calling the pipeline.

## Environment Variables

Make sure the following environment variables are set:

- `DATABASE_URL`: PostgreSQL connection string (required for database operations)

## Prefect Cloud/Server Deployment

To deploy to Prefect Cloud or Server:

1. **Create a deployment**:

```python
from prefect.deployments import Deployment
from workflows.ml_pipeline import ml_training_pipeline

deployment = Deployment.build_from_flow(
    flow=ml_training_pipeline,
    name="ml-training-pipeline",
    work_queue_name="default"
)

deployment.apply()
```

2. **Run the deployment**:

```bash
prefect deployment run ml-training-pipeline/ml-training-pipeline
```

## Monitoring

When running with Prefect Cloud/Server, you can monitor:

- Task execution status
- Task run times
- Task logs
- Flow run history
- Error tracking

## Error Handling & Retries

The pipeline includes comprehensive error handling:

- **Retry Logic**: 
  - Model training tasks: 3 retries with exponential backoff (2s, 4s, 8s)
  - Database connection: 2 retries with 30s delay
  - Jitter factor to prevent thundering herd
  
- **Error Handling**:
  - Individual model failures don't stop the pipeline
  - Detailed error logging with context
  - Graceful degradation (partial success handling)
  
- **Notifications**:
  - Success notifications when all models train successfully
  - Warning notifications for partial failures
  - Failure notifications for complete pipeline failures

See `ERROR_HANDLING.md` for detailed information.

## Notifications

The pipeline supports Discord notifications via webhook.

Configure the Discord webhook URL in your `.env` file. See `NOTIFICATIONS_SETUP.md` for detailed setup instructions.

## Notes

- Model training tasks can run in parallel for better performance
- The pipeline assumes database connectivity is available
- Data must already be cleaned and loaded in the database
- Models handle their own feature engineering internally
- Models are saved to the `models/` directory
- Metrics are saved to `models/metrics/` directory
- Notifications are optional - pipeline works without them

## Prefect UI / "Not Found" Error

### Why You See "Not Found" Errors

Prefect 3.x runs in **ephemeral mode** by default. This means:
- A temporary server starts when you run a flow
- The server stops when the flow completes
- Links become invalid after the server stops
- **This is normal and doesn't affect your pipeline!**

### Solutions

**Option 1: Ignore the Links (Recommended for Local Development)**
- Your pipeline works perfectly fine without them
- Links are just for monitoring convenience
- No setup needed - just run: `python workflows/run_pipeline.py`

**Option 2: Start Persistent Server (For UI Access)**
```bash
# Terminal 1: Start Prefect server
prefect server start

# Terminal 2: Run your pipeline
python workflows/run_pipeline.py
```
Now links will work! Access UI at http://127.0.0.1:4200

**Option 3: Use Prefect Cloud (For Production)**
- Sign up at https://app.prefect.cloud
- Authenticate: `prefect cloud login`
- Deploy flows to cloud for persistent tracking

See `PREFECT_SETUP.md` for detailed setup instructions.

