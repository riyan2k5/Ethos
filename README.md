# Ethos

A machine learning project for Spotify genre classification with automated CI/CD pipeline.

## CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline using GitHub Actions that automates:

### Workflows

1. **Code Quality Checks** (`.github/workflows/code-checks.yml`)
   - Code formatting checks with Black
   - Linting with flake8 and Pylint
   - Type checking with mypy
   - Runs on push and pull requests

2. **Unit Tests and ML Tests** (`.github/workflows/tests.yml`)
   - Runs unit tests for preprocessing modules
   - Runs ML-specific tests for model training
   - Generates coverage reports
   - Tests on Python 3.10 and 3.11
   - Uploads coverage to Codecov

3. **Data Validation** (`.github/workflows/data-validation.yml`)
   - Validates data schema and quality
   - Checks feature distributions
   - Ensures data meets minimum requirements
   - Triggers on data file changes

4. **Model Training** (`.github/workflows/model-training.yml`)
   - Automated model training on schedule (weekly)
   - Manual trigger support
   - Validates data before training
   - Saves model artifacts and metrics
   - Can be triggered on code changes to ML modules

5. **Container Image Building** (`.github/workflows/build-container.yml`)
   - Builds Docker images for training and production
   - Multi-stage builds for optimization
   - Pushes to GitHub Container Registry
   - Supports multi-platform builds (AMD64, ARM64)

6. **Deployment Pipeline** (`.github/workflows/deploy.yml`)
   - Staging and production deployments
   - Pre-deployment validation
   - Smoke tests after deployment
   - Manual and automated triggers

7. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Combined workflow for quick checks
   - Runs on every push and PR
   - Includes code quality, tests, and data validation

### Running Tests Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test files
pytest tests/test_preprocessing.py
pytest tests/test_ml_model.py
```

### Data Validation

```bash
# Validate data
python scripts/validate_data.py --data-path data/spotify_data_reduced.csv --target-column genre
```

### Building Docker Images

```bash
# Build training image
docker build --target training -t ethos:training .

# Build production image
docker build --target production -t ethos:production .

# Run training container
docker run -v $(pwd)/data:/app/data ethos:training
```

### Workflow Triggers

- **Push to main/develop**: Runs code checks, tests, and data validation
- **Pull Requests**: Runs all checks and tests
- **Tags (v*)**: Triggers container builds and deployments
- **Manual Dispatch**: All workflows support manual triggering
- **Scheduled**: Model training runs weekly on Mondays at 2 AM UTC

### Artifacts

The workflows generate and store:
- Test coverage reports
- Model artifacts (models, metrics)
- Data validation reports
- Container images (in GitHub Container Registry)

### Environment Variables

For deployment, you may need to configure:
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- Container registry credentials (if using external registry)
- Deployment target credentials (Kubernetes, cloud services, etc.)

## Project Structure

```
Ethos/
├── .github/
│   └── workflows/          # GitHub Actions workflows
├── data/                   # Data files (gitignored)
├── ml/                     # ML model training code
├── preprocessing/          # Data preprocessing scripts
├── scripts/                # Utility scripts (validation, etc.)
├── tests/                  # Test suite
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Validate data: `python scripts/validate_data.py`
5. Train model: `python ml/train_genre_model.py`

## Contributing

1. Create a feature branch
2. Make your changes
3. Ensure all tests pass: `pytest`
4. Check code quality: `black --check .` and `flake8 .`
5. Create a pull request

The CI/CD pipeline will automatically validate your changes.
