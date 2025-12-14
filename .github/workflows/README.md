# GitHub Actions Workflows

This directory contains all GitHub Actions workflows for the CI/CD pipeline.

## Workflow Overview

### 1. `code-checks.yml`
**Purpose**: Code quality and style checks
- Runs on: Push to main/develop, Pull requests
- Checks: Black formatting, flake8 linting, Pylint, mypy type checking

### 2. `tests.yml`
**Purpose**: Run unit tests and ML tests
- Runs on: Push to main/develop, Pull requests, Manual dispatch
- Tests: Preprocessing unit tests, ML model tests
- Coverage: Generates coverage reports and uploads to Codecov
- Python versions: 3.10, 3.11

### 3. `data-validation.yml`
**Purpose**: Validate data quality and schema
- Runs on: Data file changes, Pull requests, Manual dispatch
- Validates: Schema, data quality, feature distributions

### 4. `model-training.yml`
**Purpose**: Automated model training
- Runs on: Push to main (ML changes), Scheduled (weekly), Manual dispatch
- Steps: Data validation → Model training → Artifact storage

### 5. `build-container.yml`
**Purpose**: Build and push Docker container images
- Runs on: Push to main, Tags (v*), Pull requests (dry run), Manual dispatch
- Builds: Training and production images
- Registry: GitHub Container Registry (ghcr.io)

### 6. `deploy.yml`
**Purpose**: Deployment pipeline
- Runs on: Push to main, Tags (v*), Manual dispatch
- Environments: Staging, Production
- Steps: Pre-deployment tests → Deploy → Smoke tests

### 7. `ci.yml`
**Purpose**: Combined CI pipeline for quick feedback
- Runs on: Push to main/develop, Pull requests
- Combines: Code quality, tests, data validation, container build (dry run)

## Workflow Dependencies

```
code-checks ──┐
              ├──> ci.yml (combined)
tests ────────┤
              │
data-validation
              │
model-training ──> build-container ──> deploy
```

## Manual Triggers

All workflows support manual triggering via GitHub Actions UI:
1. Go to Actions tab
2. Select the workflow
3. Click "Run workflow"
4. Configure inputs (if any)
5. Run

## Scheduled Jobs

- **Model Training**: Every Monday at 2 AM UTC (configurable in `model-training.yml`)

## Secrets and Variables

Required secrets (configure in repository settings):
- `GITHUB_TOKEN`: Automatically provided
- Container registry credentials (if using external registry)
- Deployment credentials (Kubernetes, cloud services, etc.)

## Customization

To customize workflows:
1. Edit the respective `.yml` file
2. Adjust triggers, steps, or conditions
3. Commit and push changes
4. Workflows will use the updated configuration

## Troubleshooting

- **Workflow fails on data validation**: Ensure data files exist or workflow will create sample data
- **Container build fails**: Check Dockerfile syntax and dependencies
- **Tests fail**: Run tests locally first: `pytest tests/`
- **Deployment fails**: Verify deployment credentials and target environment configuration