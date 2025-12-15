# DeepChecks ML Testing

This directory contains DeepChecks ML tests for automated ML validation in CI/CD.

## Overview

DeepChecks provides comprehensive ML testing including:
- **Data Integrity**: Detects data quality issues, missing values, duplicates
- **Train-Test Validation**: Detects data drift and distribution shifts
- **Model Performance**: Validates model performance and identifies issues
- **Error Analysis**: Analyzes model errors and weak segments

## Test Structure

### `test_ml_deepchecks.py`

Contains three main test classes:

1. **TestDataIntegrity**: Validates data quality
   - Data integrity checks
   - Train-test validation

2. **TestModelValidation**: Validates model performance
   - Model performance validation
   - Confusion matrix reports
   - Model error analysis

3. **TestDataDriftDetection**: Detects data drift
   - Feature drift detection
   - Distribution shifts

## Running Tests

### Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run all DeepChecks tests
pytest src/tests/test_ml_deepchecks.py -v

# Run specific test class
pytest src/tests/test_ml_deepchecks.py::TestDataIntegrity -v

# Run with database (requires DATABASE_URL in .env)
export DATABASE_URL=your_database_url
pytest src/tests/test_ml_deepchecks.py -v
```

### In CI/CD

DeepChecks tests run automatically:
- **On every push/PR**: Full test suite runs
- **Before model training**: Data integrity checks run
- **After model training**: Model validation runs

## Test Requirements

### Prerequisites

1. **Database Access**: Tests load data from PostgreSQL database
2. **Trained Models**: Some tests require trained models in `models/` directory
3. **Environment Variables**: `DATABASE_URL` must be set

### Test Data

Tests load data from the `spotify_songs` table:
- Loads up to 1000 samples for testing
- Creates train/test splits (80/20)
- Applies same feature engineering as training pipeline

## CI/CD Integration

### GitHub Actions Workflows

1. **`.github/workflows/tests.yml`**
   - Runs DeepChecks tests on every push/PR
   - Validates data integrity and model performance

2. **`.github/workflows/model-training.yml`**
   - Runs data integrity checks **before** training
   - Runs model validation **after** training
   - Prevents training if critical data issues detected

## Test Results

### Pass/Fail Criteria

- **Data Integrity**: Must pass (critical)
- **Train-Test Validation**: Warnings acceptable, failures investigated
- **Model Performance**: Validates minimum thresholds
- **Data Drift**: Warnings logged, not blocking

### Interpreting Results

- ✅ **PASSED**: No issues detected
- ⚠️ **WARNINGS**: Issues found but not critical
- ❌ **FAILED**: Critical issues that need attention

## Customization

### Adjusting Thresholds

Edit `test_ml_deepchecks.py` to adjust:
- Minimum accuracy thresholds
- Data quality requirements
- Drift detection sensitivity

### Adding New Checks

```python
from deepchecks.tabular.checks import YourCheck

def test_custom_check(self, prepare_datasets):
    check = YourCheck()
    result = check.run(...)
    assert result.passed
```

## Troubleshooting

### Tests Skipped

- **No database**: Set `DATABASE_URL` environment variable
- **No data**: Ensure database has data in `spotify_songs` table
- **No model**: Run model training first to generate models

### Tests Failing

1. Check data quality in database
2. Verify feature engineering matches training
3. Review DeepChecks output for specific issues
4. Check model performance metrics

## Best Practices

1. **Run before training**: Catch data issues early
2. **Run after training**: Validate model quality
3. **Monitor drift**: Track data changes over time
4. **Review warnings**: Address non-critical issues proactively

## Resources

- [DeepChecks Documentation](https://docs.deepchecks.com/)
- [DeepChecks GitHub](https://github.com/deepchecks/deepchecks)

