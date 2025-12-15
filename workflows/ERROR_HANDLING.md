# Error Handling & Retry Logic

## Overview

The Prefect ML pipeline includes comprehensive error handling and retry logic to ensure robust model training.

## Retry Logic

### Task-Level Retries

All model training tasks use:
- **3 retries** per task
- **Exponential backoff** (delays: 2s, 4s, 8s)
- **Jitter factor** (0.5) to prevent thundering herd

```python
@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5
)
```

### Database Connection Retries

Database verification uses:
- **2 retries**
- **30 second delay** between retries

## Error Handling Strategy

### Individual Model Error Handling

Each model training task is wrapped in try-except:
- **Continues pipeline** if one model fails
- **Logs detailed errors** for debugging
- **Tracks success/failure** status per model
- **Sends partial failure warnings** via notifications

### Flow-Level Error Handling

The main flow:
- **Catches all exceptions**
- **Sends failure notifications**
- **Logs comprehensive error details**
- **Tracks duration before failure**

## Error Types Handled

1. **Database Connection Errors**
   - Connection timeouts
   - Authentication failures
   - Network issues

2. **Model Training Errors**
   - Insufficient data
   - Feature extraction failures
   - Memory errors
   - Model serialization errors

3. **Notification Errors**
   - Discord webhook failures (non-blocking)
   - Network timeouts

## Error Recovery

### Automatic Recovery

- **Transient errors**: Automatically retried (network issues, timeouts)
- **Database connection**: Retried with exponential backoff
- **Model training**: Each model retried independently

### Manual Recovery

For persistent failures:
1. Check logs for specific error messages
2. Verify database connectivity
3. Check data availability
4. Review model-specific requirements

## Error Logging

All errors are logged with:
- **Error message**
- **Error type**
- **Stack trace** (for debugging)
- **Context** (which model, what operation)
- **Timestamp**

## Best Practices

1. **Monitor notifications** for warnings/failures
2. **Check logs** for detailed error information
3. **Verify prerequisites** before running pipeline
4. **Use retries** for transient failures
5. **Handle partial failures** gracefully

## Example Error Scenarios

### Scenario 1: One Model Fails

```
✅ Genre model: Success
✅ Clustering model: Success
❌ Energy model: Failed (insufficient data)
✅ Popularity model: Success
✅ Similar songs model: Success

Result: Pipeline continues, warning notification sent
```

### Scenario 2: Database Connection Fails

```
❌ Database connection: Failed (timeout)
Result: Pipeline stops, failure notification sent
```

### Scenario 3: All Models Succeed

```
✅ All models: Success
Result: Success notification sent
```

## Debugging Failed Runs

1. **Check Prefect logs** for detailed error messages
2. **Review notification details** for error summary
3. **Verify environment** (database, data availability)
4. **Test individual models** to isolate issues
5. **Check retry attempts** in logs

