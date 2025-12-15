# Discord Notifications Setup Guide

This guide explains how to set up Discord notifications for the ML training pipeline.

## Quick Setup (2 Minutes)

### Step 1: Create Discord Webhook

1. Open Discord and go to your server
2. Go to **Server Settings** > **Integrations** > **Webhooks**
3. Click **New Webhook**
4. Give it a name (e.g., "ML Pipeline Notifications")
5. Choose a channel where notifications should appear
6. Click **Copy Webhook URL**

### Step 2: Add to .env File

Add this line to your `.env` file:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

Replace the URL with your actual webhook URL.

### Step 3: Test

Run the pipeline:

```bash
python workflows/run_pipeline.py
```

You should receive notifications in your Discord channel!

## Notification Types

The pipeline sends three types of notifications:

### 1. Success Notification
Sent when all models train successfully:
- ✅ Title: "ML Training Pipeline Completed Successfully"
- Duration
- List of trained models

### 2. Warning Notification
Sent when some models fail but pipeline continues:
- ⚠️ Title: "ML Training Pipeline Warning"
- Number of successful/failed models
- List of failed models

### 3. Failure Notification
Sent when pipeline fails completely:
- ❌ Title: "ML Training Pipeline Failed"
- Error message
- Duration before failure

## Testing Notifications

You can test notifications without running the full pipeline:

```python
from workflows.notifications import NotificationService

service = NotificationService()

# Test success notification
service.notify_success(
    "Test Pipeline",
    120.5,
    {"models": "genre, clustering"}
)

# Test failure notification
service.notify_failure(
    "Test Pipeline",
    "Test error message",
    {"error_type": "TestError"}
)
```

## Disabling Notifications

To disable notifications, set `enable_notifications=False`:

```python
from workflows.ml_pipeline import ml_training_pipeline

result = ml_training_pipeline(
    table_name="spotify_songs",
    enable_notifications=False
)
```

## Troubleshooting

### "Invalid Webhook" Error
- Check that the webhook URL is correct
- Ensure the webhook hasn't been deleted
- Verify the URL format: `https://discord.com/api/webhooks/...`

### No Notifications Sent
- Check that `DISCORD_WEBHOOK_URL` is set in `.env`
- Verify `.env` file is in project root
- Check logs for notification errors
- Ensure webhook URL is valid

### Webhook Deleted
If you delete the webhook, create a new one and update the URL in `.env`.

## Security Notes

⚠️ **Important Security Tips:**

1. **Never commit `.env` file** - It contains your webhook URL
2. **Rotate webhook URLs** periodically
3. **Limit webhook permissions** in Discord (only send messages)
4. **Use environment variables** in production (not `.env` files)

## Example .env Configuration

```bash
# Database
DATABASE_URL=postgresql://...

# Discord Webhook
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

## Notification Format

Discord notifications use rich embeds with:
- Color-coded status (green=success, red=error, yellow=warning)
- Detailed fields with model information
- Timestamps
- Footer with pipeline name

This makes notifications easy to read and visually distinct!
