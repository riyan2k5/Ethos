# Prefect Setup Guide

## Understanding Prefect Links and "Not Found" Errors

When you run Prefect flows, you may see links in the output like:
```
View flow run: http://127.0.0.1:4200/runs/flow-run/abc123
```

If you click these links and get `{"detail":"Not Found"}`, it's because:

### The Problem

Prefect 3.x runs in **ephemeral mode** by default. This means:
- A temporary server starts automatically when you run a flow
- The server stops when the flow completes
- Links are generated but become invalid once the server stops
- This is designed for quick local runs without needing a persistent server

### Solutions

You have three options:

## Option 1: Run Without Server (Simplest) ✅ **RECOMMENDED**

Just ignore the links! The pipeline works perfectly fine without them. The links are just for monitoring - your flows will still run and complete successfully.

```bash
python workflows/run_pipeline.py
```

**Pros:**
- No setup needed
- Works immediately
- Perfect for local development

**Cons:**
- No UI to monitor runs
- Can't view historical runs

## Option 2: Start Persistent Prefect Server

If you want to use the UI and view flow runs:

### Step 1: Start Prefect Server

In a **separate terminal**, run:

```bash
prefect server start
```

This will:
- Start a persistent Prefect server
- Keep it running until you stop it (Ctrl+C)
- Make all links accessible at http://127.0.0.1:4200

### Step 2: Run Your Pipeline

In another terminal:

```bash
python workflows/run_pipeline.py
```

Now the links will work! You can:
- Click on flow run links to see details
- View task execution logs
- Monitor run history
- See retry attempts and errors

### Step 3: Stop the Server

When done, press `Ctrl+C` in the server terminal.

**Pros:**
- Full UI access
- Historical run tracking
- Better debugging

**Cons:**
- Need to keep server running
- Uses some system resources

## Option 3: Use Prefect Cloud (For Production)

For production deployments, use Prefect Cloud:

1. Sign up at https://app.prefect.cloud
2. Get your API key
3. Authenticate: `prefect cloud login`
4. Deploy your flow to cloud
5. All links will work and persist

**Pros:**
- Persistent storage
- Team collaboration
- Production-ready
- Free tier available

**Cons:**
- Requires account setup
- More complex for simple local runs

## Current Setup

Your current setup uses **ephemeral mode** (Option 1). This is perfect for:
- Local development
- Quick model training runs
- Testing workflows

The "Not Found" errors are **normal** and don't indicate a problem with your pipeline. Your flows are running successfully - you just can't view them in the UI after they complete.

## Quick Test

To verify everything works:

```bash
# Run the pipeline (ephemeral mode - links won't work after completion)
python workflows/run_pipeline.py

# Or start server first, then run pipeline (links will work)
prefect server start  # Terminal 1
python workflows/run_pipeline.py  # Terminal 2
```

## Summary

- ✅ **"Not Found" errors are normal** in ephemeral mode
- ✅ **Your pipeline works fine** - ignore the links
- ✅ **To use links**: Start `prefect server start` first
- ✅ **For production**: Use Prefect Cloud

The pipeline functionality is not affected by these links - they're just for monitoring convenience!

