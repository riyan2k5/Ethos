# Deployment Guide

This guide explains how to deploy the Ethos Music Recommender web app to various hosting platforms.

## Quick Deploy Options

### Option 1: Render (Recommended - Free Tier Available)

**Render** offers a free tier with automatic deployments from GitHub.

#### Steps:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git push origin main
   ```

2. **Sign up/Login to Render**: Go to [render.com](https://render.com) and sign up with your GitHub account

3. **Create a New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository: `Ethos`

4. **Configure the service**:
   - **Name**: `ethos-music-recommender` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Select "Free" (or upgrade if needed)

5. **Set Environment Variables**:
   - `SPOTIFY_CLIENT_ID`: Your Spotify Client ID
   - `SPOTIFY_CLIENT_SECRET`: Your Spotify Client Secret
   - `PYTHONPATH`: `/opt/render/project/src`

6. **Deploy**: Click "Create Web Service"

   Your app will be available at: `https://ethos-music-recommender.onrender.com` (or your custom domain)

**Note**: Render free tier spins down after 15 minutes of inactivity. First request may take 30-60 seconds to wake up.

---

### Option 2: Railway (Free Tier Available)

**Railway** offers a free tier with $5/month credit.

#### Steps:

1. **Push your code to GitHub**

2. **Sign up/Login to Railway**: Go to [railway.app](https://railway.app) and sign up with GitHub

3. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `Ethos` repository

4. **Configure**:
   - Railway will auto-detect Python
   - It will use the `Procfile` automatically
   - If not, set start command: `uvicorn src.app.main:app --host 0.0.0.0 --port $PORT`

5. **Set Environment Variables**:
   - Go to "Variables" tab
   - Add:
     - `SPOTIFY_CLIENT_ID`
     - `SPOTIFY_CLIENT_SECRET`

6. **Deploy**: Railway will automatically deploy

   Your app will be available at: `https://your-app-name.up.railway.app`

---

### Option 3: Fly.io (Free Tier Available)

**Fly.io** offers a free tier with generous limits.

#### Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Initialize Fly app**:
   ```bash
   fly launch
   ```
   - Follow prompts to create app
   - It will detect Dockerfile automatically

4. **Set Secrets**:
   ```bash
   fly secrets set SPOTIFY_CLIENT_ID=your_client_id
   fly secrets set SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

5. **Deploy**:
   ```bash
   fly deploy
   ```

   Your app will be available at: `https://your-app-name.fly.dev`

---

### Option 4: Docker Deployment

If you prefer to deploy using Docker:

1. **Build the image**:
   ```bash
   docker build -t ethos-app --target production .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e SPOTIFY_CLIENT_ID=your_client_id \
     -e SPOTIFY_CLIENT_SECRET=your_client_secret \
     ethos-app
   ```

3. **Deploy to any Docker-compatible platform**:
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform
   - Heroku (with Docker support)

---

## Important Notes

### Git LFS for Model Files

Your model files (`.pkl`) are tracked with Git LFS. When deploying:

1. **Render/Railway**: These platforms automatically handle Git LFS
2. **Fly.io**: Make sure Git LFS is installed in your build process
3. **Docker**: Git LFS files are included when you clone the repo

### Environment Variables Required

Make sure to set these environment variables on your hosting platform:

- `SPOTIFY_CLIENT_ID`: Your Spotify API client ID
- `SPOTIFY_CLIENT_SECRET`: Your Spotify API client secret

### Model Files Size

Your model files are stored via Git LFS. The free tier (1GB storage, 1GB bandwidth/month) should be sufficient for most use cases.

### Database (Optional)

Currently, the app uses in-memory storage. If you need persistent storage, consider:
- PostgreSQL (available on Render, Railway, Fly.io)
- SQLite (for simple deployments)

---

## Troubleshooting

### App won't start
- Check logs on your hosting platform
- Verify environment variables are set correctly
- Ensure `requirements.txt` includes all dependencies

### Models not loading
- Verify Git LFS files are pulled correctly
- Check that `src/models/` directory exists
- Review startup logs for model loading errors

### Spotify API errors
- Verify your Spotify API credentials are correct
- Check API rate limits
- Ensure environment variables are set

---

## Cost Comparison

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| **Render** | ✅ Yes (spins down after inactivity) | $7/month+ |
| **Railway** | ✅ $5/month credit | Pay as you go |
| **Fly.io** | ✅ Generous free tier | $1.94/month+ |
| **Heroku** | ❌ No free tier | $7/month+ |

---

## Recommended: Render

For easiest setup with GitHub integration, **Render** is recommended:
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Easy environment variable management
- ✅ Built-in SSL/HTTPS
- ✅ Simple configuration

Just connect your GitHub repo and deploy! 🚀

