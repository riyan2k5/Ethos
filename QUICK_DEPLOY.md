# Quick Deploy to Render (5 minutes)

## Step 1: Push to GitHub
```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

## Step 2: Deploy on Render

1. Go to [render.com](https://render.com) and sign up/login with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select the **Ethos** repository
4. Configure:
   - **Name**: `ethos-music-recommender`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Add Environment Variables:
   - `SPOTIFY_CLIENT_ID` = (your Spotify client ID)
   - `SPOTIFY_CLIENT_SECRET` = (your Spotify client secret)
6. Click **"Create Web Service"**

## Step 3: Wait for Deployment

Render will:
- Clone your repo (including Git LFS files)
- Install dependencies
- Start your app

Your app will be live at: `https://ethos-music-recommender.onrender.com`

**Note**: Free tier spins down after 15 min inactivity. First request may take 30-60 seconds.

---

## Alternative: Railway

1. Go to [railway.app](https://railway.app) and sign up with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Add environment variables (same as above)
5. Railway auto-detects Python and uses the `Procfile`

Done! 🚀

