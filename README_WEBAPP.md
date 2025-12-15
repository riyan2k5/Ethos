# Ethos Web Application

A FastAPI web application for music discovery with ML-powered recommendations.

## Features

- 🎵 **Music Discovery**: Browse recommended songs based on popularity and your listening history
- 🤖 **AI Predictions**: See AI-predicted genre tags for each song
- 🔍 **Search**: Search through the entire dataset
- 🎨 **Beautiful UI**: Clean, modern green-themed interface
- 🎯 **Personalized Recommendations**: Recommendations improve as you interact with songs
- 🎭 **Clustered Songs**: Explore songs grouped by musical characteristics
- 🖼️ **Spotify Integration**: Automatic cover art fetching from Spotify API

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp env.template .env
   ```
   
   Edit `.env` and add your Spotify API credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

3. **Train Models (if not already done)**
   ```bash
   python scripts/train_all_models.py
   ```

4. **Run the Application**
   ```bash
   python run_app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the Web App**
   Open your browser and navigate to: `http://localhost:8000`

## Project Structure

```
Ethos/
├── app/
│   ├── main.py              # FastAPI application
│   └── services/
│       ├── data_loader.py   # Dataset loading
│       ├── ml_service.py     # ML predictions
│       ├── spotify_service.py # Spotify API integration
│       └── recommendation_service.py # Recommendation logic
├── templates/
│   └── index.html           # Main HTML template
├── static/
│   ├── styles.css           # Styling
│   └── script.js            # Frontend JavaScript
├── data/
│   └── fortest/
│       └── Spotify_Dataset_V3.csv  # Dataset
└── models/                  # Trained ML models
```

## API Endpoints

- `GET /` - Home page
- `GET /api/search?query=<query>&limit=<limit>` - Search songs
- `GET /api/song/<track_id>` - Get song details with predictions
- `GET /api/recommendations?limit=<limit>` - Get personalized recommendations
- `GET /api/clusters?limit=<limit>` - Get clustered songs
- `POST /api/interact` - Track user interactions

## How It Works

1. **Recommendations**: Initially shows popular songs. As users click/like songs, recommendations become personalized based on similar songs.

2. **AI Tags**: Uses the trained genre classification model to predict genre tags for each song.

3. **Similar Songs**: Uses the KNN similarity model to find songs with similar audio features.

4. **Clustering**: Groups songs into clusters based on energy and danceability characteristics.

5. **Cover Art**: Fetches album cover art from Spotify API using track IDs.

## Notes

- The dataset is loaded from `data/fortest/Spotify_Dataset_V3.csv`
- User interactions are stored in memory (reset on server restart)
- Cover art is cached per request
- Make sure your ML models are trained before running the app

