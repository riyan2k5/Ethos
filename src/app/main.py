"""
FastAPI web application for Spotify song recommendations with ML predictions.
"""

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from dotenv import load_dotenv

from src.app.services.data_loader import DataLoader
from src.app.services.ml_service import MLService
from src.app.services.spotify_service import SpotifyService
from src.app.services.recommendation_service import RecommendationService
from src.app.services.model_analytics import ModelAnalytics

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ethos Music Recommender", version="1.0.0")

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

# Ensure directories exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

print(f"Static directory: {static_dir}")
print(f"Templates directory: {templates_dir}")
print(
    f"Static files exist: {(static_dir / 'styles.css').exists()}, {(static_dir / 'script.js').exists()}"
)

# Initialize services
data_loader = DataLoader()
ml_service = MLService()
spotify_service = SpotifyService()
recommendation_service = RecommendationService(data_loader, ml_service, spotify_service)
model_analytics = ModelAnalytics()


@app.on_event("startup")
async def startup_event():
    """Load data and models on startup."""
    try:
        print("Loading dataset...")
        data_loader.load_dataset()
        print(
            f"✅ Dataset loaded: {len(data_loader.df) if data_loader.df is not None else 0} songs"
        )
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        import traceback

        traceback.print_exc()

    try:
        print("Loading ML models...")
        ml_service.load_models()
        print("✅ ML models loaded")
    except Exception as e:
        print(f"⚠️  Error loading ML models: {e}")
        import traceback

        traceback.print_exc()

    try:
        print("Initializing Spotify service...")
        await spotify_service.initialize()
        print("✅ Spotify service initialized")
    except Exception as e:
        print(f"⚠️  Error initializing Spotify service: {e}")
        import traceback

        traceback.print_exc()

    print("✅ All services initialized!")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with recommended and clustered songs."""
    try:
        # Get user ID from cookies
        user_id = request.cookies.get("user_id", "anonymous")

        # Get recommendations (reduced for faster loading)
        recommended = recommendation_service.get_recommendations(user_id, limit=15)

        # Get clustered songs (reduced for faster loading)
        clusters = recommendation_service.get_clustered_songs(limit=10)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "recommended": recommended or [],
                "clusters": clusters or {},
            },
        )
    except Exception as e:
        print(f"Error in home route: {e}")
        import traceback

        traceback.print_exc()
        # Return template even if there's an error
        return templates.TemplateResponse(
            "index.html", {"request": request, "recommended": [], "clusters": {}}
        )


@app.get("/api/search")
async def search(query: str, limit: int = 20):
    """Search for songs in the dataset."""
    if not query or len(query) < 2:
        return []

    results = data_loader.search_songs(query, limit)
    return results


@app.get("/api/song/{track_id}")
async def get_song(track_id: str):
    """Get song details with AI predictions."""
    import asyncio

    try:
        song = data_loader.get_song_by_id(track_id)
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")

        # Get AI predictions (fast, synchronous)
        predictions = {}
        try:
            predictions = ml_service.predict_all(song)
        except Exception as e:
            print(f"Error getting predictions for {track_id}: {e}")
            predictions = {
                "genre": {
                    "predicted": "Unknown",
                    "probabilities": {},
                    "top_predictions": [],
                }
            }

        # Get similar songs (fast, synchronous)
        similar = []
        try:
            similar = recommendation_service.get_similar_songs(
                track_id, limit=5
            )  # Reduced from 10
        except Exception as e:
            print(f"Error getting similar songs for {track_id}: {e}")
            similar = []

        # Get cover art from Spotify (async, but with shorter timeout)
        cover_art = None
        try:
            # Set a shorter timeout for Spotify API call (1.5 seconds)
            cover_art = await asyncio.wait_for(
                spotify_service.get_cover_art(track_id), timeout=1.5
            )
        except asyncio.TimeoutError:
            # Silently fail - placeholder will be used
            cover_art = None
        except Exception as e:
            # Silently fail - placeholder will be used
            cover_art = None

        # Don't load cover art for similar songs here - let frontend load lazily
        # This makes the API response much faster

        return {
            "song": song,
            "predictions": predictions,
            "similar": similar,
            "cover_art": cover_art,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_song endpoint for {track_id}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error loading song details: {str(e)}"
        )


@app.post("/api/interact")
async def track_interaction(request: Request):
    """Track user interaction (click, like, etc.) for recommendations."""
    data = await request.json()
    user_id = request.cookies.get("user_id", "anonymous")
    track_id = data.get("track_id")
    action = data.get("action", "click")

    if not track_id:
        raise HTTPException(status_code=400, detail="track_id required")

    recommendation_service.track_interaction(user_id, track_id, action)

    return {"status": "success"}


@app.get("/api/recommendations")
async def get_recommendations(request: Request, limit: int = 20):
    """Get personalized recommendations."""
    try:
        user_id = request.cookies.get("user_id", "anonymous")
        recommendations = recommendation_service.get_recommendations(user_id, limit)
        # Don't load cover art here - let frontend load lazily
        return recommendations if recommendations else []
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        import traceback

        traceback.print_exc()
        return []


@app.get("/api/clusters")
async def get_clusters(limit: int = 15):
    """Get clustered songs."""
    try:
        clusters = recommendation_service.get_clustered_songs(limit)
        # Don't load cover art here - let frontend load lazily
        return clusters if clusters else {}
    except Exception as e:
        print(f"Error getting clusters: {e}")
        import traceback

        traceback.print_exc()
        return {}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "dataset_loaded": data_loader.df is not None,
        "dataset_size": len(data_loader.df) if data_loader.df is not None else 0,
        "models_loaded": ml_service.genre_model is not None,
    }


@app.get("/api/cover/{track_id}")
async def get_cover_art_only(track_id: str):
    """Get only cover art for a track (faster endpoint)."""
    try:
        cover_art = await spotify_service.get_cover_art(track_id)
        return {"cover_art": cover_art}
    except Exception as e:
        return {"cover_art": None}


@app.get("/api/analytics")
async def get_model_analytics():
    """Get analytics for all models."""
    try:
        analytics = model_analytics.get_all_analytics()
        return analytics
    except Exception as e:
        print(f"Error getting analytics: {e}")
        import traceback

        traceback.print_exc()
        return {"models": [], "dataset_info": {}}


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a custom dataset file."""
    try:
        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")

        # Read file content
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Load the dataset
        result = data_loader.load_custom_dataset(contents, file.filename)

        if result.get("success"):
            # Update the global recommendation service
            global recommendation_service
            recommendation_service = RecommendationService(
                data_loader, ml_service, spotify_service
            )

            return JSONResponse(content=result)
        else:
            error_msg = result.get("message", "Unknown error loading dataset")
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        import traceback

        traceback.print_exc()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        raise HTTPException(
            status_code=500, detail=f"Error uploading dataset: {error_msg}"
        )


@app.post("/api/reset-dataset")
async def reset_dataset():
    """Reset to default dataset."""
    try:
        data_loader.reset_to_default()
        return {"success": True, "message": "Reset to default dataset"}
    except Exception as e:
        print(f"Error resetting dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error resetting dataset: {str(e)}"
        )


@app.get("/api/dataset-info")
async def get_dataset_info():
    """Get information about current dataset."""
    try:
        df = data_loader.df
        if df is None or df.empty:
            return {"count": 0, "is_custom": data_loader.use_custom}

        return {
            "count": len(df),
            "is_custom": data_loader.use_custom,
            "columns": list(df.columns) if hasattr(df, "columns") else [],
        }
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return {"count": 0, "is_custom": False, "columns": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
