"""
Script to load spotify_data_reduced.csv into PostgreSQL database.
All credentials are read from environment variables - NEVER hardcoded.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.db_connection import get_db_connection, test_connection
from src.api.db_config import DatabaseConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_table_if_not_exists(table_name: str = "spotify_songs"):
    """
    Create the spotify_songs table if it doesn't exist.

    Args:
        table_name: Name of the table to create
    """
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        artist_name VARCHAR(255),
        track_name VARCHAR(255),
        track_id VARCHAR(255) UNIQUE,
        genre VARCHAR(50),
        popularity INTEGER,
        danceability FLOAT,
        energy FLOAT,
        loudness FLOAT,
        speechiness FLOAT,
        acousticness FLOAT,
        instrumentalness FLOAT,
        valence FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_genre ON {table_name}(genre);
    CREATE INDEX IF NOT EXISTS idx_track_id ON {table_name}(track_id);
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                conn.commit()
                logger.info(f"Table '{table_name}' created/verified successfully")
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        raise


def truncate_table(table_name: str = "spotify_songs"):
    """
    Truncate the table to remove existing data.

    Args:
        table_name: Name of the table to truncate
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
                conn.commit()
                logger.info(f"Table '{table_name}' truncated")
    except Exception as e:
        logger.error(f"Error truncating table: {e}")
        raise


def load_csv_to_database(
    csv_path: str, table_name: str = "spotify_songs", truncate: bool = False
):
    """
    Load CSV data into PostgreSQL database.

    Args:
        csv_path: Path to the CSV file
        table_name: Name of the table to insert into
        truncate: If True, truncate table before inserting
    """
    print("=" * 60)
    print("Loading CSV Data into PostgreSQL Database")
    print("=" * 60)

    # Test connection first
    print("\n1. Testing database connection...")
    if not test_connection():
        print("❌ Database connection failed. Please check your configuration.")
        return False

    # Read CSV
    print(f"\n2. Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return False
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    # Create table
    print(f"\n3. Creating/verifying table '{table_name}'...")
    try:
        create_table_if_not_exists(table_name)
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return False

    # Truncate if requested
    if truncate:
        print(f"\n4. Truncating table '{table_name}'...")
        try:
            truncate_table(table_name)
        except Exception as e:
            print(f"❌ Error truncating table: {e}")
            return False
    else:
        print(f"\n4. Checking existing data in '{table_name}'...")
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                    existing_count = cur.fetchone()[0]
                    print(f"   Found {existing_count} existing rows")
                    if existing_count > 0:
                        response = input(
                            f"   ⚠️  Table already has data. Truncate and reload? (y/n): "
                        )
                        if response.lower() == "y":
                            truncate_table(table_name)
                        else:
                            print("   Skipping truncate - will append data")
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}")

    # Prepare data for insertion
    print(f"\n5. Preparing data for insertion...")
    # Replace NaN values with None for PostgreSQL
    df = df.where(pd.notnull(df), None)

    # Convert instrumentalness from scientific notation if needed
    if "instrumentalness" in df.columns:
        df["instrumentalness"] = pd.to_numeric(df["instrumentalness"], errors="coerce")

    # Insert data using pandas to_sql (optimized for bulk inserts)
    print(f"\n6. Inserting {len(df)} rows into '{table_name}'...")

    try:
        from sqlalchemy import create_engine
        from api.db_config import DatabaseConfig

        config = DatabaseConfig()
        conn_string = config.get_connection_string()

        # Create SQLAlchemy engine
        engine = create_engine(conn_string)

        # Use pandas to_sql - much faster than row-by-row inserts
        print("   Loading data (this may take a moment)...")
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",  # Always append, truncate was handled earlier
            index=False,
            method="multi",  # Use multi-row insert for speed
            chunksize=1000,  # Insert in chunks of 1000 rows
        )

        print(f"   ✅ Successfully inserted {len(df)} rows")
        engine.dispose()

    except ImportError:
        # Fallback to batch insert if SQLAlchemy not available
        print("   SQLAlchemy not available, using batch insert...")
        insert_query = f"""
        INSERT INTO {table_name} (
            artist_name, track_name, track_id, genre, popularity,
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, valence
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (track_id) DO UPDATE SET
            artist_name = EXCLUDED.artist_name,
            track_name = EXCLUDED.track_name,
            genre = EXCLUDED.genre,
            popularity = EXCLUDED.popularity,
            danceability = EXCLUDED.danceability,
            energy = EXCLUDED.energy,
            loudness = EXCLUDED.loudness,
            speechiness = EXCLUDED.speechiness,
            acousticness = EXCLUDED.acousticness,
            instrumentalness = EXCLUDED.instrumentalness,
            valence = EXCLUDED.valence;
        """

        batch_size = 1000
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i : i + batch_size]
                    data_tuples = [
                        (
                            row["artist_name"],
                            row["track_name"],
                            row["track_id"],
                            row["genre"],
                            (
                                int(row["popularity"])
                                if pd.notna(row["popularity"])
                                else None
                            ),
                            (
                                float(row["danceability"])
                                if pd.notna(row["danceability"])
                                else None
                            ),
                            float(row["energy"]) if pd.notna(row["energy"]) else None,
                            (
                                float(row["loudness"])
                                if pd.notna(row["loudness"])
                                else None
                            ),
                            (
                                float(row["speechiness"])
                                if pd.notna(row["speechiness"])
                                else None
                            ),
                            (
                                float(row["acousticness"])
                                if pd.notna(row["acousticness"])
                                else None
                            ),
                            (
                                float(row["instrumentalness"])
                                if pd.notna(row["instrumentalness"])
                                else None
                            ),
                            float(row["valence"]) if pd.notna(row["valence"]) else None,
                        )
                        for _, row in batch.iterrows()
                    ]
                    cur.executemany(insert_query, data_tuples)
                    conn.commit()
                    print(
                        f"   Inserted batch {i//batch_size + 1} ({min(i+batch_size, len(df))}/{len(df)} rows)"
                    )

        print(f"   ✅ Successfully inserted/updated {len(df)} rows")

    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        print(f"❌ Error inserting data: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Verify insertion
    print(f"\n7. Verifying data in '{table_name}'...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                total_count = cur.fetchone()[0]
                print(f"   ✅ Total rows in database: {total_count}")

                # Show genre distribution
                cur.execute(
                    f"""
                    SELECT genre, COUNT(*) as count 
                    FROM {table_name} 
                    GROUP BY genre 
                    ORDER BY count DESC;
                """
                )
                genre_counts = cur.fetchall()
                print(f"\n   Genre distribution:")
                for genre, count in genre_counts:
                    print(f"     {genre}: {count}")

    except Exception as e:
        logger.warning(f"Could not verify data: {e}")

    print("\n" + "=" * 60)
    print("✅ Data loading completed successfully!")
    print("=" * 60)
    return True


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load CSV data into PostgreSQL database"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/spotify_data_reduced.csv",
        help="Path to the CSV file (default: data/spotify_data_reduced.csv)",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="spotify_songs",
        help="Name of the database table (default: spotify_songs)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate table before inserting (default: False)",
    )

    args = parser.parse_args()

    # Resolve CSV path relative to project root
    csv_path = Path(__file__).parent.parent.parent / args.csv_path

    load_csv_to_database(str(csv_path), args.table_name, args.truncate)


if __name__ == "__main__":
    main()
