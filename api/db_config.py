"""
Database configuration for PostgreSQL connection.
All credentials are read from environment variables - NEVER hardcoded.
"""
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip loading .env file


class DatabaseConfig:
    """Configuration class for database connection parameters."""
    
    def __init__(self):
        """Initialize database configuration from connection string or environment variables."""
        # Check if full connection string is provided
        connection_string = os.getenv("DATABASE_URL") or os.getenv("DB_CONNECTION_STRING")
        
        if connection_string:
            # Parse connection string
            self._parse_connection_string(connection_string)
        else:
            # Use individual environment variables or defaults
            self.host: str = os.getenv("DB_HOST", "localhost")
            self.port: int = int(os.getenv("DB_PORT", "5432"))
            self.database: str = os.getenv("DB_NAME", "Ethos")
            self.user: str = os.getenv("DB_USER", "postgres")
            self.password: str = os.getenv("DB_PASSWORD", "")
            self.sslmode: str = os.getenv("DB_SSLMODE", "prefer")
            self.channel_binding: Optional[str] = os.getenv("DB_CHANNEL_BINDING")
    
    def _parse_connection_string(self, conn_str: str):
        """Parse a PostgreSQL connection string."""
        # Remove 'psql ' prefix if present
        conn_str = conn_str.replace("psql '", "").replace("'", "")
        
        parsed = urlparse(conn_str)
        
        self.user = parsed.username or "postgres"
        self.password = parsed.password or ""
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 5432
        self.database = parsed.path.lstrip("/") if parsed.path else "postgres"
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        self.sslmode = query_params.get("sslmode", ["prefer"])[0]
        self.channel_binding = query_params.get("channel_binding", [None])[0]
        
        # Extract Neon endpoint ID from hostname if it's a Neon direct connection (not pooler)
        if "neon.tech" in self.host:
            if "-pooler" in self.host:
                # Pooler endpoints don't need endpoint option - set to None
                self.endpoint_id = None
                self.is_pooler = True
            else:
                # Direct connection - extract endpoint ID
                # Format: ep-xxx-xxx-xxxxx.region.aws.neon.tech
                parts = self.host.split(".")
                if parts and parts[0].startswith("ep-"):
                    self.endpoint_id = parts[0]
                    self.is_pooler = False
                else:
                    self.endpoint_id = None
                    self.is_pooler = False
        else:
            self.endpoint_id = None
            self.is_pooler = False
        
    def get_connection_string(self, include_channel_binding: bool = False) -> str:
        """
        Returns a PostgreSQL connection string.
        
        Args:
            include_channel_binding: If True, includes channel_binding parameter (not supported by psycopg2)
        
        Returns:
            str: Connection string in format: postgresql://user:password@host:port/database?sslmode=require&options=endpoint%3D...
        """
        conn_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        
        # Add SSL parameters if needed
        params = []
        if hasattr(self, 'sslmode') and self.sslmode:
            params.append(f"sslmode={self.sslmode}")
        # channel_binding is not supported by psycopg2, so we exclude it by default
        if include_channel_binding and hasattr(self, 'channel_binding') and self.channel_binding:
            params.append(f"channel_binding={self.channel_binding}")
        
        # Add Neon endpoint ID only for direct connections (not pooler)
        # Pooler endpoints handle SNI automatically and don't need endpoint option
        if hasattr(self, 'endpoint_id') and self.endpoint_id and not getattr(self, 'is_pooler', False):
            from urllib.parse import quote
            endpoint_param = f"options=endpoint%3D{quote(self.endpoint_id)}"
            params.append(endpoint_param)
        
        if params:
            conn_str += "?" + "&".join(params)
        
        return conn_str
    
    def get_psycopg2_params(self) -> dict:
        """
        Returns connection parameters as a dictionary for psycopg2.
        
        Returns:
            dict: Dictionary with connection parameters including SSL settings
        """
        params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }
        
        # Add SSL parameters if they exist
        if hasattr(self, 'sslmode') and self.sslmode:
            params["sslmode"] = self.sslmode
        if hasattr(self, 'channel_binding') and self.channel_binding:
            params["channel_binding"] = self.channel_binding
        
        return params
    
    def __repr__(self) -> str:
        """String representation (without password for security)."""
        return (
            f"DatabaseConfig(host='{self.host}', port={self.port}, "
            f"database='{self.database}', user='{self.user}')"
        )

