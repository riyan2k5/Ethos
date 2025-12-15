"""
Database connection utilities for PostgreSQL.
All credentials are read from environment variables via DatabaseConfig.
"""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Generator
import logging

from src.api.db_config import DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def initialize_connection_pool(min_conn: int = 1, max_conn: int = 10):
    """
    Initialize a connection pool for database connections.
    
    Args:
        min_conn: Minimum number of connections in the pool
        max_conn: Maximum number of connections in the pool
    """
    global _connection_pool
    
    if _connection_pool is not None:
        logger.info("Connection pool already initialized")
        return
    
    try:
        config = DatabaseConfig()
        
        # Use connection string directly if SSL parameters are present (for Neon, etc.)
        if hasattr(config, 'sslmode') and config.sslmode:
            conn_string = config.get_connection_string()
            _connection_pool = pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                conn_string
            )
        else:
            # Use individual parameters for standard connections
            _connection_pool = pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                **config.get_psycopg2_params()
            )
        logger.info(f"Connection pool initialized successfully for database '{config.database}'")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise


def close_connection_pool():
    """Close all connections in the pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Connection pool closed")


@contextmanager
def get_db_connection(use_dict_cursor: bool = False) -> Generator:
    """
    Context manager for database connections.
    
    Args:
        use_dict_cursor: If True, returns a RealDictCursor (returns rows as dicts)
    
    Yields:
        psycopg2.connection: Database connection object
    
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()
    """
    global _connection_pool
    config = DatabaseConfig()
    
    # For SSL connections (like Neon), use direct connection instead of pool
    if hasattr(config, 'sslmode') and config.sslmode:
        conn = None
        try:
            conn = psycopg2.connect(config.get_connection_string())
            if use_dict_cursor:
                conn.cursor_factory = RealDictCursor
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    else:
        # Use connection pool for standard connections
        if _connection_pool is None:
            initialize_connection_pool()
        
        conn = None
        try:
            conn = _connection_pool.getconn()
            if use_dict_cursor:
                conn.cursor_factory = RealDictCursor
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                _connection_pool.putconn(conn)


def test_connection() -> bool:
    """
    Test the database connection.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        config = DatabaseConfig()
        
        # For Neon databases, always use connection string to ensure endpoint ID is included
        if "neon.tech" in config.host:
            import psycopg2
            conn_string = config.get_connection_string()
            logger.debug(f"Connecting to Neon database with connection string (endpoint ID: {getattr(config, 'endpoint_id', 'None')})")
            conn = psycopg2.connect(conn_string)
        elif hasattr(config, 'sslmode') and config.sslmode:
            import psycopg2
            conn = psycopg2.connect(config.get_connection_string())
        else:
            conn = psycopg2.connect(**config.get_psycopg2_params())
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logger.info(f"Database connection successful. PostgreSQL version: {version[0]}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def execute_query(query: str, params: Optional[tuple] = None, fetch: bool = True) -> Optional[list]:
    """
    Execute a SQL query and return results.
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters for parameterized queries
        fetch: If True, fetch and return results. If False, just execute.
    
    Returns:
        list: Query results if fetch=True, None otherwise
    
    Example:
        results = execute_query("SELECT * FROM songs WHERE genre = %s", ("rock",))
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                return None
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        logger.error(f"Query: {query}")
        raise


def execute_query_dict(query: str, params: Optional[tuple] = None) -> list:
    """
    Execute a SQL query and return results as dictionaries.
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters for parameterized queries
    
    Returns:
        list: List of dictionaries, each representing a row
    
    Example:
        results = execute_query_dict("SELECT * FROM songs WHERE genre = %s", ("rock",))
        # Returns: [{'id': 1, 'name': 'Song Name', 'genre': 'rock'}, ...]
    """
    try:
        with get_db_connection(use_dict_cursor=True) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        logger.error(f"Query: {query}")
        raise
