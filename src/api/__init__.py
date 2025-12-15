"""
API package for database connections and utilities.
All credentials are read from environment variables - NEVER hardcoded.
"""

from src.api.db_config import DatabaseConfig
from src.api.db_connection import (
    get_db_connection,
    test_connection,
    execute_query,
    execute_query_dict,
    initialize_connection_pool,
    close_connection_pool,
)

__all__ = [
    "DatabaseConfig",
    "get_db_connection",
    "test_connection",
    "execute_query",
    "execute_query_dict",
    "initialize_connection_pool",
    "close_connection_pool",
]
