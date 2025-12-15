"""
Test script to verify database connection.
Run this to test your PostgreSQL connection.
All credentials are read from environment variables.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_connection import test_connection, execute_query_dict
from api.db_config import DatabaseConfig


def main():
    """Test database connection and display basic information."""
    print("=" * 60)
    print("Testing PostgreSQL Database Connection")
    print("=" * 60)
    
    # Display configuration (without password)
    config = DatabaseConfig()
    print(f"\nDatabase Configuration:")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Database: {config.database}")
    print(f"  User: {config.user}")
    print()
    
    # Test connection
    print("Attempting to connect...")
    if test_connection():
        print("\n✅ Connection successful!")
        
        # Try to list tables
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """
            tables = execute_query_dict(query)
            
            if tables:
                print(f"\n📊 Found {len(tables)} table(s) in database:")
                for table in tables:
                    print(f"  - {table['table_name']}")
            else:
                print("\n📊 No tables found in the database.")
        except Exception as e:
            print(f"\n⚠️  Could not list tables: {e}")
    else:
        print("\n❌ Connection failed!")
        print("\nPlease check:")
        print("  1. PostgreSQL is running")
        print("  2. Database exists")
        print("  3. Credentials are correct")
        print("  4. Environment variables are set")
        print("\nYou can set environment variables:")
        print("  export DB_HOST=localhost")
        print("  export DB_PORT=5432")
        print("  export DB_NAME=your_database")
        print("  export DB_USER=your_user")
        print("  export DB_PASSWORD=your_password")
        print("\nOr set DATABASE_URL:")
        print("  export DATABASE_URL='postgresql://user:password@host:port/database'")
        print("\nOr create a .env file with these variables.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

