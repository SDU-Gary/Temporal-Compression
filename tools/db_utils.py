"""Database connection and initialization utilities."""

import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

# Database path (project root)
DB_PATH = Path(__file__).parent.parent / "project.db"


@contextmanager
def get_db():
    """
    Context manager for database connection.

    Usage:
        with get_db() as conn:
            conn.execute("INSERT INTO ...")

    Features:
    - Automatic commit on success
    - Automatic rollback on error
    - Row factory for dict-like access
    - Foreign keys enabled
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access: row['column_name']
    conn.execute("PRAGMA foreign_keys = ON")  # Enforce foreign key constraints
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db(schema_path: Optional[Path] = None):
    """
    Initialize database from schema.sql.

    Args:
        schema_path: Path to schema.sql. If None, uses ../schema.sql

    Returns:
        Number of tables created

    Raises:
        FileNotFoundError: If schema.sql doesn't exist
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with get_db() as conn:
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())

        # Count tables
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        )
        table_count = cursor.fetchone()[0]

    print(f"✅ Database initialized at {DB_PATH}")
    print(f"   Tables created: {table_count}")
    return table_count


def verify_db():
    """
    Verify database structure and return stats.

    Returns:
        dict with table names and row counts
    """
    with get_db() as conn:
        # Get all tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Count rows in each table
        stats = {}
        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[table] = count

    return stats


def get_next_run_order():
    """
    Get next sequential run_order for experiments.

    Returns:
        Next run_order (starts from 1)
    """
    with get_db() as conn:
        cursor = conn.execute("SELECT MAX(run_order) FROM experiments")
        max_order = cursor.fetchone()[0]
        return 1 if max_order is None else max_order + 1


def get_next_exp_id(date: Optional[str] = None):
    """
    Generate next experiment ID for a given date.

    Args:
        date: Date string in YYYYMMDD format. If None, uses today.

    Returns:
        Experiment ID in format EXP-YYYYMMDD-###
    """
    if date is None:
        date = time.strftime("%Y%m%d")

    with get_db() as conn:
        # Count existing experiments for this date
        cursor = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE exp_id LIKE ?",
            (f"EXP-{date}-%",)
        )
        count = cursor.fetchone()[0]

    # Format: EXP-20260104-001
    return f"EXP-{date}-{count + 1:03d}"


def get_db_last_modified():
    """
    Get last modification time of database.

    Returns:
        Unix timestamp or None if DB doesn't exist
    """
    if not DB_PATH.exists():
        return None
    return int(DB_PATH.stat().st_mtime)


if __name__ == "__main__":
    # Test database initialization
    print("Initializing database...")
    init_db()

    print("\nVerifying database structure...")
    stats = verify_db()
    for table, count in stats.items():
        print(f"  {table}: {count} rows")

    print("\nDatabase ready!")
