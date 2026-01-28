"""
Module for managing PostgreSQL database operations.
- This module provides functions to track user uploaded / generated files / data.
- It includes creating tables, adding files and embeddings, and deleting the database.
- Each table has col 'available' to mark if the record is still valid or has been deleted.
"""

import bcrypt
import os
import psycopg2
from typing import List

import pytz
from datetime import datetime, timedelta

from src.core.logger import get_logger
log = get_logger(name="core_database")
CST = pytz.timezone('America/Chicago')

# PostgreSQL connection parameters
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'chat_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
}


# ------------------------------------------------------------------------------
# Database Management Functions:
# ------------------------------------------------------------------------------

def get_connection():
    """Creates and returns a PostgreSQL database connection.
    - The connection is set to use the database configured via environment variables.
    """

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        log.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def create_tables():
    """Creates the necessary tables in the PostgreSQL database.
    - The `users` table tracks registered users.
    - The `uploads` table tracks user uploaded files.
    - The `embeddings` table tracks embeddings associated with those files' chunks.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # USERS(user_id*, name, password_hash, last_login)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    last_login TEXT
                )
            """)

            # UPLOADS(file_id*, user_id^, filename, created_at, file_size, file_type, chunk_count, updated_at, file_modified_at, source_created_at, available, source, embedding_status)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    file_id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_size INTEGER,
                    file_type TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    updated_at TEXT,
                    file_modified_at TEXT,
                    source_created_at TEXT,
                    available INTEGER DEFAULT 1,
                    source TEXT DEFAULT 'manual',
                    embedding_status TEXT DEFAULT 'pending',
                    embedding_model TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)

            # EMBEDDINGS(id*, file_id^, vector_id, available)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    vector_id TEXT NOT NULL,
                    available INTEGER DEFAULT 1,
                    FOREIGN KEY (file_id) REFERENCES uploads(file_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            log.info("Database tables created successfully.")
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while creating tables: {e}")
        raise


# ------------------------------------------------------------------------------
# User Management Functions:
# ------------------------------------------------------------------------------

def add_user(user_id: str, name: str, password: str) -> bool:
    """Adds a new user to the database.

    Args:
        user_id (str): The unique ID of the user.
        name (str): The name of the user.
        password (str): The raw password of the user.

    Returns:
        bool: True if the user was added successfully, False otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cst_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            cur.execute("""
                INSERT INTO users (user_id, name, password_hash, last_login)
                VALUES (%s, %s, %s, %s)
            """, (user_id, name, hashed_password.decode('utf-8'), cst_time))
            conn.commit()

            if cur.rowcount == 0:
                log.error(f"Failed to insert user '{name}' with ID '{user_id}'")
                return False

            log.info(f"User '{name}' added with ID '{user_id}'")
            return True
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while adding user '{name}' with ID '{user_id}': {e}")
        return False


def check_user_exists(user_id: str) -> bool:
    """Checks if a user exists in the database.

    Args:
        user_id (str): The ID of the user to check.

    Returns:
        bool: True if the user exists, False otherwise.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            if result:
                log.info(f"User '{user_id}' exists in the database.")
                return True
            else:
                log.warning(f"User '{user_id}' does not exist in the database.")
                return False
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while checking user '{user_id}': {e}")
        return False


def authenticate_user(user_id: str, password: str) -> tuple[bool, str]:
    """Authenticates a user by checking their ID and password hash.
    + If the user exists and the password matches, it updates the last login time.

    Args:
        user_id (str): The ID of the user to authenticate.
        password (str): The raw password of the user.

    Returns:
        tuple (bool, str):
            + bool: True if the authentication is successful, False otherwise.
            + str: A message indicating the error or The user-name if authentication is successful.
    """
    try:
        with get_connection() as conn:
            # Fetch the hashed password from the database
            cur = conn.cursor()
            cur.execute("SELECT password_hash, name FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()

            if result:
                hashed_password = result[0].encode('utf-8')

                # Check if the provided password matches the hashed password
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    log.info(f"User '{user_id}' authenticated successfully.")
                    # Update last login time
                    cst_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")

                    cur.execute("""
                        UPDATE users SET last_login = %s
                        WHERE user_id = %s
                    """, (cst_time, user_id))
                    conn.commit()
                    log.info(f"Authentication successful for '{user_id}' @ {cst_time}")
                    return True, result[1]  # Return True and the user's name

                else:
                    log.warning(f"Authentication failed for user '{user_id}': Incorrect password.")
                    return False, "Incorrect password."

            else:
                log.warning(f"Authentication failed for user '{user_id}': User does not exist.")
                return False, "User does not exist."

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while authenticating user '{user_id}': {e}")
        return False, "Database error during authentication."


# ------------------------------------------------------------------------------
# File Management Functions:
# ------------------------------------------------------------------------------

def add_file(user_id: str, filename: str, file_size: int = None, file_type: str = None, file_path: str = None, source_created_at: str = None, source: str = 'manual') -> int:
    """Adds a file upload record to the database.

    Args:
        user_id (str): The ID of the user uploading the file.
        filename (str): The name of the file being uploaded.
        file_size (int): Size of the file in bytes (optional).
        file_type (str): MIME type of the file (optional).
        file_path (str): Path to the file to extract modification time (optional).
        source_created_at (str): The original document creation date from source metadata (optional).
        source (str): Source of the file - 'manual', 'google_drive', 'local', etc. Defaults to 'manual'.
    Returns:
        int: The ID (=file_id) of the newly created file record, or -1 if an error occurred.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cst_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
            
            # Get file modification time if path provided
            file_modified_at = None
            if file_path and os.path.exists(file_path):
                try:
                    mod_timestamp = os.path.getmtime(file_path)
                    file_modified_at = datetime.fromtimestamp(mod_timestamp, tz=CST).strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    log.warning(f"Could not get modification time for {file_path}: {e}")

            cur.execute(
                "INSERT INTO uploads (user_id, filename, created_at, file_size, file_type, updated_at, file_modified_at, source_created_at, source) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING file_id",
                (user_id, filename, cst_time, file_size, file_type, cst_time, file_modified_at, source_created_at, source)
            )
            file_id = cur.fetchone()[0]
            conn.commit()

            if file_id is None:
                log.error(f"Failed to insert file '{filename}' for '{user_id}', no ID was returned")
                return -1

            log.info(f"File '{filename}' added for user '{user_id}' with ID {file_id}")
            return file_id

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while adding file '{filename}' for user '{user_id}': {e}")
        return -1


def get_user_files(user_id: str) -> List[str]:
    """Retrieves all files uploaded by a user.
    - Only retrieves files that are marked as available (not deleted).

    Args:
        user_id (str): The ID of the user whose files are being queried.

    Returns:
        List[str]: A list of file names uploaded by the user.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT filename FROM uploads
                WHERE user_id = %s AND available = 1
            """, (user_id,))
            files = cur.fetchall()

            log.info(f"Retrieved {len(files)} files for user '{user_id}'")
            return [file[0] for file in files]

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while retrieving files for user '{user_id}': {e}")
        return []


def get_old_files(user_id: str, time: int = 12*3600) -> dict[str, List[str]]:
    """Retrieves files uploaded by a user that are older than a specified time.

    Args:
        user_id (str): The ID of the user whose files are being queried.
        time (int): The time in seconds to consider a file as old. Default is 12 hours.
                   If time=1, retrieves ALL files (used for clearing all uploads).

    Returns:
        dict[str, List[str]]: A dictionary with keys [files, embeddings] and values as lists of file names and embedding vector IDs respectively.    
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # If time=1, clear all files (ignore cutoff_time logic)
            if time <= 1:
                # Get ALL files for the user (both available and unavailable)
                # This ensures we clean up partially deleted files from previous runs
                cur.execute("""
                    SELECT filename, file_id FROM uploads
                    WHERE user_id = %s
                """, (user_id,))
            else:
                # Calc cutoff time as string in CST
                cutoff_time = datetime.now(CST) - timedelta(seconds=time)
                threshold_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Get files older than cutoff_time that are still marked as available
                cur.execute("""
                    SELECT filename, file_id FROM uploads
                    WHERE user_id = %s AND available = 1 AND created_at < %s
                """, (user_id, threshold_str))
            
            files = cur.fetchall()

            # Get embeddings for those files (both available and unavailable)
            # This ensures we clean up all embeddings even if partially deleted
            file_ids = [file[1] for file in files]
            if not file_ids:
                return {"files": [], "embeddings": []}

            placeholders = ', '.join(['%s'] * len(file_ids))
            cur.execute(f"""
                SELECT vector_id FROM embeddings
                WHERE file_id IN ({placeholders})
            """, file_ids)
            embeddings = cur.fetchall()

            log.info(
                f"Retrieved {len(files)} old files and {len(embeddings)} embeddings for user '{user_id}'")
            return {
                "files": [file[0] for file in files],
                "embeddings": [embedding[0] for embedding in embeddings]
            }

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while retrieving old files for user '{user_id}': {e}")
        return {"files": [], "embeddings": []}


def get_file_id_by_name(user_id: str, file_name: str) -> int:
    """Retrieves the file ID for a given file name and user ID.
    - Checks only the active files and not old or deleted files.

    Args:
        user_id (str): The ID of the user who owns the file.
        file_name (str): The name of the file to search for.

    Returns:
        int: The ID of the file if found, -1 otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT file_id FROM uploads
                WHERE user_id = %s AND filename = %s AND available = 1
            """, (user_id, file_name))
            result = cur.fetchone()

            if result:
                log.info(f"File '{file_name}' resolved ID={result[0]} for user '{user_id}'")
                return result[0]
            else:
                log.warning(f"File '{file_name}' not found to resolve ID for user '{user_id}'")
                return -1

    except psycopg2.Error as e:
        log.error(
            f"PostgreSQL error while retrieving file ID for '{file_name}' and user '{user_id}': {e}")
        return -1


def get_uploads(user_id: str) -> dict:
    """Get all uploaded files for a user.
    
    Args:
        user_id (str): The ID of the user.
    
    Returns:
        dict: Dictionary with 'files' list containing file information.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT filename, created_at, available, embedding_status, embedding_model, file_size, chunk_count
                FROM uploads
                WHERE user_id = %s AND available = 1
                ORDER BY created_at DESC
            """, (user_id,))
            rows = cur.fetchall()
            
            files_list = []
            for row in rows:
                files_list.append({
                    'filename': row[0],
                    'created_at': row[1] if isinstance(row[1], str) else (row[1].isoformat() if row[1] else None),
                    'available': row[2],
                    'embedding_status': row[3],
                    'embedding_model': row[4],
                    'size': row[5],
                    'chunk_count': row[6]
                })
            
            return {'files': files_list}
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while retrieving uploads for user '{user_id}': {e}")
        return {'files': []}


def mark_file_removed(user_id: str, file_id: int) -> bool:
    """Marks a file as unavailable (deleted) in the database.

    Args:
        user_id (str): The ID of the user who owns the file.
        file_id (int): The ID of the file to be marked as unavailable.

    Returns:
        bool: True if the file was marked as unavailable, False otherwise.

    ## `Warning:`
        - This function does not physically delete any file or entry
        - It is just useful for managing / tracking user data
        - Use this to:
            1. Get the embedding doc ids and then use them to delete in Qdrant
            2. Get file names and delete files using files module
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE uploads SET available = 0
                WHERE user_id = %s AND file_id = %s
            """, (user_id, file_id))
            conn.commit()

            if cur.rowcount == 0:
                log.warning(f"No file found for user '{user_id}' with ID {file_id}")
                return False

            log.info(f"File ID {file_id} marked as deleted for user '{user_id}'")
            return True

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while removing file ID {file_id} for user '{user_id}': {e}")
        return False


# ------------------------------------------------------------------------------
# Embedding Management Functions:
# ------------------------------------------------------------------------------

def add_embedding(file_id: int, vector_id: str) -> bool:
    """Adds an embedding record for a file in the database.

    Args:
        file_id (int): The ID of the file to which the embedding belongs.
        vector_id (str): The ID of the embedding vector.

    Returns:
        bool: True if the embedding was added successfully, False otherwise.
    """

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO embeddings (file_id, vector_id) VALUES (%s, %s)",
                (file_id, vector_id,)
            )
            conn.commit()
            log.info(f"Embedding added for file ID {file_id} with vector ID '{vector_id}'")
            return True

    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while adding embedding for file ID {file_id}: {e}")
        return False


def mark_embeddings_removed(vector_ids: List[str]) -> bool:
    """Marks an embedding as unavailable (deleted) in the database.

    Args:
        vector_ids (List[str]): A list of vector IDs to be marked as unavailable.
    Returns:
        bool: True if the embeddings were marked as unavailable, False otherwise.
    """
    if not vector_ids:
        log.warning("No vector IDs provided to mark as removed.")
        return False

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            placeholders = ', '.join(['%s'] * len(vector_ids))
            log.info(f"Marking {len(vector_ids)} embeddings as removed: {vector_ids[:3]}...")
            cur.execute(f"""
                UPDATE embeddings SET available = 0
                WHERE vector_id IN ({placeholders})
            """, vector_ids)
            conn.commit()

            log.info(f"Updated {cur.rowcount} rows in embeddings table")
            if cur.rowcount != len(vector_ids):
                log.warning(f"Marked {cur.rowcount}/{len(vector_ids)} embeddings as removed.")
                return False
            else:
                log.info(f"Marked all ({len(vector_ids)}) embeddings as removed.")
                return True
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while marking embeddings as removed: {e}")
        return False


def get_embeddings_by_file_id(file_id: int) -> List[str]:
    """Get all vector IDs (embeddings) for a given file.

    Args:
        file_id (int): The file ID to get embeddings for.
    Returns:
        List[str]: A list of vector IDs for the file, or empty list if none found.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT vector_id FROM embeddings
                WHERE file_id = %s AND available = 1
            """, (file_id,))
            
            rows = cur.fetchall()
            if rows:
                log.info(f"Found {len(rows)} embeddings for file_id {file_id}")
                return [row[0] for row in rows]
            else:
                log.info(f"No embeddings found for file_id {file_id}")
                return []
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while getting embeddings for file {file_id}: {e}")
        return []


def update_file_chunk_count(file_id: int, chunk_count: int, user_id: str = None) -> bool:
    """Updates the chunk count and updated_at timestamp for a file.
    
    Args:
        file_id (int): The ID of the file to update.
        chunk_count (int): The number of chunks the file was split into.
        user_id (str): Optional user ID for logging.
    
    Returns:
        bool: True if update was successful, False otherwise.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cst_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
            
            cur.execute("""
                UPDATE uploads 
                SET chunk_count = %s, updated_at = %s
                WHERE file_id = %s
            """, (chunk_count, cst_time, file_id))
            conn.commit()
            
            if cur.rowcount > 0:
                user_info = f" for user '{user_id}'" if user_id else ""
                log.info(f"Updated file_id {file_id}{user_info} with chunk_count={chunk_count}")
                return True
            else:
                log.warning(f"No file found with file_id {file_id} to update chunk count")
                return False
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while updating chunk count for file {file_id}: {e}")
        return False


def get_embedding_status(file_id: int) -> str:
    """Gets the current embedding status for a file.
    
    Args:
        file_id (int): The ID of the file to check.
    
    Returns:
        str: Embedding status - 'pending', 'in_progress', 'completed', or 'failed'.
             Returns 'pending' if file not found.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT embedding_status FROM uploads
                WHERE file_id = %s
            """, (file_id,))
            result = cur.fetchone()
            
            if result:
                return result[0]
            else:
                log.warning(f"File ID {file_id} not found when checking embedding status")
                return 'pending'
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while getting embedding status for file {file_id}: {e}")
        return 'pending'


def set_embedding_status(file_id: int, status: str) -> bool:
    """Sets the embedding status for a file.
    
    Args:
        file_id (int): The ID of the file to update.
        status (str): Status to set - 'pending', 'in_progress', 'completed', 'failed'.
    
    Returns:
        bool: True if update was successful, False otherwise.
    """
    if status not in ['pending', 'in_progress', 'completed', 'failed']:
        log.error(f"Invalid embedding status: {status}")
        return False
    
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cst_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
            
            cur.execute("""
                UPDATE uploads
                SET embedding_status = %s, updated_at = %s
                WHERE file_id = %s
            """, (status, cst_time, file_id))
            conn.commit()
            
            if cur.rowcount > 0:
                log.info(f"Embedding status for file {file_id} set to '{status}'")
                return True
            else:
                log.warning(f"File ID {file_id} not found when setting embedding status")
                return False
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while setting embedding status for file {file_id}: {e}")
        return False


def clear_embeddings_for_file(file_id: int) -> bool:
    """Clears all embeddings for a file before re-ingestion.
    
    Args:
        file_id (int): The ID of the file whose embeddings to clear.
    
    Returns:
        bool: True if cleared successfully, False otherwise.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Get count before deletion
            cur.execute("SELECT COUNT(*) FROM embeddings WHERE file_id = %s", (file_id,))
            count = cur.fetchone()[0]
            
            # Delete all embeddings for this file
            cur.execute("""
                DELETE FROM embeddings WHERE file_id = %s
            """, (file_id,))
            conn.commit()
            
            log.info(f"Cleared {count} embeddings for file {file_id}")
            return True
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while clearing embeddings for file {file_id}: {e}")
        return False


def update_embedding_model(user_id: str, filename: str, embedding_model: str) -> bool:
    """Updates the embedding model for a specific file.

    Args:
        user_id (str): The ID of the user.
        filename (str): The name of the file.
        embedding_model (str): The name of the embedding model used.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE uploads
                SET embedding_model = %s
                WHERE user_id = %s AND filename = %s
            """, (embedding_model, user_id, filename))
            conn.commit()
            return True
    except psycopg2.Error as e:
        log.error(f"Error updating embedding model for {filename}: {e}")
        return False


def clear_all_embeddings() -> int:
    """Clears all embeddings from the database.
    
    Returns:
        int: Number of embeddings cleared.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Get count before deletion
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            
            # Delete all embeddings
            cur.execute("DELETE FROM embeddings")
            conn.commit()
            
            log.info(f"Cleared all {count} embeddings from database")
            return count
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while clearing all embeddings: {e}")
        return 0


def mark_all_files_for_reembedding() -> int:
    """Marks all files as 'pending' for re-embedding.
    
    Returns:
        int: Number of files marked.
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Update all files to pending status
            cur.execute("""
                UPDATE uploads
                SET embedding_status = 'pending'
                WHERE embedding_status IN ('completed', 'failed')
            """)
            count = cur.rowcount
            conn.commit()
            
            log.info(f"Marked {count} files for re-embedding")
            return count
    
    except psycopg2.Error as e:
        log.error(f"PostgreSQL error while marking files for re-embedding: {e}")
        return 0


if __name__ == "__main__":
    print("PostgreSQL Database Module Test:")
    print("\nCreating tables...")
    create_tables()
    print("\t - Database and tables created successfully.")
    print("\nPostgreSQL module ready for use.")
