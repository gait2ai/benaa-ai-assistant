#!/usr/bin/env python3
# database.py - Module for handling SQLite database operations for conversation storage

import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict

# Import database path from config
from config.config import DB_PATH

# Basic logging configuration in case it's not set elsewhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Type alias for message dictionary
class MessageDict(TypedDict):
    id: int
    session_id: str
    role: str
    message: str
    timestamp: str

def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.
    Ensures the parent directory exists.
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    # Ensure the directory for the database exists
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created directory for database: {db_dir}")
    
    # Connect to the database with row factory for dict-like access
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    Initialize the database by creating the messages table if it doesn't exist.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create messages table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Create index on session_id for faster retrieval
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        conn.close()

def save_message(session_id: str, role: str, message: str, timestamp: Optional[str] = None) -> int:
    """
    Save a message to the database.
    
    Args:
        session_id (str): Unique identifier for the conversation session
        role (str): Either 'user' or 'assistant'
        message (str): The content of the message
        timestamp (str, optional): ISO formatted timestamp. If None, current time will be used.
    
    Returns:
        int: ID of the inserted message
    
    Raises:
        ValueError: If role is not 'user' or 'assistant' or if input validation fails
        sqlite3.Error: If there's a database error
    """
    # Input validation
    if not session_id or session_id.strip() == "":
        raise ValueError("session_id cannot be empty")
    
    if not message or message.strip() == "":
        raise ValueError("message cannot be empty")
        
    if role not in ('user', 'assistant'):
        raise ValueError("Role must be either 'user' or 'assistant'")
    
    # Use current time if no timestamp provided
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO messages (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, message, timestamp)
        )
        
        conn.commit()
        message_id = cursor.lastrowid
        logger.debug(f"Saved {role} message for session {session_id}")
        return message_id
    except sqlite3.Error as e:
        logger.error(f"Error saving message: {e}")
        raise
    finally:
        conn.close()

def get_messages(session_id: str) -> List[MessageDict]:
    """
    Retrieve all messages for a given session ID, ordered by timestamp.
    
    Args:
        session_id (str): Unique identifier for the conversation session
    
    Returns:
        List[MessageDict]: List of message dictionaries with keys:
                          id, session_id, role, message, timestamp
    
    Raises:
        ValueError: If session_id is empty
        sqlite3.Error: If there's a database error
    """
    # Input validation
    if not session_id or session_id.strip() == "":
        raise ValueError("session_id cannot be empty")
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, session_id, role, message, timestamp FROM messages "
            "WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        
        # Convert row objects to dictionaries
        messages = [dict(row) for row in cursor.fetchall()]
        logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
        return messages
    except sqlite3.Error as e:
        logger.error(f"Error retrieving messages: {e}")
        raise
    finally:
        conn.close()

def delete_session(session_id: str) -> int:
    """
    Delete all messages for a given session ID.
    
    Args:
        session_id (str): Unique identifier for the conversation session
    
    Returns:
        int: Number of deleted messages
    
    Raises:
        ValueError: If session_id is empty
        sqlite3.Error: If there's a database error
    """
    # Input validation
    if not session_id or session_id.strip() == "":
        raise ValueError("session_id cannot be empty")
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted_count} messages for session {session_id}")
        return deleted_count
    except sqlite3.Error as e:
        logger.error(f"Error deleting session: {e}")
        raise
    finally:
        conn.close()

def get_all_sessions() -> List[str]:
    """
    Get a list of all unique session IDs in the database.
    
    Returns:
        List[str]: List of unique session IDs
    
    Raises:
        sqlite3.Error: If there's a database error
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT session_id FROM messages")
        
        # Extract session IDs from result rows
        sessions = [row['session_id'] for row in cursor.fetchall()]
        return sessions
    except sqlite3.Error as e:
        logger.error(f"Error retrieving sessions: {e}")
        raise
    finally:
        conn.close()

# Execute init_db() only when the module is run directly
if __name__ == "__main__":
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Database initialization complete.")
