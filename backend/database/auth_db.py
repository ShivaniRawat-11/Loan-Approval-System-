import sqlite3
import os
import hashlib
import json
import logging

_DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_data")
DB_FILE = os.environ.get("DB_PATH", os.path.join(_DEFAULT_DB_DIR, "loan_portal.db"))
logger = logging.getLogger("auth_db")

def init_db():
    """Initialize the SQLite database and create tables if they do not exist."""
    try:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash BLOB NOT NULL,
                name TEXT NOT NULL
            )
        """)
        
        # Create chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                email TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL,
                FOREIGN KEY (email) REFERENCES users (email)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {DB_FILE}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def hash_password(password: str, salt: bytes = None) -> bytes:
    """Hash a password using pbkdf2_hmac with a random salt."""
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + pwd_hash

def verify_password(stored_password: bytes, provided_password: str) -> bool:
    """Verify a provided password against the stored hash."""
    salt = stored_password[:16]
    stored_hash = stored_password[16:]
    pwd_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return pwd_hash == stored_hash

def create_user(email: str, password: str, name: str = None) -> tuple:
    """Create a new user. Returns (success, message)."""
    email = email.strip().lower()
    if not name:
        name = email.split("@")[0].capitalize()
        
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered."
            
        hashed_pw = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email, hashed_pw, name)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False, f"Database error: {e}"

def authenticate_user(email: str, password: str) -> tuple:
    """Authenticate a user. Returns (success, name, message)."""
    email = email.strip().lower()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, name FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            stored_password, name = result
            if verify_password(stored_password, password):
                return True, name, "Login successful."
            else:
                return False, None, "Invalid password."
        else:
            return False, None, "Email not found."
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return False, None, f"Database error: {e}"

def get_chat_history(email: str) -> list:
    """Retrieve the user's chat messages as a list of dicts."""
    email = email.strip().lower()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT messages_json FROM chats WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return []
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

def save_chat_history(email: str, messages: list) -> bool:
    """Save the user's chat messages (list of dicts). Overwrites existing."""
    email = email.strip().lower()
    try:
        messages_json = json.dumps(messages)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Upsert logic
        cursor.execute("""
            INSERT INTO chats (email, messages_json)
            VALUES (?, ?)
            ON CONFLICT(email) DO UPDATE SET messages_json=excluded.messages_json
        """, (email, messages_json))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        return False

def clear_chat_history(email: str) -> bool:
    """Delete the user's chat history."""
    email = email.strip().lower()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE email = ?", (email,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False
