"""
SQLite-based memory store for solved problems, OCR corrections, and user feedback.
"""
import sqlite3, json, datetime
from config.settings import SQLITE_DB_PATH

def get_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS solved_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT,
            original_input TEXT,
            parsed_problem TEXT,
            topic TEXT,
            solution TEXT,
            solution_steps TEXT,
            tools_used TEXT,
            verification_confidence REAL,
            is_correct BOOLEAN DEFAULT 1,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id INTEGER,
            feedback_type TEXT,
            comment TEXT,
            category TEXT,
            sentiment TEXT,
            is_actionable BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(problem_id) REFERENCES solved_problems(id)
        );
        
        CREATE TABLE IF NOT EXISTS hitl_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_agent TEXT,
            original_content TEXT,
            corrected_content TEXT,
            action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # ── Auto-Migration: Add missing columns if database was initialized with older schema ──
    try:
        # Check hitl_corrections
        cursor.execute("PRAGMA table_info(hitl_corrections)")
        hc_columns = [row[1] for row in cursor.fetchall()]
        if 'hitl_reason' not in hc_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'hitl_reason' to 'hitl_corrections'")
            cursor.execute("ALTER TABLE hitl_corrections ADD COLUMN hitl_reason TEXT DEFAULT ''")
        
        # Check user_feedback
        cursor.execute("PRAGMA table_info(user_feedback)")
        uf_columns = [row[1] for row in cursor.fetchall()]
        if 'category' not in uf_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'category' to 'user_feedback'")
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN category TEXT DEFAULT 'General'")
        if 'sentiment' not in uf_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'sentiment' to 'user_feedback'")
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN sentiment TEXT DEFAULT 'Neutral'")
        if 'is_actionable' not in uf_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'is_actionable' to 'user_feedback'")
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN is_actionable BOOLEAN DEFAULT 0")

        # Check solved_problems
        cursor.execute("PRAGMA table_info(solved_problems)")
        sp_columns = [row[1] for row in cursor.fetchall()]
        if 'is_correct' not in sp_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'is_correct' to 'solved_problems'")
            cursor.execute("ALTER TABLE solved_problems ADD COLUMN is_correct BOOLEAN DEFAULT 1")
        if 'explanation' not in sp_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'explanation' to 'solved_problems'")
            cursor.execute("ALTER TABLE solved_problems ADD COLUMN explanation TEXT DEFAULT ''")
        if 'verification_confidence' not in sp_columns:
            print("[DEBUG] 🔄 Migrating database: Adding 'verification_confidence' to 'solved_problems'")
            cursor.execute("ALTER TABLE solved_problems ADD COLUMN verification_confidence REAL DEFAULT 0")

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")

    conn.commit()
    conn.close()

def save_solved_problem(data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO solved_problems
        (input_type, original_input, parsed_problem, topic, solution, solution_steps, tools_used, verification_confidence, is_correct, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("input_type"), data.get("original_input"), 
        json.dumps(data.get("parsed_problem", {})),
        data.get("topic"), data.get("solution"),
        json.dumps(data.get("solution_steps", [])),
        json.dumps(data.get("tools_used", [])),
        data.get("verification_confidence", 0),
        data.get("is_correct", True),
        data.get("explanation")
    ))
    problem_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return problem_id

def save_feedback(problem_id: int, feedback_type: str, comment: str = "", category: str = "General", sentiment: str = "Neutral", is_actionable: bool = False):
    conn = get_connection()
    conn.execute("""
        INSERT INTO user_feedback 
        (problem_id, feedback_type, comment, category, sentiment, is_actionable) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (problem_id, feedback_type, comment, category, sentiment, 1 if is_actionable else 0))
    conn.commit()
    conn.close()

def save_hitl_correction(trigger_agent: str, reason: str, original: str, corrected: str, action: str):
    conn = get_connection()
    conn.execute("INSERT INTO hitl_corrections (trigger_agent, hitl_reason, original_content, corrected_content, action) VALUES (?, ?, ?, ?, ?)",
                 (trigger_agent, reason, original, corrected, action))
    conn.commit()
    conn.close()

def get_all_solved_problems() -> list[dict]:
    conn = get_connection()
    rows = conn.cursor().execute("SELECT * FROM solved_problems ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_flagged_problems() -> list[dict]:
    """Retrieve problems that failed verification or have negative feedback."""
    conn = get_connection()
    query = """
    SELECT p.*, f.feedback_type, f.comment as user_comment
    FROM solved_problems p
    LEFT JOIN user_feedback f ON p.id = f.problem_id
    WHERE p.is_correct = 0 OR f.feedback_type IN ('negative', 'incorrect')
    ORDER BY p.created_at DESC
    """
    rows = conn.cursor().execute(query).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_hitl_history() -> list[dict]:
    """Retrieve history of human-in-the-loop corrections."""
    conn = get_connection()
    rows = conn.cursor().execute("SELECT * FROM hitl_corrections ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]
