import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./memory/math_mentor.db")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
OCR_CONFIDENCE_THRESHOLD = int(os.getenv("OCR_CONFIDENCE_THRESHOLD", 70))
VERIFIER_CONFIDENCE_THRESHOLD = int(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", 75))

# Model names (optimized for Groq free tier)
MATH_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
MATH_MODEL_BACKUPS = [
    "qwen/qwen3-32b", 
    "llama-3.3-70b-versatile", 
    "openai/gpt-oss-120b"
]

REASONING_MODEL = "openai/gpt-oss-120b"
REASONING_MODEL_BACKUPS = [
    "groq/compound", 
    "moonshotai/kimi-k2-instruct-0905", 
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

FAST_MODEL = "qwen/qwen3-32b"
FAST_MODEL_BACKUPS = [
    "openai/gpt-oss-20b", 
    "llama-3.1-8b-instant", 
    "allam-2-7b"
]

MINI_MODEL = "groq/compound-mini"
MINI_MODEL_BACKUPS = [
    "openai/gpt-oss-safeguard-20b", 
    "meta-llama/llama-prompt-guard-2-86m", 
    "meta-llama/llama-prompt-guard-2-22m"
]

# RAG settings
TOP_K_RETRIEVAL = 1 

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Scoped math topics
ALLOWED_TOPICS = ["algebra", "probability", "calculus", "linear_algebra"]
