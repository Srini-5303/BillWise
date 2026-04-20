from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

repo_root = Path(__file__).resolve().parents[1]
env_path = repo_root / ".env"
load_dotenv(env_path)

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError(f"GEMINI_API_KEY not set in {env_path}")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly one word: OK",
    config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=8,
    ),
)

print((response.text or "").strip())