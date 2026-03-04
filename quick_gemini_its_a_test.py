"""Quick sanity check: LLM connectivity via .env (no keys in repo)."""
from pathlib import Path
import os

# Load .env from project root so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

backend = os.getenv("LLM_BACKEND", "gemini").lower()

if backend == "openai":
    from openai import OpenAI

    base_url = os.getenv("OPENAI_BASE_URL", "https://cbsai.business.columbia.edu/api/v1")
    client = OpenAI(base_url=base_url)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Say hello in Chinese in one sentence."}],
    )
    print(resp.choices[0].message.content)
else:
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY from env or .env
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents="Say hello in Chinese in one sentence.",
    )
    print(resp.text)
