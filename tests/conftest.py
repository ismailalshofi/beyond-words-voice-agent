import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database before anything imports app.settings
_db_dir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{Path(_db_dir, 'test.db').as_posix()}"
os.environ.setdefault("OPENAI_API_KEY", "test")
# The tests expect OpenAI defaults, whatever the local .env says
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = ""
os.environ["LLM_REASONING_EFFORT"] = ""
os.environ["CALLS_DIR"] = ""
