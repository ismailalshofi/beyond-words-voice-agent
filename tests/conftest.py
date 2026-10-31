import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database before anything imports app.settings
_db_dir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{Path(_db_dir, 'test.db').as_posix()}"
os.environ.setdefault("OPENAI_API_KEY", "test")
