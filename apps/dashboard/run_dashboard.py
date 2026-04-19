from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    env_path = repo_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    app_path = here / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    main()