"""Launch script for the Streamlit viewer."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "viewer" / "app.py"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "true"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
