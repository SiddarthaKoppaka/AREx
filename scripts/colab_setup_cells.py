"""Drive, repository, and Python setup cells for generated Colab notebook."""


def setup_cells() -> list[tuple[str, str, str]]:
    return [
        (
            "markdown",
            "intro",
            """
            # AREx on Colab: Transformers/A100
            Run cells in order. The model path and ID are explicit overrides below.
            Results, logs, and checkpoints persist in Google Drive.
        """,
        ),
        (
            "code",
            "mount-drive",
            """
            from google.colab import drive
            drive.mount('/content/drive')
        """,
        ),
        (
            "code",
            "sync-repository",
            """
            import os, subprocess
            from pathlib import Path
            REPO = Path('/content/AREx-source')
            BRANCH = os.environ.get(
                'AREX_GIT_REF', 'codex/centralized-runtime-config')
            if not REPO.exists():
                subprocess.run(['git', 'clone', '--branch', BRANCH,
                    'https://github.com/SiddarthaKoppaka/AREx', str(REPO)], check=True)
            else:
                subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin'], check=True)
                subprocess.run(['git', '-C', str(REPO), 'switch', BRANCH], check=True)
                subprocess.run(
                    ['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
            print(subprocess.check_output(
                ['git', '-C', str(REPO), 'rev-parse', 'HEAD']).decode())
        """,
        ),
        (
            "code",
            "install-python312",
            """
            import shutil, subprocess, sys
            from pathlib import Path
            REPO = Path('/content/AREx-source')
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
            UV = shutil.which('uv')
            assert UV, 'uv installation failed'
            subprocess.run([UV, 'python', 'install', '3.12'], check=True)
            subprocess.run(
                [UV, 'venv', '/content/arex-py312', '--python', '3.12'], check=True)
            subprocess.run([UV, 'pip', 'install', '--python',
                '/content/arex-py312/bin/python',
                '-e', f'{REPO}[colab]'], check=True)
        """,
        ),
    ]
