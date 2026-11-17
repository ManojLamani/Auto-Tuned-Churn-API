"""CI gate for checksum-verified model releases."""
import sys

from training.artifacts import load_release
from training.config import MODELS_DIR


def check(models_dir=MODELS_DIR):
    try:
        load_release(models_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return [str(exc)]
    return []


if __name__ == "__main__":
    problems = check()
    for problem in problems:
        print("FAIL:", problem)
    print("Artifact validation", "failed" if problems else "passed")
    sys.exit(bool(problems))
