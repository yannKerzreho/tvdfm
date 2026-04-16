"""Make the repo root available when running pytest directly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
