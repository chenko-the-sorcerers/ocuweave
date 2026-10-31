"""Check synchronized Python/native/citation release metadata."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"', (root / 'pyproject.toml').read_text(), re.M).group(1)
checks = {
    'src/ocuweave/__init__.py': f'__version__ = "{version}"',
    'CMakeLists.txt': f'project(ocuweave VERSION {version} LANGUAGES CXX)',
    'CITATION.cff': f'version: {version}',
    'README.md': f'--branch v{version}',
    'CHANGELOG.md': f'## [{version}]',
}
for file, expected in checks.items():
    if expected not in (root / file).read_text():
        raise SystemExit(f'Release mismatch: {file}')
print(f'Release metadata synchronized: {version}')
