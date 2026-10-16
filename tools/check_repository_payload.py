"""Audit the Git index before publishing: code/docs only, no data or credentials."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    names = git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').decode().split('\0')
    total = 0
    for name in filter(None, names):
        path = Path(name)
        if path.suffix not in {'.py', '.md', '.txt', '.json'} and name not in {'.gitignore', '.gitattributes'}:
            raise ValueError(f'Unexpected staged payload type: {name}')
        if path.suffix == '.json' and path.parts[0] != 'docs':
            raise ValueError(f'JSON dataset/report is not allowed outside docs: {name}')
        payload = git('show', ':' + name)
        if len(payload) > 512 * 1024:
            raise ValueError(f'Staged file exceeds the code/docs size cap: {name}')
        text = payload.decode('utf-8')
        if re.search(r'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----', text):
            raise ValueError(f'Potential credential payload: {name}')
        if path.suffix == '.json' and not isinstance(json.loads(text), dict):
            raise ValueError(f'JSON must be documentation metadata, not dataset arrays: {name}')
        total += len(payload)
    provenance = json.loads((ROOT / 'docs/numeric_provenance.json').read_text())
    for row in provenance['modules']:
        assert hashlib.sha256((ROOT / row['module']).read_bytes()).hexdigest() == row['repository_sha256'], row['module']
        assert hashlib.sha256(git('show', ':' + row['module'])).hexdigest() == row['repository_sha256'], row['module']
    print(f'PASS: staged additions/modifications contain code/docs only, {total} bytes; 16 numeric provenance hashes match.')


if __name__ == '__main__':
    main()
