"""
Reproducibility metadata for VoI runs.

Stamps each saved result with the git state of this repo, plus the stisim
and starsim packages it was linked against. Lets us compare runs across
model versions and understand what changed.

Usage:
    from version_utils import save_with_meta, load_meta, capture_environment

    save_with_meta(draws_df, 'results/voi_draws.df', run={'n_draws': 200})
    meta = load_meta('results/voi_draws.df')

Sidecar convention: meta is written to `<path>.meta.json` alongside the
pickled object; an empty file counts as no metadata.
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import sciris as sc


def _run(cmd, cwd):
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def _git_info(path):
    """Return {sha, branch, dirty} for a git repo at path, or None."""
    p = Path(path)
    if not (p / '.git').exists():
        return None
    sha    = _run(['git', 'rev-parse', 'HEAD'], cwd=p)
    branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=p)
    status = _run(['git', 'status', '--porcelain'], cwd=p)
    if sha is None:
        return None
    return {
        'sha':    sha[:12],
        'branch': branch,
        'dirty':  bool(status),
    }


def _pkg_info(module_name):
    """Return {version, path, git} for an installed package."""
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    info = {
        'version': getattr(mod, '__version__', 'unknown'),
        'path':    str(Path(mod.__file__).resolve().parent),
    }
    repo_root = Path(mod.__file__).resolve().parent.parent
    info['git'] = _git_info(repo_root)
    return info


def capture_environment(run=None):
    """
    Capture a reproducibility snapshot: timestamp, python, package versions,
    git state of the analysis repo and its main deps.

    Args:
        run (dict): optional run-specific metadata (n_draws, scenario, etc.)
    """
    meta = {
        'timestamp':       dt.datetime.now().isoformat(timespec='seconds'),
        'python':          sys.version.split()[0],
        'stisim':          _pkg_info('stisim'),
        'starsim':         _pkg_info('starsim'),
        'anc_sti_repo':    _git_info(Path(__file__).resolve().parent),
    }
    if run is not None:
        meta['run'] = run
    return meta


def _meta_path(obj_path):
    return Path(str(obj_path) + '.meta.json')


def save_with_meta(obj, path, run=None):
    """Save obj with sc.saveobj and write a sidecar .meta.json."""
    sc.saveobj(str(path), obj)
    meta = capture_environment(run=run)
    with open(_meta_path(path), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(path):
    """Load sidecar metadata for a saved object, or None if absent."""
    mp = _meta_path(path)
    if not mp.exists():
        return None
    with open(mp) as f:
        return json.load(f)


def print_env(meta=None):
    """Print a compact summary of an environment snapshot."""
    if meta is None:
        meta = capture_environment()
    ts = meta.get('timestamp', '?')
    py = meta.get('python', '?')
    sti = meta.get('stisim') or {}
    ss  = meta.get('starsim') or {}
    repo = meta.get('anc_sti_repo') or {}
    def _gitstr(g):
        if not g: return 'no git info'
        d = ' (dirty)' if g.get('dirty') else ''
        return f"{g.get('branch','?')}@{g.get('sha','?')[:8]}{d}"
    print(f'timestamp        {ts}')
    print(f'python           {py}')
    print(f'stisim           {sti.get("version","?"):<10} {_gitstr(sti.get("git"))}')
    print(f'starsim          {ss.get("version","?"):<10} {_gitstr(ss.get("git"))}')
    print(f'anc_sti_repo     {"":10} {_gitstr(repo)}')
    if 'run' in meta:
        print('run              ' + json.dumps(meta['run']))


if __name__ == '__main__':
    print_env()
