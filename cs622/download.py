"""Download the project datasets into data/raw/ and verify their checksums.

Usage: python -m cs622.download [deepsynergy] [drugcomb_summary] [drugcomb_raw] [almanac]
With no arguments, downloads everything (about 2.9 GB on disk). Files already present
with the right checksum are skipped.
"""
import hashlib, sys, urllib.request, zipfile
from pathlib import Path

RAW = Path('data/raw')

DATASETS = {
    'deepsynergy': dict(
        path='deepsynergy/labels.csv',
        url='https://www.bioinf.jku.at/software/DeepSynergy/labels.csv',
        sha256='4ef40678fff51faa84fa4de63344c5b51e2e4abe7c28a417af24ec2b50fa135a'),
    'drugcomb_summary': dict(
        path='drugcomb/summary_table_v1.4.csv',
        url='https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content',
        md5='c11efbdcae4a860c2374c1505a66599b'),
    'drugcomb_raw': dict(
        path='drugcomb/drugcomb_data_v1.4.csv',
        url='https://zenodo.org/api/records/18449193/files/drugcomb_data_v1.4.csv/content',
        md5='f5f071de351ec38ffe29c328519a7be5'),
    'almanac': dict(
        path='almanac/ComboDrugGrowth_Nov2017.zip',
        url='https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip?version=1&modificationDate=1510057275000&api=v2',
        sha256='69367a8b654908ebf70a752548c8d240fda2b751336eff0241a5986628696100',
        unzip='ComboDrugGrowth_Nov2017.csv'),
}


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def ok(path, spec):
    algo = 'sha256' if 'sha256' in spec else 'md5'
    return path.exists() and digest(path, algo) == spec[algo]


def fetch(name, spec):
    dest = RAW / spec['path']
    done = dest.with_name(spec['unzip']) if 'unzip' in spec else dest
    if done.exists() and ('unzip' in spec or ok(dest, spec)):
        print(f'{name}: already present ({done.relative_to(RAW)})')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'{name}: downloading {spec["url"]}')
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(spec['url']) as r, open(tmp, 'wb') as fh:
        total = int(r.headers.get('Content-Length') or 0); got = 0
        for chunk in iter(lambda: r.read(1 << 20), b''):
            fh.write(chunk); got += len(chunk)
            if total:
                print(f'\r  {got / 1e6:,.0f} / {total / 1e6:,.0f} MB', end='', flush=True)
    print()
    tmp.replace(dest)
    if not ok(dest, spec):
        sys.exit(f'{name}: checksum mismatch for {dest}; delete it and retry')
    if 'unzip' in spec:
        with zipfile.ZipFile(dest) as z:
            z.extract(spec['unzip'], dest.parent)
        dest.unlink()
    print(f'{name}: ok')


if __name__ == '__main__':
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        if n not in DATASETS:
            sys.exit(f'unknown dataset {n!r}; choose from {", ".join(DATASETS)}')
        fetch(n, DATASETS[n])
