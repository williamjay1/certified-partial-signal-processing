"""Download one fresh, immutable official HAR archive after the feasibility gate.

This script does not parse outcomes or test signals. Source membership is metadata.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile

WORK = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
RAW = Path(r'F:\AcademicData\ASC_CertifiedProcessing\raw')
URL = 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip'
SOURCE = 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones'

def main():
    drives = {d: shutil.disk_usage(d).free for d in ('D:\\', 'E:\\', 'F:\\')}
    if drives['F:\\'] < 500_000_000 or drives['D:\\'] < 2_000_000_000:
        raise RuntimeError('Insufficient space at the authorized raw/work drives')
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = RAW / ('uci_har_240__' + stamp + '_' + uuid.uuid4().hex[:10] + '.zip')
    sha = hashlib.sha256()
    req = urllib.request.Request(URL, headers={'User-Agent': 'Academic-research-source-download'})
    with urllib.request.urlopen(req, timeout=120) as src, path.open('xb') as dst:
        headers = dict(src.headers)
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            sha.update(chunk)
    if not zipfile.is_zipfile(path):
        raise RuntimeError('Downloaded source is not a ZIP archive')
    expected = headers.get('Content-Length')
    if expected is not None and int(expected) != path.stat().st_size:
        raise RuntimeError('Source response size mismatch')
    with zipfile.ZipFile(path) as z:
        # Inspect only central-directory metadata. Test-member payloads remain
        # unopened; CRC checks occur when the training-only loader reads members.
        members = [{'name': x.filename, 'uncompressed_bytes': x.file_size,
                    'compressed_bytes': x.compress_size} for x in z.infolist()]
    os.chmod(path, 0o444)
    manifest = {'source_url': SOURCE, 'download_url': URL, 'path': str(path),
                'downloaded_utc': stamp, 'bytes': path.stat().st_size,
                'sha256': sha.hexdigest(), 'license': 'CC BY 4.0 per official UCI page',
                'data_doi': '10.24432/C54S4K', 'disk_free_bytes_before': drives,
                'response_headers': {k: v for k, v in headers.items()
                                     if k.lower() in ('content-length', 'content-type', 'last-modified')},
                'archive_members': members, 'outcome_or_test_signal_files_parsed': 0,
                'status': 'downloaded_and_zip_metadata_checked',
                'test_member_payloads_opened': 0,
                'crc_policy': 'CRC checked only for members later opened by train-only loader'}
    dest = WORK / 'datasets' / ('raw_source_manifest_' + stamp + '.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(dest), 'raw': str(path), 'bytes': manifest['bytes'],
                      'sha256': manifest['sha256'], 'member_count': len(members)}), flush=True)

if __name__ == '__main__':
    main()
