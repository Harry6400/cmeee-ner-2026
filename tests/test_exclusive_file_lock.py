"""Real process contention and crash release; no network or model workload."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import exclusive_file_lock as lock


class ExclusiveFileLockTests(unittest.TestCase):
    def probe(self, path):
        code = (
            'import os,sys; sys.path.insert(0,sys.argv[1]); import exclusive_file_lock as lock\n'
            'fd=os.open(sys.argv[2],os.O_CREAT|os.O_RDWR,0o600)\n'
            'try:\n lock.acquire(fd)\nexcept BlockingIOError:\n print("blocked")\n'
            'else:\n print("acquired"); lock.release(fd)\nfinally:\n os.close(fd)\n'
        )
        return subprocess.check_output(
            [sys.executable, '-X', 'utf8', '-B', '-c', code, str(SCRIPTS), str(path)],
            text=True, encoding='utf-8', timeout=10,
        ).strip()

    def test_contention_release_and_other_directory(self):
        with tempfile.TemporaryDirectory(prefix='中文 互斥 ') as folder:
            path = Path(folder) / 'run.lock'
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                lock.acquire(fd)
                self.assertEqual(self.probe(path), 'blocked')
                self.assertEqual(self.probe(Path(folder) / 'other.lock'), 'acquired')
                lock.release(fd)
                self.assertEqual(self.probe(path), 'acquired')
                self.assertEqual(path.read_bytes(), b'')
            finally:
                os.close(fd)

    def test_owner_exit_releases_lock_and_preserves_position(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.lock'
            code = (
                'import os,sys; sys.path.insert(0,sys.argv[1]); import exclusive_file_lock as lock; '
                'fd=os.open(sys.argv[2],os.O_CREAT|os.O_RDWR,0o600); '
                'lock.acquire(fd); os._exit(0)'
            )
            subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPTS), str(path)], check=True, timeout=10)
            self.assertEqual(self.probe(path), 'acquired')
            fd = os.open(path, os.O_RDWR)
            try:
                os.lseek(fd, 7, os.SEEK_SET)
                lock.acquire(fd)
                self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 7)
                lock.release(fd)
                self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 7)
            finally:
                os.close(fd)


if __name__ == '__main__':
    unittest.main()
