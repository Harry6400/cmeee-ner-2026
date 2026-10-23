"""Real Windows owned-child script imports, Unicode argv and process cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import owned_process


@unittest.skipUnless(os.name == 'nt', 'Windows Job Object integration')
class WindowsScriptLaunchTests(unittest.TestCase):
    def test_script_can_import_sibling_and_keep_unicode_arguments(self):
        with tempfile.TemporaryDirectory(prefix='中文 脚本 ') as folder:
            root = Path(folder)
            scripts = root / 'scripts'
            scripts.mkdir()
            (scripts / 'local_helper.py').write_text('VALUE = 42\n', encoding='utf-8')
            entry = scripts / 'entry.py'
            entry.write_text(
                'import json, sys\nfrom local_helper import VALUE\n'
                'print(json.dumps([VALUE, sys.argv, sys.dont_write_bytecode]), flush=True)\n',
                encoding='utf-8',
            )
            process = owned_process.spawn(
                [sys.executable, '-B', '-u', str(entry), '中文 空格😀'],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8',
            )
            try:
                output = process.stdout.read()
                self.assertEqual(process.wait(timeout=10), 0, output)
                self.assertEqual(json.loads(output), [42, [str(entry), '中文 空格😀'], True])
                self.assertTrue(owned_process.gone(process))
                self.assertFalse((scripts / '__pycache__').exists())
            finally:
                if process.poll() is None:
                    owned_process.terminate(process)
                    process.wait(timeout=5)
                process.stdout.close()
                owned_process.release(process)


if __name__ == '__main__':
    unittest.main()
