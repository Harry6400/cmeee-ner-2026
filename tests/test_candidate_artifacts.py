"""Local candidate selection must not serve a stale release or escape the project."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from teacher_project import TeacherProject


class CandidateArtifactsTests(unittest.TestCase):
    def test_current_pointer_selects_new_report_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=root/'deliverables/stage6/candidate';old.mkdir(parents=True)
            (old/'报告.pdf').write_bytes(b'old synthetic PDF')
            current=root/'deliverables/stage6/revision/candidate';current.mkdir(parents=True)
            (current/'报告.pdf').write_bytes(b'new synthetic PDF')
            archive=current.parent/'new.zip';archive.write_bytes(b'synthetic archive')
            pointer=root/'deliverables/stage6/current-candidate.json'
            pointer.write_text(json.dumps(dict(directory=current.relative_to(root).as_posix(),archive=archive.relative_to(root).as_posix())))
            project=object.__new__(TeacherProject);project.root=root
            paths=project.artifact_paths()
            self.assertEqual(paths['report'][1],(current/'报告.pdf').resolve())
            self.assertEqual(paths['source-zip'][1],archive.resolve())

    def test_missing_or_outside_pointer_fails_without_stale_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'deliverables/stage6';directory.mkdir(parents=True)
            project=object.__new__(TeacherProject);project.root=root
            for name in ('missing','../../../outside'):
                (directory/'current-candidate.json').write_text(json.dumps(dict(directory=name,archive='missing.zip')))
                with self.assertRaises(ValueError):project.artifact_paths()


if __name__=='__main__':unittest.main()
