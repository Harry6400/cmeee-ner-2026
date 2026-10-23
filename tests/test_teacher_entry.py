"""Synthetic failure contracts;real complete reproduction is tested from the candidate archive separately."""
import unittest,tempfile,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from teacher_reproduce import execute,verify_model_assets
class TeacherEntryTests(unittest.TestCase):
 def test_existing_output_is_never_changed(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'output';out.mkdir();marker=out/'keep';marker.write_text('original')
   with self.assertRaises(FileExistsError):execute(root,root/'absent.zip',out)
   self.assertEqual(marker.read_text(),'original');self.assertEqual(list(out.iterdir()),[marker])
 def test_missing_assets_produce_actual_failure_log(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'new';result=execute(root,root/'absent.zip',out)
   self.assertEqual(result['status'],'FAIL');self.assertFalse(result['private_ledger_required']);self.assertEqual(result['API_calls'],0)
   self.assertTrue((out/'run.log').is_file());self.assertIn('FileNotFoundError',(out/'summary.json').read_text())
 def test_wrong_seed_and_tampered_model_are_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);d=root/'models/trained';d.mkdir(parents=True);items=[]
   for name in ['N0','N1']:
    (d/(name+'.pt')).write_bytes(b'synthetic-not-a-pickle');items.append(dict(method=name,file=name+'.pt',seed=20261019,sha256=hashlib.sha256(b'synthetic-not-a-pickle').hexdigest()))
   (d/'manifest.json').write_text(json.dumps(dict(items=items)));self.assertEqual(verify_model_assets(root)['seed'],20261019)
   (d/'N1.pt').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'mismatch'):verify_model_assets(root)
   items[0]['seed']=20261020;(d/'manifest.json').write_text(json.dumps(dict(items=items)))
   with self.assertRaisesRegex(ValueError,'seed'):verify_model_assets(root)
if __name__=='__main__':unittest.main()
