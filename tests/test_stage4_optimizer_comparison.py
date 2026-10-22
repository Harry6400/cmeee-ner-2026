"""Synthetic regression for actual CPU-checker tuple/list bug;no GPU or real-data access."""
import ast,copy,json,unittest
from pathlib import Path
class OptimizerSettingsRepresentation(unittest.TestCase):
 def test_equal_serialization_but_changed_values_or_missing_keys_rejected(self):
  p=Path(__file__).resolve().parents[1]/'scripts/inspect_stage4_checkpoints.py';tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='json_equivalent');space={'json':json};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),space);equal=space['json_equivalent']
  actual=[dict(lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False,foreach=None)];frozen=json.loads(json.dumps(actual));self.assertTrue(equal(actual,frozen))
  for key,value in [('lr',3e-5),('betas',[.9,.99])]:
   changed=copy.deepcopy(frozen);changed[0][key]=value;self.assertFalse(equal(actual,changed))
  missing=copy.deepcopy(frozen);del missing[0]['weight_decay'];self.assertFalse(equal(actual,missing))
if __name__=='__main__':unittest.main()
