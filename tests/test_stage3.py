"""课程指标的独立反例；不作为模型效果。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ner_core import score
from stage3_common import extra_metrics,flat_upper_bound


def row(entities):
 return dict(id='fixture',text='甲乙丙丁',entities=[dict(start=a,end=b,type=t,text='甲乙丙丁'[a:b]) for a,b,t in entities])

class MetricTests(unittest.TestCase):
 def test_token_overlap_is_not_exact_entity(self):
  g=row([(0,3,'bod')]);p=row([(0,2,'bod')]);e=extra_metrics([g],[p])
  self.assertEqual(score([g],[p])['micro']['f1'],0)
  self.assertEqual(e['character_type_micro']['f1'],.8)
  self.assertEqual(e['relaxed_overlap_same_type_one_to_one']['f1'],1)
 def test_relaxed_one_to_one_not_double_count(self):
  g=row([(0,3,'bod')]);p=row([(0,2,'bod'),(1,3,'bod')]);e=extra_metrics([g],[p])
  self.assertEqual(e['relaxed_overlap_same_type_one_to_one']['tp'],1)
 def test_nested_and_coextensive_multihot(self):
  g=row([(0,4,'bod'),(1,2,'sym'),(0,4,'dis')]);p=row([(0,4,'bod')]);e=extra_metrics([g],[p])
  self.assertEqual(e['character_type_micro']['gold'],9)
  self.assertEqual(flat_upper_bound([g])['max_representable_without_overlapping_intervals'],1)
 def test_crossing_flat_max_cardinality(self):
  g=row([(0,2,'bod'),(1,3,'sym'),(2,4,'dis')])
  self.assertEqual(flat_upper_bound([g])['max_representable_without_overlapping_intervals'],2)

if __name__=='__main__':unittest.main()
