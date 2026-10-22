"""Synthetic counterexamples only;not evidence of model effectiveness."""
import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage4_metrics import *
from verify_stage4 import independent_wire


def row(entities=(),text='甲乙丙丁',ident='synthetic'):
    return dict(id=ident,text=text,entities=[dict(start=s,end=e,type=t,text=text[s:e]) for s,e,t in entities])

class FrozenMetricTests(unittest.TestCase):
    def test_empty_and_duplicates(self):
        g=row([(0,2,'bod')]);p=row([(0,2,'bod'),(0,2,'bod')]);r=evaluate_route([g],[p],set())
        self.assertEqual(r['main_metrics']['micro'],counts(1,1,1));self.assertEqual(r['main_metrics']['duplicate_predictions'],1)
        self.assertEqual(evaluate_route([row()],[row()],set())['main_metrics']['micro']['f1'],0)
    def test_boundary_type_nested_crossing_coextensive(self):
        g=row([(0,3,'bod'),(1,2,'sym'),(0,3,'dis'),(2,4,'sym')]);p=row([(0,2,'bod'),(1,2,'dis')]);m=evaluate_route([g],[p],set())['main_metrics']
        self.assertEqual(m['micro']['tp'],0);self.assertGreater(m['relaxed_overlap_same_type_one_to_one']['tp'],0)
        self.assertEqual(m['structural_reference_recall']['coextensive_different_type']['gold'],2)
        self.assertGreater(m['structural_reference_recall']['crossing']['gold'],0)
        self.assertEqual(m['structural_reference_recall']['strict_nesting']['gold'],3)
    def test_invalid_is_na_and_not_empty(self):
        g=[row([(0,1,'bod')]),row(ident='s2')];p=[dict(id='synthetic',text='甲乙丙丁',status='invalid',validation_error='bad JSON'),row(ident='s2')]
        r=evaluate_route(g,p,set(),True);self.assertIsNone(r['main_metrics']);self.assertEqual(r['valid'],1);self.assertEqual(r['conditional_valid_only_metrics']['samples'],1)
        with self.assertRaises(ValueError):evaluate_route(g,p,set(),False)
    def test_order_id_surface_validation(self):
        g=[row(),row(ident='s2')]
        with self.assertRaises(ValueError):validate_predictions(g,list(reversed(g)))
        p=row([(0,1,'bod')]);p['entities'][0]['text']='错'
        with self.assertRaises(ValueError):entity_keys(p)
        p=row([(0,1,'bod')]);p['entities'][0]['start']=False
        with self.assertRaises(ValueError):entity_keys(p)
    def test_unicode_and_uncovered_denominator(self):
        text='甲😀e\u0301乙';g=row([(1,2,'bod'),(2,4,'sym')],text=text)
        self.assertEqual(evaluate_route([g],[g],set())['main_metrics']['character_type_micro']['gold'],3)
        text='甲'*300;g=row([(0,300,'bod')],text=text);p=row(text=text)
        m=evaluate_route([g],[p],set())['main_metrics'];self.assertEqual(m['micro']['gold'],1);self.assertEqual(len(m['window_uncovered_reference_entities']),1)
    def test_relaxed_maximum_one_to_one(self):
        # First prediction overlaps both references;second can only matchfirst.
        g=row([(0,1,'bod'),(2,3,'bod')]);p=row([(0,3,'bod'),(0,1,'bod')])
        self.assertEqual(evaluate_route([g],[p],set())['main_metrics']['relaxed_overlap_same_type_one_to_one']['tp'],2)
    def test_oov_surface_filters_both_sides(self):
        g=row([(0,1,'bod'),(1,2,'bod')]);p=row([(0,1,'bod'),(2,3,'bod')]);m=evaluate_route([g],[p],{'甲'})['main_metrics']
        self.assertEqual(m['unseen_surface_strict'],counts(0,1,1));self.assertEqual(m['micro'],counts(1,2,2))
    def test_seed_sample_sd_and_no_best_seed(self):
        s=seed_summary({20261019:.1,20261020:.2,20261021:.3});self.assertAlmostEqual(s['sample_sd'],.1);self.assertEqual(s['primary_value'],.1)
        with self.assertRaises(ValueError):seed_summary({20261019:.1,20261020:.2})
    def test_api_wire_exact_overlapping_occurrence(self):
        g=row(text='甲甲甲😀')
        wire={'entities':[{'text':'甲甲','type':'bod','occurrence':1},{'text':'甲甲','type':'bod','occurrence':1}]}
        entities=independent_wire(g,wire);self.assertEqual(entities,[dict(start=1,end=3,type='bod',text='甲甲')])
        with self.assertRaises(ValueError):independent_wire(g,{'entities':[{'text':'甲甲','type':'bod','occurrence':2}]})
        with self.assertRaises(ValueError):independent_wire(g,{'entities':[{'text':'甲甲','type':'bod','occurrence':True}]})
    def test_paired_bootstrap_complete_and_shared(self):
        g=[row([(0,1,'bod')]),row([(1,2,'sym')],ident='s2')];r=evaluate_route(g,g,set());bad=evaluate_route(g,[dict(id='synthetic',text=g[0]['text'],status='not_sent'),g[1]],set(),True)
        result=paired_bootstrap({'D0':r,'N1':r,'L0':bad},20,20261019)
        self.assertEqual(result['differences']['N1 minus D0']['ci95'],[0.,0.]);self.assertEqual(result['differences']['L0 minus D0']['status'],'N/A')
        r2=evaluate_route(list(reversed(g)),list(reversed(g)),set())
        with self.assertRaises(ValueError):paired_bootstrap({'D0':r,'N1':r2},20,1)

if __name__=='__main__':unittest.main()
