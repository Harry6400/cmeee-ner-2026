"""Synthetic counterexamples, not measured extraction results."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage5_repairs import repair_wire,Engine

class Repairs(unittest.TestCase):
    def test_covered_overflow_is_explicit(self):
        r=repair_wire({'id':'s','text':'GH和GH'}, {'entities':[{'text':'GH','type':'ite','occurrence':n} for n in [0,1,2]]})
        self.assertEqual(r['status'],'repaired');self.assertEqual(len(r['actions']),1)
        self.assertEqual([(e['start'],e['end']) for e in r['prediction']['entities']],[(0,2),(3,5)])
    def test_incomplete_coverage_stays_invalid(self):
        r=repair_wire({'id':'s','text':'GH和GH'}, {'entities':[{'text':'GH','type':'ite','occurrence':n} for n in [0,2]]})
        self.assertEqual(r['status'],'invalid');self.assertIsNone(r['prediction'])
    def test_hallucination_stays_invalid(self):
        self.assertEqual(repair_wire({'id':'s','text':'鼻病毒等病毒感染'},{'entities':[{'text':'鼻病毒感染','type':'dis','occurrence':0}]})['status'],'invalid')
    def test_bool_and_unknown_type_rejected(self):
        for typ,n in [('bod',True),('fake',0),('bod',-1)]:
            self.assertEqual(repair_wire({'id':'s','text':'肺'},{'entities':[{'text':'肺','type':typ,'occurrence':n}]})['status'],'invalid')
    def test_unicode_and_nested_positions(self):
        r=repair_wire({'id':'s','text':'😀肺炎肺'}, {'entities':[{'text':'肺炎','type':'dis','occurrence':0},{'text':'肺','type':'bod','occurrence':0},{'text':'肺','type':'bod','occurrence':1}]})
        self.assertEqual([(e['start'],e['end']) for e in r['prediction']['entities']],[(1,3),(1,2),(3,4)])
    def test_prediction_interface_rejects_gold(self):
        with self.assertRaises(ValueError): Engine.predict(None,{'id':'s','text':'肺','entities':[]})

if __name__=='__main__': unittest.main(verbosity=2)
