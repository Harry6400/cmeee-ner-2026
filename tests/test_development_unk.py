import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_development_unk import analyze


class DevelopmentUNKTest(unittest.TestCase):
    def setUp(self):
        self.gold = dict(id='synthetic-1', text='甲A😀乙', entities=[
            dict(start=0, end=1, type='bod', text='甲'),
            dict(start=1, end=3, type='dis', text='A😀'),
            dict(start=0, end=4, type='sym', text='甲A😀乙')])
        self.prediction = dict(self.gold, entities=[self.gold['entities'][0]] * 2)

    def test_unicode_dedup_and_denominators(self):
        result = analyze([self.gold], [self.prediction], {'甲', '乙', 'a'})
        self.assertEqual(result['strict'], dict(tp=1, predicted=1, gold=3, fn=2, fp=0, f1=.5))
        overall = [r for r in result['rows'] if r['dimension'] == 'overall']
        self.assertEqual([(r['gold'], r['tp'], r['fn'], r['recall']) for r in overall], [(2, 0, 2, 0), (1, 1, 0, 1)])
        self.assertEqual(result['unknown_characters'], 2)
        for dimension in ('overall', 'type', 'length'):
            rows = [r for r in result['rows'] if r['dimension'] == dimension]
            self.assertEqual(sum(r['gold'] for r in rows), 3)
            self.assertEqual(sum(r['fn'] for r in rows), 2)
            self.assertEqual(sum(r['share_of_all_FN'] for r in rows), 1)
        empty = next(r for r in result['rows'] if r['stratum'] == '>20')
        self.assertIsNone(empty['recall'])

    def test_changed_text_id_order_and_duplicate_ids_fail(self):
        for changed in (dict(self.prediction, text='别的'), dict(self.prediction, id='wrong')):
            with self.assertRaises(ValueError):
                analyze([self.gold], [changed], {'甲'})
        with self.assertRaises(ValueError):
            analyze([self.gold] * 2, [self.prediction] * 2, {'甲'})
        with self.assertRaises(ValueError):
            analyze([self.gold], [], {'甲'})

    def test_wrong_unicode_span_type_or_surface_fail(self):
        for entity in (dict(start=1, end=4, type='dis', text='A😀'),
                       dict(start=1, end=3, type='bad', text='A😀'),
                       dict(start=True, end=3, type='dis', text='A😀')):
            with self.assertRaises(ValueError):
                analyze([self.gold], [dict(self.prediction, entities=[entity])], {'甲'})


if __name__ == '__main__':
    unittest.main()
