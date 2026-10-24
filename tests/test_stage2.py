"""合成边界与模型机制测试；不是正式效果实验。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from ner_core import TYPES,score,validate,windows,merge,dictionary,dictionary_predict,coverage
from ner_model import Extractor,collate,loss_fn,decode,configure


def sample(text='甲😀乙丙'):
    return dict(id='synthetic',text=text,entities=[])


def ent(row,a,b,t='bod'):
    return dict(start=a,end=b,type=t,text=row['text'][a:b])


class CoreTests(unittest.TestCase):
    def test_strict_metrics_duplicate_and_empty(self):
        row=sample();row['entities']=[ent(row,0,2),ent(row,1,2,'sym')]
        pred=dict(row,entities=[row['entities'][0]]*2+[ent(row,0,2,'dis')])
        metrics=score([row],[pred])
        self.assertEqual(metrics['micro']['tp'],1)
        self.assertEqual(metrics['micro']['gold'],2)
        self.assertEqual(metrics['micro']['predicted'],2)
        self.assertEqual(metrics['duplicate_predictions'],1)
        self.assertEqual(score([sample()],[sample()])['micro']['f1'],0)
        self.assertEqual(metrics['nested_recall'],.5)

    def test_wrong_boundary_type_and_missing_predictions(self):
        row=sample();row['entities']=[ent(row,0,2)]
        for e in [ent(row,0,1),ent(row,0,2,'dis')]:
            self.assertEqual(score([row],[dict(row,entities=[e])])['micro']['tp'],0)
        for pred in [[],[dict(row,entities=None)],[dict(row,text='different')]]:
            with self.assertRaises((ValueError,TypeError)):
                score([row],pred)

    def test_nesting_crossing_coextensive_roundtrip(self):
        row=sample('甲乙丙丁戊己庚辛')
        row['entities']=[ent(row,0,4),ent(row,1,3,'sym'),ent(row,2,5,'dis'),ent(row,0,4,'sym')]
        ws=windows(row,6,4)
        reconstructed=merge(row,[(w,w['entities']) for w in ws])
        self.assertEqual(score([row],[reconstructed])['micro']['f1'],1)

    def test_long_window_all_characters_covered(self):
        row=sample('甲'*4870)
        row['entities']=[ent(row,240,379),ent(row,4860,4870)]
        ws=windows(row)
        self.assertEqual({i for w in ws for i in range(w['start'],w['end'])},set(range(4870)))
        self.assertEqual(coverage([row],256,160)['uncovered_entities'],[])
        impossible=dict(row,entities=[ent(row,0,300)])
        self.assertEqual(len(coverage([impossible],256,160)['uncovered_entities']),1)
        self.assertEqual(score([impossible],[dict(row,entities=[])])['micro']['gold'],1)

    def test_unicode_no_normalization(self):
        row=sample('A e\u0301 😀 中\n')
        row['entities']=[ent(row,2,4,'dru'),ent(row,5,6)]
        validate(row)
        self.assertEqual(merge(row,[(w,w['entities']) for w in windows(row,6,4)])['entities'],row['entities'])
        with self.assertRaises(ValueError):validate(dict(row,entities=[dict(row['entities'][0],text='é')]))

    def test_dictionary_nested_and_tie_policy(self):
        row=sample('甲乙甲')
        row['entities']=[ent(row,0,2),ent(row,0,1),ent(row,2,3,'sym')]
        table,ambiguity=dictionary([row])
        self.assertEqual(table['甲'],'sym')
        self.assertIn('甲',ambiguity)
        pred=dictionary_predict(row,table)
        self.assertEqual(len(pred['entities']),3)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure(123,2)

    def test_mask_loss_gradient_and_padding_independence(self):
        cfg=dict(embedding_dim=8,lstm_hidden=8,head_size=4,rope=True)
        vocab={'[PAD]':0,'[UNK]':1,'甲':2,'乙':3,'[CLS]':4,'[SEP]':5}
        row=sample('甲乙');row['entities']=[ent(row,0,2),ent(row,1,2,'sym')]
        short=windows(row)[0];long=windows(sample('甲乙甲乙'))[0]
        model=Extractor('N0',cfg,vocab).eval()
        a=collate([short],vocab,'N0');b=collate([short,long],vocab,'N0')
        x=model(*a[:3]);y=model(*b[:3])
        self.assertTrue(torch.allclose(x[0,:,:2,:2],y[0,:,:2,:2],atol=1e-6))
        loss=loss_fn(y,b[3]);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertIsNotNone(model.head.proj.weight.grad)
        self.assertTrue((y[0,:,:,2:] < -1e10).all())
        self.assertTrue((y[:,:,1,0] < -1e10).all())
        decode(y,[short,long],'N0',0)

    def test_bert_special_tokens_and_unicode_char_alignment(self):
        cfg=dict(head_size=4,rope=True)
        bc=dict(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                intermediate_size=32,max_position_embeddings=512,pad_token_id=0)
        vocab={'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,'甲':4,'乙':5}
        row=sample('甲😀乙');row['entities']=[ent(row,1,2)]
        ws=windows(row);batch=collate(ws,vocab,'N1')
        self.assertEqual(batch[0].tolist(),[[2,4,1,5,3]])
        self.assertEqual(batch[3][0,TYPES.index('bod'),2,2].item(),1)
        model=Extractor('N1',cfg,vocab,bc).eval()
        logits=model(*batch[:3])
        self.assertTrue((logits[:,:,0,:]<-1e10).all())
        self.assertTrue((logits[:,:,:,-1]<-1e10).all())
        self.assertTrue(torch.isfinite(loss_fn(logits,batch[3])))
        self.assertTrue(all(e['text']==row['text'][e['start']:e['end']]
                            for e in decode(logits,ws,'N1',0)[0]))

if __name__=='__main__':unittest.main()
