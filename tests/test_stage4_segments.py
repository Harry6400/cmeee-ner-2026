"""Synthetic recovery-chain fixtures only;never claimed as real training/checkpoints."""
import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage4_metrics import digest
from verify_stage4 import check_training_segments
from historical_evidence import REGISTRY_PATH

class FakeEvents:
    def __init__(self,*a,**kw):pass
    def Reload(self):pass
    def Tags(self):return dict(scalars=['train/loss','train/grad_norm','train/lr','train/optimizer_step','train/throughput_samples_per_sec','system/gpu_memory_peak_mb','val/f1'])
    def Scalars(self,tag):return [types.SimpleNamespace(value=.1)]

class SegmentTests(unittest.TestCase):
    def fixture(self,root):
        def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
        scripts=root/'scripts';scripts.mkdir()
        names=('ner_core.py','ner_model.py','ner.py','stage3_common.py','train_stage4.py')
        for n in names:(scripts/n).write_text('# synthetic source fixture only\n')
        code={n:digest(scripts/n) for n in names}
        registry=root/REGISTRY_PATH;registry.parent.mkdir(parents=True)
        saved=registry.parent/'synthetic-source-snapshot';saved.mkdir()
        for n in names:(saved/n).write_bytes((scripts/n).read_bytes())
        save(registry,dict(schema_version=1,entries={'scripts/'+n:dict(sha256=code[n],snapshot=(saved/n).relative_to(root).as_posix()) for n in names}))
        run=root/'runs/synthetic';archive=run/'segments/segment-00';archive.mkdir(parents=True)
        parent=archive/'last.pt';parent.write_bytes(b'synthetic non-loadable checkpoint fixture only')
        sha=digest(parent);base=dict(method='N0',seed=20261020,precision='fp32',amp=False,tf32=False,train_samples=1,dev_samples=1,windows=1,vocab_size=2,sources={'train':'synthetic','dev':'synthetic'},code=code)
        first=dict(base,parent_checkpoint=None);last=dict(base,parent_checkpoint=sha)
        complete1=dict(epoch=1,global_step=2);complete2=dict(epoch=2,global_step=4)
        s0=dict(records=[complete1],global_step=3,seconds=.3,status='segment_limit')
        sf=dict(records=[complete1,complete2],global_step=4,seconds=.2,status='completed')
        def event(step,epoch,cursor):return dict(event='train',global_step=step,epoch=epoch,window_cursor=cursor,loss=.2,grad_norm=.1,seconds=.1,throughput_windows_per_sec=320.,peak_mb=1.)
        for directory,meta,summary,updates in ((archive,first,s0,[event(1,1,32),event(2,1,64),event(3,2,32)]),(run,last,sf,[event(3,2,32),event(4,2,64)])):
            save(directory/'config.json',{});save(directory/'metadata.json',meta);save(directory/'summary.json',summary)
            (directory/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updates));(directory/'tensorboard').mkdir(exist_ok=True)
        arch=dict(last_complete_global_step=2,discarded_partial_updates=1,discarded_partial_update_seconds=.1,last_checkpoint_sha256=sha)
        save(archive/'archive.json',arch)
        restore=dict(parent_sha256=sha,global_step=2,model_equal=True,optimizer_equal=True,python_rng_equal=True,numpy_rng_equal=True,torch_rng_equal=True,cuda_rng_equal=True);save(run/'restore-check.json',restore)
        queue=dict(status='completed',segments=[dict(segment=0,status='interrupted',wall_seconds=1.,archive_dir=str(archive.relative_to(root)),**arch),dict(segment=1,status='completed',wall_seconds=.8)])
        return run,sf,last,queue
    def test_all_segments_discarded_and_wall_reconstructed(self):
        module=types.ModuleType('tensorboard.backend.event_processing.event_accumulator');module.EventAccumulator=FakeEvents
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-segments-') as d,patch.dict(sys.modules,{'tensorboard.backend.event_processing.event_accumulator':module}):
            root=Path(d);run,sf,meta,q=self.fixture(root)
            # A later valid checkout is independent of the archived training version.
            for source in (root/'scripts').glob('*.py'):source.write_text('# newer synthetic implementation\n')
            with patch('historical_evidence.REGISTRY_SHA256',digest(root/REGISTRY_PATH)):
                result=check_training_segments(root,run,sf,meta,{'train/loss','val/f1'},q)
            self.assertEqual(result['logged_updates'],5);self.assertEqual(result['discarded_partial_updates'],1);self.assertAlmostEqual(result['discarded_partial_update_seconds'],.1);self.assertAlmostEqual(result['total_runner_seconds'],.5);self.assertAlmostEqual(result['total_child_wall_seconds'],1.8)
    def test_modified_archive_or_discard_count_rejected(self):
        module=types.ModuleType('tensorboard.backend.event_processing.event_accumulator');module.EventAccumulator=FakeEvents
        with tempfile.TemporaryDirectory(prefix='synthetic-stage4-segments-') as d,patch.dict(sys.modules,{'tensorboard.backend.event_processing.event_accumulator':module}):
            root=Path(d);run,sf,meta,q=self.fixture(root);q['segments'][0]['discarded_partial_updates']=0
            with patch('historical_evidence.REGISTRY_SHA256',digest(root/REGISTRY_PATH)),self.assertRaises(ValueError):check_training_segments(root,run,sf,meta,{'train/loss','val/f1'},q)

if __name__=='__main__':unittest.main()
