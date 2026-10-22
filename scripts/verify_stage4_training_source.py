"""Read-only structural conservation check between original scientific loop and new explicit-seed runner."""
import argparse,ast,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loop(source):
    tree=ast.parse(source.read_text())
    loops=[x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='epoch']
    if len(loops)!=1:raise ValueError('epoch loop ambiguous')
    return ast.dump(loops[0],include_attributes=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    same=loop(a.baseline)==loop(a.candidate)
    s=a.candidate.read_text();no_final=not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and ('/sealed/' in n.value or n.value.startswith('data/sealed')) for n in ast.walk(ast.parse(s)))
    result=dict(status='PASS' if same and no_final else 'FAIL',full_epoch_train_validation_checkpoint_loop_ast_identical=same,training_runner_does_not_read_sealed=no_final,baseline_sha256=sha(a.baseline),candidate_sha256=sha(a.candidate),allowed_changes='explicit frozen config/seed; restore-state audit; historical save-seconds metadata reconciliation; terminal progress logger; no scientific loop change')
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
