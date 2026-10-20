"""Pure exact-surface grounding. No credentials, network client, or private ledger dependency."""
from ner_core import TYPES,validate
def occurrences(text,surface):
    if not isinstance(surface,str) or not surface:raise ValueError('实体原文为空或非法')
    return [i for i in range(len(text)) if text.startswith(surface,i)]
def from_wire(row,parsed):
    if not isinstance(parsed,dict) or not isinstance(parsed.get('entities'),list):raise ValueError('缺少entities')
    entities=[]
    for e in parsed['entities']:
        n=e.get('occurrence');surface=e.get('text');positions=occurrences(row['text'],surface)
        if e.get('type') not in TYPES or type(n)is not int or not 0<=n<len(positions):raise ValueError('类别、原文或出现序号非法')
        start=positions[n];entities.append(dict(start=start,end=start+len(surface),type=e['type'],text=surface))
    result=dict(id=row['id'],text=row['text'],entities=entities);validate(result);return result
