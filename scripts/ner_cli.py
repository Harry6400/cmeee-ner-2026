"""Validate and serialize one Unicode-offset entity record."""
from __future__ import annotations
import argparse
import json
from ner_core import validate

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', default='example')
    parser.add_argument('--text', required=True)
    parser.add_argument('--entities', default='[]', help='JSON array of exact entity spans')
    args = parser.parse_args()
    try:
        entities = json.loads(args.entities)
        if not isinstance(entities, list):
            raise ValueError('entities must be a JSON array')
        row = {'id': args.id, 'text': args.text, 'entities': entities}
        validate(row)
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
