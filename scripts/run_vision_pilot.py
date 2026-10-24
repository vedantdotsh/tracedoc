"""Default offline replay; --preflight only checks public pricing; --execute charges."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_lab.vision import preflight, run_pilot, description_preflight, describe_sources

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--execute', action='store_true')
    group.add_argument('--description-preflight', action='store_true')
    group.add_argument('--execute-descriptions', action='store_true')
    args = parser.parse_args()
    if args.description_preflight:
        result = description_preflight(ROOT)
    elif args.execute_descriptions:
        result = describe_sources(ROOT, execute=True)
    else:
        result = preflight(ROOT) if args.preflight else run_pilot(ROOT, execute=args.execute)
    print(json.dumps(result, indent=2, ensure_ascii=False))
