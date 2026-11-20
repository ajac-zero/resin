"""Own deterministic fixtures only; never import arbitrary candidate code locally."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numerics import bf16, gelu, reference

mode = sys.argv[1]
request = json.loads(open(sys.argv[2]).read())
outputs = []
cache = None
for case in request['cases']:
    g, u = case['gate'], case['up']
    out = reference(g, u)
    if mode == 'zero':
        out = [0.] * len(g)
    elif mode == 'skip_round':
        out = [bf16(gelu(x)*y) for x, y in zip(g, u)]
    elif mode == 'tail':
        out = out[:-1]
    elif mode == 'stale':
        if cache is None:
            cache = out
        out = cache
    elif mode == 'hang':
        time.sleep(30)
    elif mode == 'crash':
        raise RuntimeError('fixture crash')
    elif mode == 'malformed':
        print('not JSON')
        sys.exit(0)
    elif mode != 'reference':
        raise ValueError('unknown fixture')
    outputs.append(out)
print(json.dumps({'outputs': outputs}, allow_nan=False))
