"""Reproduce paired-round inference for the frozen dispatch; no GPU needed."""
import json
import random
import statistics
from pathlib import Path


def dispatch(n):
    return 'rust_256' if n <= 12288 else 'rust_512' if n <= 405504 else 'triton'


def interval(values, seed=719, repetitions=4000):
    rng=random.Random(seed)
    medians=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(repetitions))
    return [medians[int(repetitions*.025)],medians[int(repetitions*.975)]]


def analyze(report):
    rows=[]
    for r in report['results']:
        label=dispatch(r['elements'])
        baseline=r['us_per_call']['triton'];candidate=r['us_per_call'][label]
        paired=[100*(1-c/b) for b,c in zip(baseline,candidate)]
        rows.append({'elements':r['elements'],'selected':label,
                     'triton_us':statistics.median(baseline),'selected_us':statistics.median(candidate),
                     'paired_median_latency_reduction_percent':statistics.median(paired),
                     'paired_bootstrap_95_interval':interval(paired)})
    return rows


if __name__=='__main__':
    root=Path(__file__).parent/'results'
    for name in ('b200-shared-buffer-tuning','b200-shared-buffer-heldout-1193','b200-shared-buffer-heldout-2399'):
        print(name)
        for row in analyze(json.loads((root/(name+'.json')).read_text())):
            print(json.dumps(row))
