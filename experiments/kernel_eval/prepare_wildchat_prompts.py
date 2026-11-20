"""Build the real-text Gemma prompt artifact used by the B200 correctness checks.

CPU only. Needs: transformers, tokenizers, pyarrow, jinja2, huggingface_hub.
Source: allenai/WildChat-1M shard 0 (ODC-BY, ungated). Filter: English, not toxic,
not redacted, nonempty alternating turns, distinct first user message, first 4000
in file order. Packing is Resin's own `resin.bench_prompts` policy. NOTE: at short
contexts that policy selects the LONGEST user messages and truncates them, so these
prompts are real user text but code/paste heavy, not a representative chat mix.
The manifest's `prompt_source_id` says "sharegpt" because Resin's writer defaults
that label; the data here is WildChat (see `corpus` in the manifest).
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

REVISION = '3e22461f65e89153144f8adb70e3b8c2cc9845a7'
work = Path(sys.argv[1] if len(sys.argv) > 1 else 'prompts'); work.mkdir(exist_ok=True)
shard = hf_hub_download('allenai/WildChat-1M', 'data/train-00000-of-00014.parquet', repo_type='dataset', local_dir=work)
rows = pq.read_table(shard, columns=['conversation_hash', 'conversation', 'language', 'toxic', 'redacted']).to_pylist()
kept, seen = [], set()
for r in rows:
    if r['language'] != 'English' or r['toxic'] or r['redacted']:
        continue
    msgs = [{'role': m['role'], 'content': m['content']} for m in r['conversation']]
    msgs = msgs[: len(msgs) // 2 * 2]
    if len(msgs) < 2 or any(not m['content'].strip() for m in msgs):
        continue
    key = hashlib.sha256(msgs[0]['content'].encode()).hexdigest()
    if key in seen:
        continue
    seen.add(key); kept.append({'id': r['conversation_hash'], 'conversations': msgs})
    if len(kept) >= 4000:
        break
corpus = work / 'wildchat-en-4000.json'; corpus.write_text(json.dumps(kept, ensure_ascii=False))
meta = {'source': 'allenai/WildChat-1M', 'file': 'data/train-00000-of-00014.parquet', 'license': 'odc-by',
        'filter': 'language==English, toxic false, redacted false, nonempty alternating turns, distinct first user message, first 4000 in file order',
        'file_sha256': hashlib.sha256(Path(shard).read_bytes()).hexdigest()}
tokenizer_dir = snapshot_download('google/gemma-4-E2B-it', revision=REVISION,
    allow_patterns=['tokenizer*', '*.jinja', 'special_tokens_map.json', 'generation_config.json', 'config.json'], local_dir=work / 'gemma-tokenizer')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
spec = importlib.util.spec_from_file_location('bench_prompts', Path(__file__).resolve().parents[2] / 'resin/bench_prompts.py')
module = importlib.util.module_from_spec(spec); sys.modules['bench_prompts'] = module; spec.loader.exec_module(module)
artifact = module.prepare_prompt_artifact(corpus, work / 'wildchat-gemma-e2b.zip', tokenizer=tokenizer,
    tokenizer_identity={'repo_id': 'google/gemma-4-E2B-it', 'revision': REVISION}, source_metadata=meta,
    contexts=(128, 256, 384, 512, 768, 1024), count=1, seed=41)
print(artifact.metadata['artifact_sha256'])
