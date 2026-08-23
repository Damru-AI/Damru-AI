#!/usr/bin/env python3
# ============================================================================
#  merge_buckets.py  --  Damru AI :: HF buckets -> damru-knowledge merger
#  ----------------------------------------------------------------------------
#  KYA KARTA HAI
#    * Tere HF ke N dataset repos ("buckets") padhta hai
#    * Har row ko damru-knowledge schema me normalize:
#        { question, answer, text, domain, source, intent, lang, bucket }
#    * Ek bucket = ek shard -> DEST_REPO/<DEST_PREFIX>/<name>.jsonl
#    * NON-DESTRUCTIVE: purani files ko haath nahi lagata, sirf naya
#      buckets/ folder add karta hai (galti ho to bas wo folder delete)
#    * Robust reader: parquet/jsonl/json/csv/tsv, manifest/state skip,
#      dedup, ek file kharab -> sirf wahi skip (run zinda rehta hai)
#
#  KAHAN CHALE:  Kaggle / Colab (Internet ON). Sandbox me net OFF hai.
#    pip install -U huggingface_hub pandas pyarrow
#    import os
#    os.environ['HF_TOKEN'] = 'hf_xxx'            # WRITE token
#    # apne buckets do (best) ya auto-discover pe chhod do:
#    # os.environ['SRC_REPOS'] = 'user/ds1, user/ds2, ...'
#    # pehli baar preview: os.environ['DRY_RUN'] = '1'
#    !python merge_buckets.py
#
#  ENV KNOBS
#    HF_TOKEN              (required, WRITE)
#    DEST_REPO            default Damaru-ai/damru-knowledge
#    DEST_PREFIX          default buckets
#    SRC_REPOS            explicit list (comma/space separated) -- best
#    SRC_AUTHOR           default Damaru-ai (auto-discover fallback)
#    DRY_RUN              1 = sirf plan+count, upload nahi (default 0)
#    MAX_ROWS_PER_BUCKET  0 = all (default 0)
#    MAX_FILES_PER_BUCKET 0 = all (default 0)
#    MIN_LEN              default 20
# ============================================================================
import os, sys, json, time, hashlib, re

NL  = chr(10)
TAB = chr(9)

DEST_REPO   = os.environ.get('DEST_REPO', 'Damaru-ai/damru-knowledge')
DEST_PREFIX = os.environ.get('DEST_PREFIX', 'buckets').strip('/')
SRC_AUTHOR  = os.environ.get('SRC_AUTHOR', 'Damaru-ai')
HF_TOKEN    = os.environ.get('HF_TOKEN', '') or os.environ.get('HUGGINGFACE_TOKEN', '')
DRY_RUN     = os.environ.get('DRY_RUN', '0') == '1'
MAX_ROWS    = int(os.environ.get('MAX_ROWS_PER_BUCKET', '0') or 0)
MAX_FILES   = int(os.environ.get('MAX_FILES_PER_BUCKET', '0') or 0)
MIN_LEN     = int(os.environ.get('MIN_LEN', '20') or 20)

# --- apne buckets yahan bhi daal sakta hai (ya SRC_REPOS env) ---
BUCKETS = [
    # 'your-hf-username/bucket-1',
    # 'your-hf-username/bucket-2',
]

# infra repos -- kabhi merge mat karo
EXCLUDE = {
    'Damaru-ai/damru-knowledge',
    'Damaru-ai/damru-rag-index',
    'Damaru-ai/damru-14b-lora',
    'Damaru-ai/damru-14b-gguf',
    'Damaru-ai/damru-gguf',
    'Damaru-ai/damru-train',
    'Damaru-ai/damru-gurukul',
    'Damaru-ai/damru-oracle',
}

_SKIP_TOKENS = ('manifest', 'state', 'stats', '_meta', 'metadata', 'readme',
                'dataset_infos', 'gitattributes', 'checkpoint', 'progress',
                '_index', 'config', 'scorecard')
_DATA_EXT = ('.parquet', '.jsonl', '.ndjson', '.json', '.csv', '.tsv')
_Q_KEYS = ('question', 'prompt', 'instruction', 'problem', 'query', 'task',
           'input', 'title', 'text', 'content', 'body')
_A_KEYS = ('answer', 'output', 'response', 'completion', 'solution', 'target',
           'label', 'assistant', 'chosen', 'value')


def log(*a):
    print('[merge]', *a, flush=True)


def _lang(s):
    lo = chr(0x900)
    hi = chr(0x97F)
    for ch in s:
        if lo <= ch <= hi:
            return 'hi'
    return 'en'


def _is_data_file(path):
    low = str(path).lower()
    if not low.endswith(_DATA_EXT):
        return False
    base = low.rsplit('/', 1)[-1]
    for bad in _SKIP_TOKENS:
        if bad in base:
            return False
    return True


def _lowmap(row):
    m = {}
    for k in row.keys():
        if isinstance(k, str):
            m[k.lower()] = k
    return m


def _first_str(row, keys, lm=None):
    if not isinstance(row, dict):
        return ''
    lm = lm or _lowmap(row)
    for k in keys:
        real = lm.get(k)
        if real is None:
            continue
        v = row.get(real)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ''


def _from_messages(row):
    msgs = row.get('messages')
    if msgs is None:
        msgs = row.get('conversations')
    q, a = '', ''
    if isinstance(msgs, (list, tuple)):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or m.get('from') or '').lower()
            val = m.get('content')
            if val is None:
                val = m.get('value')
            if not isinstance(val, str):
                continue
            val = val.strip()
            if role in ('user', 'human') and not q:
                q = val
            elif role in ('assistant', 'gpt', 'bot', 'ai') and not a:
                a = val
    return q, a


def normalize(row, bucket, rel):
    if not isinstance(row, dict):
        return None
    lm = _lowmap(row)
    q = _first_str(row, _Q_KEYS, lm)
    a = _first_str(row, _A_KEYS, lm)
    instr = _first_str(row, ('instruction',), lm)
    inp = _first_str(row, ('input',), lm)
    if instr:
        q = instr + ((NL + NL + inp) if (inp and inp != instr) else '')
        if not a:
            a = _first_str(row, ('output', 'response', 'answer', 'completion'), lm)
    if (not q) or (not a):
        mq, ma = _from_messages(row)
        q = q or mq
        a = a or ma
    if not q:
        return None
    text = q if not a else (q + NL + a)
    if len(text.strip()) < MIN_LEN:
        return None
    short = bucket.split('/')[-1]
    return {
        'question': q,
        'answer': a,
        'text': text,
        'domain': short,
        'source': 'bucket:' + bucket + '::' + rel,
        'intent': 'qa',
        'lang': _lang(q),
        'bucket': bucket,
    }


def read_file_rows(local):
    low = str(local).lower()
    try:
        import pandas as pd
    except Exception as e:
        log('pandas missing:', e)
        return []
    try:
        if low.endswith('.parquet'):
            df = pd.read_parquet(local)
        elif low.endswith(('.jsonl', '.ndjson')):
            df = pd.read_json(local, lines=True)
        elif low.endswith('.json'):
            with open(local, 'r', encoding='utf-8', errors='ignore') as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                for key in ('data', 'rows', 'items', 'examples', 'problems'):
                    if isinstance(raw.get(key), list):
                        return [r for r in raw[key] if isinstance(r, dict)]
                return []
            if isinstance(raw, list):
                return [r for r in raw if isinstance(r, dict)]
            return []
        elif low.endswith('.tsv'):
            df = pd.read_csv(local, sep=TAB)
        else:
            df = pd.read_csv(local)
    except Exception as e:
        log('SKIP unreadable', local, '::', str(e)[:140])
        return []
    try:
        return df.to_dict(orient='records')
    except Exception as e:
        log('SKIP convert-failed', local, '::', str(e)[:140])
        return []


def discover(api):
    ids = []
    try:
        for d in api.list_datasets(author=SRC_AUTHOR):
            rid = getattr(d, 'id', None) or getattr(d, 'name', None)
            if rid:
                ids.append(rid)
    except Exception as e:
        log('auto-discover failed:', e)
    return ids


def resolve_buckets(api):
    lst = [b.strip() for b in BUCKETS if b.strip()]
    if not lst:
        env = os.environ.get('SRC_REPOS', '')
        if env.strip():
            lst = [x.strip() for x in env.replace(',', ' ').split() if x.strip()]
    if not lst:
        log('no explicit list -> auto-discovering under author=' + SRC_AUTHOR)
        lst = discover(api)
    out = []
    seen = set()
    for r in lst:
        if (r in EXCLUDE) or (r == DEST_REPO) or (r in seen):
            continue
        seen.add(r)
        out.append(r)
    return out


def process_bucket(api, hf_hub_download, repo, out_fh):
    try:
        files = api.list_repo_files(repo_id=repo, repo_type='dataset')
    except Exception as e:
        log('SKIP repo (list failed)', repo, '::', str(e)[:140])
        return None
    cands = sorted([f for f in files if _is_data_file(f)])
    if MAX_FILES > 0:
        cands = cands[:MAX_FILES]
    log('bucket=' + repo, '| files=' + str(len(files)), '| data=' + str(len(cands)))
    seen = set()
    n = 0
    for rel in cands:
        try:
            local = hf_hub_download(repo_id=repo, filename=rel,
                                    repo_type='dataset', token=HF_TOKEN or None)
        except Exception as e:
            log('  skip dl', rel, '::', str(e)[:120])
            continue
        for r in read_file_rows(local):
            rec = normalize(r, repo, rel)
            if not rec:
                continue
            key = hashlib.md5((rec['question'][:400] + '|' + rec['answer'][:200]).encode('utf-8', 'ignore')).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out_fh.write(json.dumps(rec, ensure_ascii=False) + NL)
            n += 1
            if MAX_ROWS > 0 and n >= MAX_ROWS:
                break
        log('  ' + rel + ' -> kept total=' + str(n))
        if MAX_ROWS > 0 and n >= MAX_ROWS:
            break
    return n


def main():
    if not HF_TOKEN:
        log('FATAL: HF_TOKEN missing (WRITE token chahiye).')
        sys.exit(1)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        log('FATAL: huggingface_hub import failed:', e)
        sys.exit(1)
    api = HfApi(token=HF_TOKEN)
    buckets = resolve_buckets(api)
    if not buckets:
        log('NO buckets resolved. BUCKETS list bharo / SRC_REPOS env set karo / SRC_AUTHOR theek karo.')
        sys.exit(1)
    log('==== RESOLVED ' + str(len(buckets)) + ' BUCKETS ====')
    for b in buckets:
        log('   - ' + b)
    tmpdir = '/tmp/damru_buckets'
    os.makedirs(tmpdir, exist_ok=True)
    grand = 0
    summary = []
    for repo in buckets:
        safe = re.sub('[^A-Za-z0-9_.-]', '_', repo)
        localp = os.path.join(tmpdir, safe + '.jsonl')
        with open(localp, 'w', encoding='utf-8') as fh:
            n = process_bucket(api, hf_hub_download, repo, fh)
        if n is None:
            summary.append((repo, 'ERROR', 0))
            continue
        grand += n
        dest = DEST_PREFIX + '/' + safe + '.jsonl'
        if n == 0:
            log('  (0 rows, skip upload) ' + repo)
            summary.append((repo, 'EMPTY', 0))
            continue
        if DRY_RUN:
            log('  [DRY_RUN] would upload ' + str(n) + ' rows -> ' + DEST_REPO + '/' + dest)
            summary.append((repo, 'DRY', n))
            continue
        try:
            api.upload_file(path_or_fileobj=localp, path_in_repo=dest,
                            repo_id=DEST_REPO, repo_type='dataset',
                            commit_message='merge bucket ' + repo + ' (' + str(n) + ' rows)')
            log('  UPLOADED -> ' + DEST_REPO + '/' + dest)
            summary.append((repo, 'OK', n))
        except Exception as e:
            log('  UPLOAD FAILED ' + dest + ' :: ' + str(e)[:160])
            summary.append((repo, 'UPLOAD_FAIL', n))
    log('================= SUMMARY =================')
    for repo, st, n in summary:
        log('  %-42s %-12s %d' % (repo, st, n))
    log('  TOTAL rows = ' + str(grand))
    if (not DRY_RUN) and grand > 0:
        man = {
            'merged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'dest_repo': DEST_REPO,
            'dest_prefix': DEST_PREFIX,
            'total_rows': grand,
            'buckets': [{'repo': r, 'status': s, 'rows': n} for r, s, n in summary],
        }
        try:
            mp = os.path.join(tmpdir, '_buckets_manifest.json')
            with open(mp, 'w', encoding='utf-8') as fh:
                json.dump(man, fh, ensure_ascii=False, indent=2)
            api.upload_file(path_or_fileobj=mp,
                            path_in_repo=DEST_PREFIX + '/_buckets_manifest.json',
                            repo_id=DEST_REPO, repo_type='dataset',
                            commit_message='bucket merge manifest')
            log('  manifest uploaded -> ' + DEST_PREFIX + '/_buckets_manifest.json')
        except Exception as e:
            log('  manifest upload skip: ' + str(e)[:120])
    log('DONE. Next: RAG index rebuild (damru-rag-index) taaki buckets search me aaye.')


if __name__ == '__main__':
    main()
