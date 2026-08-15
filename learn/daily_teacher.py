#!/usr/bin/env python3
'''
Damru Daily Direct-Teacher v2 (1M+ lines/day)
Writes version-controlled daily learning corpus into git at
learn/daily/YYYY-MM-DD.jsonl and pushes to HF knowledge dataset.
Record schema: question, answer, domain, source, intent, lang.

Scale: ~1M lines/day produced by automated cron using Groq API.
Set GROQ_API_KEY. Without keys only curated seed packs written.

Features:
- Parallel batch processing (20+ concurrent requests)
- Quality filtering and deduplication
- Multi-language support (Hindi + English)
- Progress tracking and retry logic
'''
import os, sys, json, glob, time, datetime, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARN_DIR = os.environ.get('LEARN_DIR', os.path.join(ROOT, 'learn'))
DAILY_DIR = os.path.join(LEARN_DIR, 'daily')
TARGET = int(os.environ.get('DAILY_TARGET_LINES', '1000000'))
HF_REPO = os.environ.get('HF_REPO', 'Damaru-ai/damru-knowledge')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
MAX_WORKERS = int(os.environ.get('PARALLEL_WORKERS', '20'))
NL = chr(10)

SUBJECTS = [
    ('human_behaviour', 'human behaviour, emotions, social cues, empathy and real conversation'),
    ('psychology', 'practical psychology, motivation, habits and cognitive biases'),
    ('conversation', 'natural conversation, small talk, active listening and de-escalation'),
    ('coding', 'clean production code, algorithms, debugging and system design'),
    ('mathematics', 'mathematics from arithmetic to calculus with worked steps'),
    ('science', 'physics, chemistry and biology concepts explained simply'),
    ('india_gk', 'India general knowledge, history, polity, geography and economy'),
    ('life_skills', 'decision making, money basics, health, productivity and communication'),
    ('language', 'English and Hindi language, grammar, vocabulary and translation'),
    ('reasoning', 'logical reasoning, puzzles and step by step problem solving'),
]

lock = threading.Lock()
stats = {'total': 0, 'success': 0, 'failed': 0, 'batches': 0}

def log(*a):
    with lock:
        print('[daily-teacher]', *a, flush=True)

def today_str():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')

def rec(q, a, domain, intent='qa', lang='en', source='damru-daily-teach'):
    return {'question': q, 'answer': a, 'domain': domain, 'source': source, 'intent': intent, 'lang': lang}

def load_seed_packs():
    out = []
    for path in sorted(glob.glob(os.path.join(LEARN_DIR, '*.jsonl'))):
        try:
            f = open(path, 'r', encoding='utf-8')
        except Exception as e:
            log('seed open failed', path, e)
            continue
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('question') and o.get('answer'):
                o.setdefault('domain', 'general')
                o.setdefault('source', 'damru-seed')
                o.setdefault('intent', 'qa')
                o.setdefault('lang', 'en')
                out.append(o)
        f.close()
    return out

def pick_subject():
    doy = datetime.date.today().timetuple().tm_yday
    return SUBJECTS[doy % len(SUBJECTS)]

def active_provider():
    if GROQ_API_KEY:
        return 'groq'
    return ''

def detect_language(text):
    """Detect if text contains Hindi characters"""
    for c in text[:200]:  # Check first 200 chars
        if 0x900 <= ord(c) <= 0x97f:
            return 'hi'
    return 'en'

def groq_batch_generate(subject_key, subject_desc, per_batch):
    """Generate one batch from Groq API with timeout and retry"""
    if not GROQ_API_KEY:
        return []
    
    try:
        import requests
    except Exception:
        return []
    
    url = 'https://api.groq.com/openai/v1/chat/completions'
    headers = {
        'Authorization': 'Bearer ' + GROQ_API_KEY,
        'Content-Type': 'application/json'
    }
    
    sysp = ('You are a master teacher building a training dataset for Damru AI. '
            'Return ONLY a JSON array of objects with keys: question, answer. '
            'Each answer must be 10-50 lines, correct, detailed and self-contained. '
            'No markdown, no code fences, pure JSON only. '
            'Make questions diverse, varying difficulty beginner to advanced. '
            'Include both English and Hindi questions naturally.')
    
    user = (f'Subject: {subject_desc}. '
            f'Generate {per_batch} unique, non-repeating question-answer pairs. '
            f'Mix Hindi and English. Return ONLY the JSON array.')
    
    body = {
        'model': GROQ_MODEL,
        'messages': [
            {'role': 'system', 'content': sysp},
            {'role': 'user', 'content': user}
        ],
        'temperature': 0.7,
        'max_tokens': 16000,
        'top_p': 0.95
    }
    
    out = []
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            
            if r.status_code == 429:  # Rate limit
                wait_time = int(r.headers.get('retry-after', 30))
                log(f'Rate limited. Waiting {wait_time}s...')
                time.sleep(min(wait_time, 60))
                retry_count += 1
                continue
            
            if r.status_code != 200:
                log(f'Groq error {r.status_code}: {r.text[:100]}')
                retry_count += 1
                time.sleep(2 ** retry_count)
                continue
            
            txt = r.json()['choices'][0]['message']['content'].strip()
            
            # Clean up code fences
            if txt.startswith('```'):
                txt = txt.strip('`')
                nl = txt.find(NL)
                if nl != -1:
                    txt = txt[nl + 1:]
            
            # Extract JSON
            if not txt.lstrip().startswith('['):
                i = txt.find('[')
                j = txt.rfind(']')
                if i != -1 and j != -1 and j > i:
                    txt = txt[i:j + 1]
            
            arr = json.loads(txt)
            
            for o in arr:
                q = (o.get('question') or '').strip()
                a = (o.get('answer') or '').strip()
                if q and len(q) > 5 and a and len(a) > 20:
                    lang = detect_language(q)
                    out.append(rec(q, a, subject_key, lang=lang))
            
            with lock:
                stats['success'] += 1
            return out
            
        except json.JSONDecodeError as e:
            log(f'JSON parse error: {e}')
            retry_count += 1
            time.sleep(2 ** retry_count)
        except Exception as e:
            log(f'Groq batch error: {e}')
            retry_count += 1
            time.sleep(2 ** retry_count)
    
    with lock:
        stats['failed'] += 1
    return out

def llm_generate_parallel(subject_key, subject_desc, n_batches, per_batch):
    """Generate batches in parallel using ThreadPoolExecutor"""
    if not active_provider():
        return []
    
    out = []
    log(f'Starting parallel generation: {n_batches} batches, {per_batch} per batch')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for b in range(n_batches):
            future = executor.submit(groq_batch_generate, subject_key, subject_desc, per_batch)
            futures.append(future)
        
        completed = 0
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                out.extend(batch_results)
                completed += 1
                with lock:
                    stats['batches'] += 1
                    stats['total'] += len(batch_results)
                if completed % 5 == 0:
                    log(f'Progress: {completed}/{n_batches} batches, {len(out)} records')
            except Exception as e:
                log(f'Future failed: {e}')
                completed += 1
    
    return out

def approx_lines(records):
    n = 0
    for o in records:
        q_lines = max(1, o.get('question', '').count(NL) + 1)
        a_lines = max(1, o.get('answer', '').count(NL) + 1)
        n += q_lines + a_lines + 2  # 2 for separators
    return n

def main():
    os.makedirs(DAILY_DIR, exist_ok=True)
    day = today_str()
    subj_key, subj_desc = pick_subject()
    
    log(f'=== Damru Daily Teacher (1M lines/day) ===')
    log(f'Date: {day}')
    log(f'Subject: {subj_key}')
    log(f'Target Lines: {TARGET}')
    log(f'Parallel Workers: {MAX_WORKERS}')
    
    # Load seed packs
    records = []
    seeds = load_seed_packs()
    records.extend([s for s in seeds if s.get('domain') == subj_key])
    records.extend([s for s in seeds if s.get('domain') != subj_key])
    log(f'Loaded {len(records)} seed records')
    
    # Generate new records via LLM
    prov = active_provider()
    if prov:
        per_batch = 50  # Increased from 25
        need = max(0, TARGET - approx_lines(records))
        
        # Calculate batches: assume ~650 lines per batch (50 QA pairs * ~13 lines each)
        n_batches = max(1, need // 650)
        n_batches = min(n_batches, 2000)  # Cap at 2000 to avoid infinite loops
        
        log(f'LLM Provider: {prov}')
        log(f'Need ~{need} lines')
        log(f'Generating {n_batches} batches of {per_batch} QA pairs')
        log(f'ETA: ~{max(1, n_batches // MAX_WORKERS)} minutes')
        
        start_time = time.time()
        gen = llm_generate_parallel(subj_key, subj_desc, n_batches, per_batch)
        elapsed = time.time() - start_time
        
        records.extend(gen)
        log(f'Generated {len(gen)} new records in {elapsed:.1f}s')
        log(f'Stats: success={stats["success"]}, failed={stats["failed"]}, batches={stats["batches"]}')
    else:
        log('No LLM key set, using seed packs only')
    
    # Deduplication
    seen = set()
    uniq = []
    for o in records:
        k = ' '.join((o.get('question') or '').lower().split()[:10])  # Hash first 10 words
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(o)
    
    log(f'Deduplicated: {len(records)} -> {len(uniq)} unique records')
    
    # Write to disk
    out_path = os.path.join(DAILY_DIR, day + '.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for o in uniq:
            f.write(json.dumps(o, ensure_ascii=False) + NL)
    
    lines = approx_lines(uniq)
    log(f'Wrote {out_path}')
    log(f'Records: {len(uniq)}, Approx Lines: {lines}')
    
    # Update manifest
    man_path = os.path.join(LEARN_DIR, 'daily_manifest.json')
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except Exception:
            man = {}
    
    man[day] = {
        'subject': subj_key,
        'records': len(uniq),
        'approx_lines': lines,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    with open(man_path, 'w') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    
    # Push to HuggingFace
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo='daily/' + day + '.jsonl',
                repo_id=HF_REPO,
                repo_type='dataset',
                commit_message=f'daily-teach {day} {subj_key} {lines}L'
            )
            log(f'Pushed to HF: {HF_REPO}')
        except Exception as e:
            log(f'HF push failed: {e}')
    else:
        log('No HF_TOKEN, skipped HF push')
    
    print(f'DAILY_TEACH_OK records={len(uniq)} lines={lines} subject={subj_key}')

if __name__ == '__main__':
    main()
