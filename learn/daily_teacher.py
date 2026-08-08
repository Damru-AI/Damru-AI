#!/usr/bin/env python3
'''
Damru Daily Direct-Teacher
Writes a version-controlled daily learning corpus into git at
learn/daily/YYYY-MM-DD.jsonl and optionally pushes it to the HF knowledge
dataset so Damru RAG brain ingests it.
Record schema matches rag.py: question, answer, domain, source, intent, lang.
Honest scale: ~50k lines/day is produced by the automated cron using an LLM
API (GROQ_API_KEY). Without a key the curated seed packs are written.
'''
import os, sys, json, glob, time, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARN_DIR = os.environ.get('LEARN_DIR', os.path.join(ROOT, 'learn'))
DAILY_DIR = os.path.join(LEARN_DIR, 'daily')
TARGET = int(os.environ.get('DAILY_TARGET_LINES', '50000'))
HF_REPO = os.environ.get('HF_REPO', 'Damaru-ai/damru-knowledge')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
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

def log(*a):
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

def groq_generate(subject_key, subject_desc, n_batches, per_batch):
    if not GROQ_API_KEY:
        return []
    try:
        import requests
    except Exception:
        return []
    url = 'https://api.groq.com/openai/v1/chat/completions'
    headers = {'Authorization': 'Bearer ' + GROQ_API_KEY, 'Content-Type': 'application/json'}
    out = []
    sysp = ('You are a master teacher building a training dataset for an AI named Damru. '
            'Return ONLY a JSON array of objects with keys question and answer. '
            'Answers must be correct, detailed 8 to 40 lines and self contained. '
            'No code fences, only raw JSON.')
    for b in range(n_batches):
        user = ('Subject: ' + subject_desc + '. Create ' + str(per_batch) +
                ' diverse non repeating question answer pairs. Vary difficulty from '
                'beginner to advanced. Mix Hindi and English questions naturally. '
                'Return a JSON array only.')
        body = {'model': GROQ_MODEL,
                'messages': [{'role': 'system', 'content': sysp},
                             {'role': 'user', 'content': user}],
                'temperature': 0.8, 'max_tokens': 8000}
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code != 200:
                log('groq status', r.status_code, r.text[:160])
                time.sleep(2)
                continue
            txt = r.json()['choices'][0]['message']['content'].strip()
            fence = chr(96) * 3
            if txt.startswith(fence):
                txt = txt.strip(chr(96))
                nl = txt.find(NL)
                if nl != -1:
                    txt = txt[nl + 1:]
            arr = json.loads(txt)
            for o in arr:
                q = (o.get('question') or '').strip()
                a = (o.get('answer') or '').strip()
                if q and a:
                    lang = 'en'
                    for c in q:
                        if 0x900 <= ord(c) <= 0x97f:
                            lang = 'hi'
                            break
                    out.append(rec(q, a, subject_key, lang=lang))
        except Exception as e:
            log('groq batch failed', e)
            time.sleep(2)
            continue
    return out

def approx_lines(records):
    n = 0
    for o in records:
        n += 1 + (o.get('answer', '').count(NL) + 1)
    return n

def main():
    os.makedirs(DAILY_DIR, exist_ok=True)
    day = today_str()
    subj_key, subj_desc = pick_subject()
    log('day', day, 'subject', subj_key, 'target_lines', TARGET)
    records = []
    seeds = load_seed_packs()
    records.extend([s for s in seeds if s.get('domain') == subj_key])
    records.extend([s for s in seeds if s.get('domain') != subj_key])
    log('seed records', len(records))
    if GROQ_API_KEY:
        per_batch = 25
        need = max(0, TARGET - approx_lines(records))
        n_batches = max(1, need // (per_batch * 13))
        if n_batches > 400:
            n_batches = 400
        log('groq batches', n_batches)
        gen = groq_generate(subj_key, subj_desc, n_batches, per_batch)
        records.extend(gen)
        log('generated records', len(gen))
    else:
        log('no GROQ_API_KEY, writing curated seed only; add secret for large daily scale')
    seen = set()
    uniq = []
    for o in records:
        k = ' '.join((o.get('question') or '').lower().split())
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(o)
    out_path = os.path.join(DAILY_DIR, day + '.jsonl')
    f = open(out_path, 'w', encoding='utf-8')
    for o in uniq:
        f.write(json.dumps(o, ensure_ascii=False) + NL)
    f.close()
    lines = approx_lines(uniq)
    log('wrote', out_path, 'records', len(uniq), 'approx_lines', lines)
    man_path = os.path.join(LEARN_DIR, 'daily_manifest.json')
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except Exception:
            man = {}
    man[day] = {'subject': subj_key, 'records': len(uniq), 'approx_lines': lines}
    json.dump(man, open(man_path, 'w'), ensure_ascii=False, indent=2)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=out_path,
                            path_in_repo='daily/' + day + '.jsonl',
                            repo_id=HF_REPO, repo_type='dataset',
                            commit_message='daily-teach ' + day + ' ' + subj_key)
            log('pushed to HF', HF_REPO)
        except Exception as e:
            log('HF push failed', e)
    else:
        log('no HF_TOKEN, skipped HF push; git commit still connects the file')
    print('DAILY_TEACH_OK records=' + str(len(uniq)) + ' lines=' + str(lines) + ' subject=' + subj_key)

if __name__ == '__main__':
    main()
