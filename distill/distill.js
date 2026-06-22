/* ============================================================
   DAMRU AI — Distillation cron (Phase 3)
   A "teacher AI" generates high-quality Q&A and stores them in
   Supabase (damru_knowledge) so Damru's brain grows automatically.
   Runs on GitHub Actions (free). No npm dependencies (Node 20 fetch).
   ============================================================ */

const SUPABASE_URL  = process.env.SUPABASE_URL;
const SUPABASE_KEY  = process.env.SUPABASE_KEY;      // anon or service_role
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;   // optional -> falls back to Pollinations

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing SUPABASE_URL / SUPABASE_KEY secrets.');
  process.exit(1);
}

/* Topics Damru should keep learning (intent, topic) */
const TOPICS = [
  ['general', 'interesting general knowledge facts about the world'],
  ['general', 'science concepts explained simply (physics, chemistry, biology)'],
  ['math',    'useful mathematics problems with full step-by-step solutions'],
  ['code',    'common programming questions in Python and JavaScript with code'],
  ['exam',    'Indian competitive exam questions (RAS, SSC, UPSC, CET, REET) with answers'],
  ['general', 'Indian polity and constitution questions and answers'],
  ['general', 'geography of India and the world'],
  ['general', 'history of India and important world events'],
  ['general', 'English grammar and vocabulary explained with examples'],
  ['general', 'logical reasoning and aptitude questions with solutions'],
  ['general', 'general awareness / current-affairs style questions'],
  ['general', 'everyday how-to and practical life advice'],
  ['general', 'technology, AI and computer fundamentals'],
  ['general', 'health, nutrition and fitness basics']
];

const FREE_MODELS = [
  'deepseek/deepseek-chat-v3-0324:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'google/gemini-2.0-flash-exp:free',
  'qwen/qwen-2.5-72b-instruct:free'
];

function pick(arr, n) {
  const c = [...arr], out = [];
  while (out.length < n && c.length) out.push(c.splice(Math.floor(Math.random() * c.length), 1)[0]);
  return out;
}

async function teacher(prompt) {
  if (OPENROUTER_KEY) {
    for (const model of FREE_MODELS) {
      try {
        const r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + OPENROUTER_KEY, 'Content-Type': 'application/json', 'X-Title': 'Damru Distill' },
          body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], temperature: 0.7, max_tokens: 2000 })
        });
        if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
      } catch (e) { /* try next model */ }
    }
  }
  try {
    const r = await fetch('https://text.pollinations.ai/openai', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'openai', messages: [{ role: 'user', content: prompt }], temperature: 0.7 })
    });
    if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
  } catch (e) {}
  return null;
}

function extractJSON(text) {
  if (!text) return [];
  let t = text.replace(/```json/gi, '```').replace(/```/g, '');
  const s = t.indexOf('['), e = t.lastIndexOf(']');
  if (s < 0 || e < 0) return [];
  try { const arr = JSON.parse(t.slice(s, e + 1)); return Array.isArray(arr) ? arr : []; } catch (err) { return []; }
}

async function exists(question) {
  try {
    const u = SUPABASE_URL + '/rest/v1/damru_knowledge?select=id&question=eq.' + encodeURIComponent(question) + '&limit=1';
    const r = await fetch(u, { headers: { 'apikey': SUPABASE_KEY, 'Authorization': 'Bearer ' + SUPABASE_KEY } });
    if (!r.ok) return false;
    const j = await r.json();
    return Array.isArray(j) && j.length > 0;
  } catch (e) { return false; }
}

async function saveQA(question, answer, intent) {
  try {
    const r = await fetch(SUPABASE_URL + '/rest/v1/damru_knowledge', {
      method: 'POST',
      headers: { 'apikey': SUPABASE_KEY, 'Authorization': 'Bearer ' + SUPABASE_KEY, 'Content-Type': 'application/json', 'Prefer': 'return=minimal' },
      body: JSON.stringify({ question, answer, intent, lang: 'en' })
    });
    return r.ok;
  } catch (e) { return false; }
}

(async () => {
  const topics = pick(TOPICS, 3);
  let total = 0;
  for (const [intent, topic] of topics) {
    const prompt =
      'You are an expert teacher creating training data for an AI assistant. ' +
      'Generate 5 high-quality, diverse and factually accurate Q&A pairs about: ' + topic + '. ' +
      'Each answer must be correct, clear and self-contained (3-8 sentences; include steps or code where useful). ' +
      'Return ONLY a valid JSON array like [{"question":"...","answer":"..."}] with no extra text.';
    const out = await teacher(prompt);
    const pairs = extractJSON(out);
    let saved = 0;
    for (const p of pairs) {
      if (p && p.question && p.answer && String(p.answer).length > 40) {
        const q = String(p.question).slice(0, 500);
        if (await exists(q)) continue;
        const ok = await saveQA(q, String(p.answer).slice(0, 4000), intent);
        if (ok) { saved++; total++; }
      }
    }
    console.log('Topic [' + topic + '] -> saved ' + saved + ' Q&A');
  }
  console.log('DONE. Total new Q&A saved: ' + total);
})();
