/* ============================================================
   DAMRU AI — Distillation cron (Phase 3, v2 high-yield)
   A "teacher AI" generates high-quality Q&A and stores them in
   Supabase (damru_knowledge) so Damru's brain grows automatically.
   Improvements: more topics/run, robust JSON parsing, retries.
   No npm dependencies (Node 20 global fetch).
   ============================================================ */

const SUPABASE_URL   = process.env.SUPABASE_URL;
const SUPABASE_KEY   = process.env.SUPABASE_KEY;     // anon or service_role
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;   // optional -> Pollinations fallback

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing SUPABASE_URL / SUPABASE_KEY secrets.');
  process.exit(1);
}

const TOPICS = [
  ['general', 'interesting general knowledge facts about the world'],
  ['exam',    'Indian competitive exam questions (RAS, SSC, UPSC, CET, REET) with answers'],
  ['general', 'Indian polity and constitution questions and answers'],
  ['general', 'geography of India and the world'],
  ['general', 'history of India and important world events'],
  ['general', 'English grammar and vocabulary explained with examples'],
  ['general', 'logical reasoning and aptitude questions with solutions'],
  ['general', 'general awareness / current-affairs style questions'],
  ['general', 'everyday how-to and practical life advice'],
  ['general', 'technology, AI and computer fundamentals'],
  ['general', 'health, nutrition and fitness basics'],
  ['general', 'environment, ecology and climate questions'],
  ['code',    'common programming questions in Python and JavaScript with code'],
  ['code',    'data structures and algorithms questions with code'],
  // --- Ethics, values & logic (persona builders) ---
  ['general', 'ethics, duty and life lessons from the Bhagavad Gita applied to real-life situations like handling failure, stress, focus and decision-making, presented respectfully as practical wisdom (not as religious authority)'],
  ['general', 'moral and logical lessons from Panchatantra and Hitopadesha stories, with the takeaway clearly explained'],
  ['general', 'world philosophy and ethics: compare different viewpoints fairly and give a balanced, practical conclusion'],
  ['general', 'critical thinking, reasoning and good decision-making explained with real-life examples'],
  // --- Hard sciences ---
  ['general', 'physics concepts and numerical problems with step-by-step solutions (mechanics, electricity, optics, thermodynamics, modern physics)'],
  ['general', 'chemistry concepts and reactions explained with examples (physical, organic, inorganic)'],
  ['general', 'biology concepts explained clearly (human body, cells, genetics, ecology, evolution)'],
  ['general', 'astronomy and the universe explained simply (stars, planets, galaxies, black holes)'],
  ['general', 'aerospace and space science explained clearly (rockets, orbits, aerodynamics, propulsion, famous space missions)'],
  ['general', 'engineering fundamentals explained simply (mechanical, electrical, civil, aerospace, computer)'],
  ['general', 'real-life applications of science: how everyday things and technology actually work'],
  // --- Maths & economics ---
  ['math',    'class 10-12 level algebra, trigonometry and calculus problems with full solutions'],
  ['math',    'advanced mathematics problems with full step-by-step solutions (calculus, linear algebra, probability, statistics)'],
  ['general', 'economics and personal finance concepts for students explained with examples'],
  ['general', 'business, entrepreneurship and money management basics explained simply']
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
          body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], temperature: 0.8, max_tokens: 2200 })
        });
        if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
      } catch (e) { /* next model */ }
    }
  }
  try {
    const r = await fetch('https://text.pollinations.ai/openai', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'openai', messages: [{ role: 'user', content: prompt }], temperature: 0.8 })
    });
    if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
  } catch (e) {}
  return null;
}

/* Loose extractor: pulls question/answer objects even from messy output */
function loosePairs(text) {
  const out = [];
  const re = /\{[^{}]*?"question"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*?"answer"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*?\}/g;
  let m;
  while ((m = re.exec(text))) {
    try { out.push({ question: JSON.parse('"' + m[1] + '"'), answer: JSON.parse('"' + m[2] + '"') }); } catch (e) {}
  }
  return out;
}

function extractJSON(text) {
  if (!text) return [];
  let t = text.replace(/```json/gi, '```').replace(/```/g, '');
  const s = t.indexOf('['), e = t.lastIndexOf(']');
  if (s >= 0 && e > s) {
    let body = t.slice(s, e + 1).replace(/,(\s*[\]}])/g, '$1'); // strip trailing commas
    try { const arr = JSON.parse(body); if (Array.isArray(arr) && arr.length) return arr; } catch (err) {}
  }
  return loosePairs(t); // fallback
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
  const topics = pick(TOPICS, 5);
  let total = 0;
  for (const [intent, topic] of topics) {
    const prompt =
      'You are an expert teacher creating training data for an AI assistant. ' +
      'Generate 6 high-quality, diverse and factually accurate Q&A pairs about: ' + topic + '. ' +
      'Each answer must be correct, clear and self-contained (3-8 sentences; include steps or code where useful). ' +
      'Return ONLY a valid JSON array, exactly like: [{"question":"...","answer":"..."}]. No markdown, no extra text.';
    let pairs = [];
    for (let attempt = 0; attempt < 2 && pairs.length === 0; attempt++) {
      const out = await teacher(prompt);
      pairs = extractJSON(out);
    }
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
