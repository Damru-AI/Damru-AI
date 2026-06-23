/* ============================================================
   DAMRU AI — Book / Text Ingester (RAG knowledge builder)
   Reads a plain-text file (a book, chapter, notes, Gita, NCERT,
   OpenStax, etc.), splits it into chunks, asks a teacher AI to
   turn each chunk into clean Q&A pairs, and stores them in
   Supabase (damru_knowledge). Damru then reads these via kbSearch.

   Usage (local or GitHub Actions):
     node ingest/ingest.js <file> <intent> <maxChunks>
   Example:
     node ingest/ingest.js ingest/sources/gita.txt general 60

   No npm dependencies (Node 20+ global fetch).
   ============================================================ */

const fs = require('fs');
const { execFileSync } = require('child_process');

const SUPABASE_URL   = process.env.SUPABASE_URL;
const SUPABASE_KEY   = process.env.SUPABASE_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;   // optional -> Pollinations fallback

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing SUPABASE_URL / SUPABASE_KEY secrets.');
  process.exit(1);
}

const FILE       = process.argv[2] || 'ingest/sources/source.txt';
const INTENT     = process.argv[3] || 'general';
const MAX_CHUNKS = parseInt(process.argv[4] || '60', 10);
const CHUNK_SIZE = 1500;   // characters per chunk
const QA_PER     = 6;      // Q&A pairs requested per chunk

if (!fs.existsSync(FILE)) {
  console.error('File not found: ' + FILE);
  process.exit(1);
}

const FREE_MODELS = [
  'deepseek/deepseek-chat-v3-0324:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'google/gemini-2.0-flash-exp:free',
  'qwen/qwen-2.5-72b-instruct:free'
];

/* Split text into ~CHUNK_SIZE chunks, preferring paragraph breaks */
function chunkText(text) {
  const paras = text.replace(/\r/g, '').split(/\n\s*\n/);
  const chunks = [];
  let buf = '';
  for (const p of paras) {
    const para = p.trim();
    if (!para) continue;
    if ((buf + '\n\n' + para).length > CHUNK_SIZE && buf) {
      chunks.push(buf.trim());
      buf = para;
    } else {
      buf = buf ? (buf + '\n\n' + para) : para;
    }
  }
  if (buf.trim()) chunks.push(buf.trim());
  // hard-split any oversized chunk
  const out = [];
  for (const c of chunks) {
    if (c.length <= CHUNK_SIZE * 1.5) { out.push(c); continue; }
    for (let i = 0; i < c.length; i += CHUNK_SIZE) out.push(c.slice(i, i + CHUNK_SIZE));
  }
  return out;
}

async function teacher(prompt) {
  if (OPENROUTER_KEY) {
    for (const model of FREE_MODELS) {
      try {
        const r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + OPENROUTER_KEY, 'Content-Type': 'application/json', 'X-Title': 'Damru Ingest' },
          body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], temperature: 0.5, max_tokens: 2200 })
        });
        if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
      } catch (e) { /* next model */ }
    }
  }
  try {
    const r = await fetch('https://text.pollinations.ai/openai', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'openai', messages: [{ role: 'user', content: prompt }], temperature: 0.5 })
    });
    if (r.ok) { const j = await r.json(); const t = j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content; if (t) return t; }
  } catch (e) {}
  return null;
}

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
    let body = t.slice(s, e + 1).replace(/,(\s*[\]}])/g, '$1');
    try { const arr = JSON.parse(body); if (Array.isArray(arr) && arr.length) return arr; } catch (err) {}
  }
  return loosePairs(t);
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

/* Strip HTML/XML/SVG tags & entities; auto-detects markup files */
function sanitizeText(raw) {
  let t = raw || '';
  // remove script/style blocks fully
  t = t.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ');
  // strip HTML/XML/SVG comments
  t = t.replace(/<!--[\s\S]*?-->/g, ' ');
  // strip DOCTYPE
  t = t.replace(/<!DOCTYPE[^>]*>/gi, ' ');
  // count tags BEFORE stripping to know if it was markup
  const tagCount = (t.match(/<[a-zA-Z!\/][^>]*>/g) || []).length;
  const isMarkup = tagCount > 20;
  if (isMarkup) console.log('[sanitize] Detected markup file (' + tagCount + ' tags). Stripping HTML/SVG/XML...');
  // strip all remaining tags
  t = t.replace(/<[^>]+>/g, ' ');
  // decode common entities
  t = t.replace(/&nbsp;/gi, ' ').replace(/&amp;/gi, '&').replace(/&lt;/gi, '<').replace(/&gt;/gi, '>').replace(/&quot;/gi, '"').replace(/&#39;/gi, "'").replace(/&[a-z]+;/gi, ' ').replace(/&#\d+;/g, ' ');
  // collapse whitespace
  t = t.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n');
  return t.trim();
}

/* True if a chunk still looks like markup or junk (after sanitize) */
function isJunkChunk(chunk) {
  if (!chunk || chunk.length < 200) return true;
  const letters = (chunk.match(/[A-Za-z\u0900-\u097F]/g) || []).length;
  if (letters / chunk.length < 0.55) return true; // too few real letters
  const codey = (chunk.match(/[{}<>\/\\;=]/g) || []).length;
  if (codey / chunk.length > 0.08) return true; // too many code-like chars
  return false;
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

/* Read a single source file. Supports .txt/.md/.pdf (uses pdftotext for PDFs) */
function readSource(filePath) {
  if (/\.pdf$/i.test(filePath)) {
    try {
      console.log('[pdf] Converting: ' + filePath);
      return execFileSync('pdftotext', [filePath, '-'], { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
    } catch (e) { console.error('[pdf] pdftotext failed for ' + filePath + ': ' + e.message); return ''; }
  }
  return fs.readFileSync(filePath, 'utf8');
}

/* List files: accepts a single file OR a folder (then scans for .txt/.md/.pdf) */
function listFiles(target) {
  const st = fs.statSync(target);
  if (st.isFile()) return [target];
  if (st.isDirectory()) {
    const base = target.replace(/\/$/, '');
    return fs.readdirSync(base)
      .filter(n => /\.(txt|md|pdf)$/i.test(n))
      .map(n => base + '/' + n)
      .sort();
  }
  return [];
}

async function processFile(path) {
  console.log('\n=== File: ' + path + ' ===');
  const raw = readSource(path);
  if (!raw) return 0;
  const cleaned = sanitizeText(raw);
  console.log('Raw chars: ' + raw.length + ' | cleaned chars: ' + cleaned.length);
  let chunks = chunkText(cleaned).filter(c => !isJunkChunk(c));
  console.log('Usable chunks: ' + chunks.length + ' (cap ' + MAX_CHUNKS + ')');
  if (chunks.length === 0) return 0;
  if (chunks.length > MAX_CHUNKS) chunks = chunks.slice(0, MAX_CHUNKS);

  let total = 0;
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const prompt =
      'You are an expert teacher creating training data for an AI assistant. ' +
      'Read the SOURCE TEXT below and create ' + QA_PER + ' high-quality Q&A pairs that capture its key knowledge, concepts, lessons or wisdom. ' +
      'Questions should be natural things a student might ask. Answers must be accurate, clear and self-contained (3-8 sentences), based on the source text. ' +
      'If the text is philosophical or ethical, frame answers as practical, balanced real-life guidance. ' +
      'Return ONLY a valid JSON array exactly like: [{"question":"...","answer":"..."}]. No markdown, no extra text.\n\n' +
      'SOURCE TEXT:\n"""\n' + chunk + '\n"""';
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
        const ok = await saveQA(q, String(p.answer).slice(0, 4000), INTENT);
        if (ok) { saved++; total++; }
      }
    }
    console.log('Chunk ' + (i + 1) + '/' + chunks.length + ' -> saved ' + saved + ' Q&A');
  }
  console.log('Subtotal for ' + path + ': ' + total + ' Q&A');
  return total;
}

(async () => {
  let targets = [];
  try { targets = listFiles(FILE); } catch (e) { console.error('Path error: ' + e.message); process.exit(1); }
  if (!targets.length) { console.error('No usable files at: ' + FILE); process.exit(1); }
  console.log('Targets: ' + targets.length + ' file(s) -> ' + targets.join(', '));
  let grand = 0;
  for (const p of targets) grand += await processFile(p);
  console.log('\nGRAND TOTAL new Q&A saved: ' + grand);
})();
