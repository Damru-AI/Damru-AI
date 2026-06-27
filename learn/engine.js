/* ============================================================
   DAMRU AI — Continuous Learning Engine ("har minute sikho")
   A long-running worker. Every CYCLE_MS (default 60s) it runs one
   learn-cycle. Each cycle launches 4 learners IN PARALLEL:
     1) bookLearner    — reads the NEXT chunk of a public-domain book
     2) wikiLearner    — random Wikipedia articles (general knowledge)
     3) journalLearner — arXiv abstracts (science journals + maths)
     4) mathLearner    — generates & SOLVES advanced maths problems
   Plus it computes a SEMANTIC EMBEDDING for every new row and
   gradually BACKFILLS embeddings for older rows, so the browser can
   do meaning-based (semantic) retrieval via Supabase pgvector.
   Knowledge becomes DAMRU'S OWN brain (Supabase damru_knowledge).
   Runs RUN_MINUTES (default 330 = 5.5h) then exits; the workflow
   re-dispatches itself => near-24/7 loop.
   Deps: @xenova/transformers (installed by the workflow).
   ============================================================ */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;
if(!SUPABASE_URL||!SUPABASE_KEY){ console.error('Missing SUPABASE_URL / SUPABASE_KEY'); process.exit(1); }

const CYCLE_MS = parseInt(process.env.CYCLE_MS||'60000',10);
const RUN_MINUTES = parseInt(process.env.RUN_MINUTES||'330',10);
const QA_PER = parseInt(process.env.QA_PER||'5',10);

const FREE_MODELS = ['deepseek/deepseek-chat-v3-0324:free','meta-llama/llama-3.3-70b-instruct:free','google/gemini-2.0-flash-exp:free','qwen/qwen-2.5-72b-instruct:free'];

const sleep = function(ms){ return new Promise(function(r){ setTimeout(r,ms); }); };
function ws(x){ return (x||'').replace(/\s+/g,' ').trim(); }
function between(s,open,close){ var i=s.indexOf(open); if(i<0) return null; var j=s.indexOf(close,i+open.length); if(j<0) return null; return s.slice(i+open.length,j); }

/* ---- Embeddings (transformers.js, all-MiniLM-L6-v2, 384-dim) ---- */
var _ex=null, _exLoading=null;
async function getEx(){
  if(_ex) return _ex;
  if(!_exLoading){ _exLoading=(async function(){ const mod=await import('@xenova/transformers'); mod.env.allowLocalModels=false; _ex=await mod.pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2'); return _ex; })(); }
  return _exLoading;
}
async function embed(t){
  try{ const ex=await getEx(); const out=await ex(String(t||'').slice(0,512),{pooling:'mean',normalize:true}); return '['+Array.from(out.data).join(',')+']'; }
  catch(e){ return null; }
}

/* Teacher: Pollinations primary (keyless, generous) -> OpenRouter booster. */
async function teacher(prompt,temp){
  temp = (temp===undefined?0.6:temp);
  try{
    const r = await fetch('https://text.pollinations.ai/openai',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model:'openai', messages:[{role:'user',content:prompt}], temperature:temp }) });
    if(r.ok){ const j = await r.json().catch(function(){return null;}); const t = j&&j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content; if(t) return t; }
  }catch(e){}
  if(OPENROUTER_KEY){
    for(const model of FREE_MODELS){
      try{
        const r = await fetch('https://openrouter.ai/api/v1/chat/completions',{ method:'POST', headers:{'Authorization':'Bearer '+OPENROUTER_KEY,'Content-Type':'application/json','X-Title':'Damru Learn'}, body: JSON.stringify({ model:model, messages:[{role:'user',content:prompt}], temperature:temp, max_tokens:2000 }) });
        if(r.ok){ const j = await r.json(); const t = j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content; if(t) return t; }
      }catch(e){}
    }
  }
  return null;
}

function extractPairs(text){
  if(!text) return [];
  var t = text.replace(/```json/gi,'```').replace(/```/g,'');
  var s = t.indexOf('['), e = t.lastIndexOf(']');
  if(s>=0 && e>s){
    var body = t.slice(s,e+1).replace(/,(\s*[\]}])/g,'$1');
    try{ var arr = JSON.parse(body); if(Array.isArray(arr)) return arr; }catch(err){}
  }
  return [];
}

async function exists(question){
  try{
    const u = SUPABASE_URL+'/rest/v1/damru_knowledge?select=id&question=eq.'+encodeURIComponent(question)+'&limit=1';
    const r = await fetch(u,{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    if(!r.ok) return false; const j = await r.json(); return Array.isArray(j)&&j.length>0;
  }catch(e){ return false; }
}

async function saveQA(question,answer,intent){
  if(!question||!answer) return false;
  const emb = await embed(question+' '+answer);
  try{
    const body = { question:question, answer:answer, intent:intent||'general', lang:'en' };
    if(emb) body.embedding = emb;
    const r = await fetch(SUPABASE_URL+'/rest/v1/damru_knowledge',{ method:'POST', headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY, 'Content-Type':'application/json', Prefer:'return=minimal' }, body: JSON.stringify(body) });
    return r.ok;
  }catch(e){ return false; }
}

async function ingestPairs(pairs,intent){
  var saved = 0;
  for(const p of pairs){
    const q = ((p.question||p.q||'')+'').trim();
    const a = ((p.answer||p.a||'')+'').trim();
    if(!q||!a||q.length<8||a.length<20) continue;
    if(await exists(q)) continue;
    if(await saveQA(q,a,intent)) saved++;
  }
  return saved;
}

/* ---- Backfill embeddings for older rows that have none ---- */
async function backfillEmbeddings(limit){
  try{
    const u = SUPABASE_URL+'/rest/v1/damru_knowledge?select=id,question,answer&embedding=is.null&limit='+(limit||8);
    const r = await fetch(u,{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    if(!r.ok) return 0; const rows = await r.json(); if(!Array.isArray(rows)||!rows.length) return 0;
    var n = 0;
    for(const row of rows){
      const emb = await embed((row.question||'')+' '+(row.answer||''));
      if(!emb) continue;
      const up = await fetch(SUPABASE_URL+'/rest/v1/damru_knowledge?id=eq.'+row.id,{ method:'PATCH', headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY, 'Content-Type':'application/json', Prefer:'return=minimal' }, body: JSON.stringify({ embedding:emb }) });
      if(up.ok) n++;
    }
    return n;
  }catch(e){ return 0; }
}

/* ---- Learner 1: progressive book reading (cursor advances each cycle) ---- */
const BOOKS = [['https://www.gutenberg.org/cache/epub/2388/pg2388.txt','general'],['https://www.gutenberg.org/cache/epub/2680/pg2680.txt','general'],['https://www.gutenberg.org/cache/epub/5740/pg5740.txt','general'],['https://www.gutenberg.org/cache/epub/2009/pg2009.txt','general'],['https://www.gutenberg.org/cache/epub/1232/pg1232.txt','general']];
var bookState = null;
const CHUNK = 1600;
async function bookLearner(){
  try{
    if(!bookState || bookState.pos>=bookState.text.length){
      const idx = bookState ? (bookState.idx+1)%BOOKS.length : Math.floor(Math.random()*BOOKS.length);
      const url = BOOKS[idx][0];
      const r = await fetch(url,{ headers:{'User-Agent':'DamruBot/1.0'} });
      if(!r.ok) return 0;
      var text = await r.text();
      const a = text.indexOf('*** START'); const b = text.lastIndexOf('*** END');
      if(a>=0){ const nl = text.indexOf('\n',a); if(nl>=0) text = text.slice(nl+1); }
      if(b>=0) text = text.slice(0,b);
      bookState = { idx:idx, text:text, pos:0 };
    }
    const chunk = bookState.text.slice(bookState.pos, bookState.pos+CHUNK);
    bookState.pos += CHUNK;
    if(ws(chunk).length<200) return 0;
    const prompt = 'From the following book passage, create '+QA_PER+' clear standalone Q&A pairs that teach its key ideas. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer. No prose, no markdown.\n\nPassage:\n'+chunk;
    const out = await teacher(prompt,0.5);
    return await ingestPairs(extractPairs(out),'general');
  }catch(e){ return 0; }
}

/* ---- Learner 2: random Wikipedia (general knowledge / articles) ---- */
async function wikiLearner(){
  try{
    const r = await fetch('https://en.wikipedia.org/api/rest_v1/page/random/summary',{ headers:{'User-Agent':'DamruBot/1.0'} });
    if(!r.ok) return 0;
    const j = await r.json();
    const topic = j.title||''; const extract = j.extract||'';
    if(extract.length<80) return 0;
    const prompt = 'Topic: '+topic+'. Summary: '+extract+'\n\nCreate '+QA_PER+' factual Q&A pairs that teach this topic clearly. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
    const out = await teacher(prompt,0.5);
    return await ingestPairs(extractPairs(out),'general');
  }catch(e){ return 0; }
}

/* ---- Learner 3: arXiv science journals (maths / CS / physics) ---- */
const ARXIV_CATS = ['math.GM','math.NT','cs.AI','cs.LG','physics.gen-ph','stat.ML','math.PR','q-bio'];
async function journalLearner(){
  try{
    const cat = ARXIV_CATS[Math.floor(Math.random()*ARXIV_CATS.length)];
    const start = Math.floor(Math.random()*200);
    const u = 'http://export.arxiv.org/api/query?search_query=cat:'+cat+'&start='+start+'&max_results=2&sortBy=lastUpdatedDate&sortOrder=descending';
    const r = await fetch(u,{ headers:{'User-Agent':'DamruBot/1.0'} });
    if(!r.ok) return 0;
    const xml = await r.text();
    const parts = xml.split('<entry>').slice(1);
    var saved = 0;
    for(const ent of parts){
      const title = ws(between(ent,'<title>','</title>')||'');
      const abs = ws(between(ent,'<summary>','</summary>')||'');
      if(abs.length<120) continue;
      const prompt = 'Research paper title: '+title+'. Abstract: '+abs+'\n\nExplain the core idea simply and create '+QA_PER+' Q&A pairs teaching the science/maths concepts. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
      const out = await teacher(prompt,0.45);
      saved += await ingestPairs(extractPairs(out),'general');
    }
    return saved;
  }catch(e){ return 0; }
}

/* ---- Learner 4: advanced maths self-practice (generate + solve) ---- */
const MATH_TOPICS = ['calculus (integration by parts, limits, infinite series)','linear algebra (eigenvalues, diagonalization)','probability and statistics','number theory','differential equations','combinatorics','complex analysis','optimization'];
async function mathLearner(){
  try{
    const topic = MATH_TOPICS[Math.floor(Math.random()*MATH_TOPICS.length)];
    const prompt = 'Create '+QA_PER+' ADVANCED problems on '+topic+'. For each, give a full step-by-step worked solution that ends with the final answer. Put the problem in the question field and the full solution in the answer field. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
    const out = await teacher(prompt,0.3);
    return await ingestPairs(extractPairs(out),'math');
  }catch(e){ return 0; }
}

async function cycle(n){
  const t0 = Date.now();
  const results = await Promise.allSettled([ bookLearner(), wikiLearner(), journalLearner(), mathLearner() ]);
  const got = results.map(function(r){ return r.status==='fulfilled'?r.value:0; });
  const total = got.reduce(function(x,y){ return x+y; },0);
  const bf = await backfillEmbeddings(8);
  console.log('[cycle '+n+'] +'+total+' learned (book:'+got[0]+' wiki:'+got[1]+' arxiv:'+got[2]+' math:'+got[3]+') +'+bf+' embed-backfill in '+((Date.now()-t0)/1000).toFixed(1)+'s');
  return total;
}

async function main(){
  console.log('=== Damru Learning Engine START | cycle='+(CYCLE_MS/1000)+'s run='+RUN_MINUTES+'m ===');
  try{ await getEx(); console.log('embeddings model ready'); }catch(e){ console.log('embeddings model load failed (will retry per-call):',e&&e.message); }
  const deadline = Date.now()+RUN_MINUTES*60*1000;
  var n = 0, grand = 0;
  while(Date.now()<deadline){
    const t = Date.now();
    grand += await cycle(++n);
    const wait = Math.max(0, CYCLE_MS-(Date.now()-t));
    if(Date.now()+wait>=deadline) break;
    await sleep(wait);
  }
  console.log('=== Damru Learning Engine END | cycles='+n+' total_learned='+grand+' ===');
}
main();
