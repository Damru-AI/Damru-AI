/* ============================================================
   DAMRU AI — Continuous Learning Engine v2 ("BEAST MODE")
   A long-running worker. Every CYCLE_MS it runs one learn-cycle.
   Goal: teach Damru 55+ subjects in a BALANCED way (not flooded
   with 'general'), plus scenario-based reasoning for its future
   role as a robotics / real-world / space-operations brain.

   Each cycle launches several learners IN PARALLEL:
     * curriculumLearner  — rotates through a 55+ subject CURRICULUM,
                            each tagged with its OWN intent (coding,
                            physics, robotics, economics, quantum...).
     * openLabLearner     — Damru's "free mind": autonomously poses
                            frontier questions and answers them.
     * journalLearner     — arXiv abstracts mapped to real subjects.
     * retagGeneral       — re-classifies old 'general' rows into
                            proper subjects to FIX past imbalance.
     * book/wiki/news     — throttled web reading (real internet).
   A rotating cursor guarantees every subject is covered evenly,
   so the knowledge base stays BALANCED across all domains.
   Embeddings (all-MiniLM-L6-v2, 384-dim) are computed for every
   new row + backfilled for old rows => semantic retrieval.
   ============================================================ */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;
if(!SUPABASE_URL||!SUPABASE_KEY){ console.error('Missing SUPABASE_URL / SUPABASE_KEY'); process.exit(1); }

const CYCLE_MS = parseInt(process.env.CYCLE_MS||'60000',10);
const RUN_MINUTES = parseInt(process.env.RUN_MINUTES||'330',10);
const QA_PER = parseInt(process.env.QA_PER||'5',10);
const SUBJECTS_PER_CYCLE = parseInt(process.env.SUBJECTS_PER_CYCLE||'3',10);

const FREE_MODELS = ['deepseek/deepseek-chat-v3-0324:free','meta-llama/llama-3.3-70b-instruct:free','google/gemini-2.0-flash-exp:free','qwen/qwen-2.5-72b-instruct:free'];

const sleep = function(ms){ return new Promise(function(r){ setTimeout(r,ms); }); };
function ws(x){ return (x||'').replace(/\s+/g,' ').trim(); }
function between(s,open,close){ var i=s.indexOf(open); if(i<0) return null; var j=s.indexOf(close,i+open.length); if(j<0) return null; return s.slice(i+open.length,j); }
function pick(a){ return a[Math.floor(Math.random()*a.length)]; }
function shuffle(a){ a=a.slice(); for(var i=a.length-1;i>0;i--){ var j=Math.floor(Math.random()*(i+1)); var t=a[i]; a[i]=a[j]; a[j]=t; } return a; }

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

/* ============================================================
   THE CURRICULUM — 55+ subjects, each with its OWN intent tag.
   style 'qa'       => factual teaching Q&A.
   style 'scenario' => real-world situation -> analysis -> step
                       solution -> risks (Damru's reasoning brain).
   ============================================================ */
const CURRICULUM = [
  { intent:'coding',          label:'Programming & Software Engineering', topics:['clean code & design patterns','async & concurrency','memory management','testing & debugging','API design','version control & CI/CD'] },
  { intent:'webdev',          label:'Web & App Development', topics:['HTML/CSS layout','JavaScript & DOM','React & state','REST & GraphQL APIs','performance & caching','PWAs & mobile'] },
  { intent:'dsa',             label:'Data Structures & Algorithms', topics:['arrays & hashing','trees & graphs','dynamic programming','sorting & searching','greedy & backtracking','complexity analysis'] },
  { intent:'systems',         label:'Operating Systems & Computer Architecture', topics:['processes & threads','scheduling','virtual memory','file systems','CPU pipelines & caches','concurrency primitives'] },
  { intent:'databases',       label:'Databases & SQL', topics:['relational design & normalization','indexing','transactions & ACID','query optimization','NoSQL models','vector databases'] },
  { intent:'networking',      label:'Computer Networks & Internet', topics:['TCP/IP & OSI','routing & switching','DNS & HTTP','TLS & security','congestion control','CDNs'] },
  { intent:'cybersecurity',   label:'Cybersecurity & Cryptography', topics:['threat modeling','encryption & hashing','authentication','common vulnerabilities (OWASP)','network defense','incident response'] },
  { intent:'ai',              label:'Artificial Intelligence & Machine Learning', topics:['neural networks','transformers & LLMs','reinforcement learning','training & optimization','embeddings & RAG','model evaluation'] },
  { intent:'datascience',     label:'Data Science & Statistics', topics:['probability distributions','hypothesis testing','regression','feature engineering','data visualization','experiment design'] },
  { intent:'quantumcomputing',label:'Quantum Computing', topics:['qubits & superposition','quantum gates','entanglement','Shor & Grover algorithms','quantum error correction','NISQ devices'] },
  { intent:'physics',         label:'Physics (Classical & Modern)', topics:['mechanics & dynamics','electromagnetism','thermodynamics','relativity','optics & waves','nuclear physics'] },
  { intent:'quantum',         label:'Quantum Physics & Quantum Fluctuation', topics:['wavefunction & uncertainty','quantum field theory','vacuum & quantum fluctuations','tunneling','Casimir effect','decoherence'] },
  { intent:'chemistry',       label:'Chemistry', topics:['atomic structure & bonding','thermochemistry','organic reactions','electrochemistry','chemical kinetics','periodic trends'] },
  { intent:'biology',         label:'Biology', topics:['cell biology','evolution & natural selection','ecology','human anatomy','microbiology','plant biology'] },
  { intent:'genetics',        label:'Genetics & Biotechnology', topics:['DNA & RNA','gene expression','CRISPR editing','heredity','genomics','synthetic biology'] },
  { intent:'neuroscience',    label:'Neuroscience & the Brain', topics:['neurons & synapses','brain regions','memory & learning','neuroplasticity','consciousness','brain-computer interfaces'] },
  { intent:'medicine',        label:'Medicine & Human Physiology', topics:['cardiovascular system','immune system','pharmacology basics','diagnostics','disease mechanisms','public health'] },
  { intent:'lifescience',     label:'Future Life Sciences & Longevity', topics:['aging biology','regenerative medicine','gene therapy','synthetic organs','biohacking science','space biology'] },
  { intent:'astronomy',       label:'Astronomy & Astrophysics', topics:['stars & stellar evolution','galaxies','black holes','exoplanets','telescopes & observation','planetary science'] },
  { intent:'cosmology',       label:'Cosmology & the Universe', topics:['Big Bang','dark matter & dark energy','cosmic inflation','CMB radiation','fate of the universe','multiverse theories'] },
  { intent:'space',           label:'Space Technology & Rocketry', topics:['rocket propulsion','orbital mechanics','satellites','reusable launch systems','spacecraft design','ISRO & global missions'] },
  { intent:'spacerobotics',   label:'Space Robotics & Interplanetary Missions', topics:['Mars rover operations','autonomous navigation in space','interplanetary trajectory planning','sample collection robotics','swarm space robots','in-situ resource utilization'], style:'scenario' },
  { intent:'lifesupport',     label:'Human Life Support Systems', topics:['closed-loop air recycling','water reclamation','radiation shielding','food production in space','thermal regulation','emergency redundancy design'], style:'scenario' },
  { intent:'robotics',        label:'Robotics & Control', topics:['kinematics & dynamics','actuators & sensors','SLAM & navigation','manipulation & grasping','human-robot interaction','fault tolerance'], style:'scenario' },
  { intent:'controltheory',   label:'Control Theory & Dynamical Systems', topics:['PID control','state-space models','stability analysis','feedback & feedforward','optimal control','adaptive control'], style:'scenario' },
  { intent:'automation',      label:'Automation & Mechatronics', topics:['PLC & industrial automation','sensor fusion','motor control','pneumatics & hydraulics','robotic assembly lines','predictive maintenance'] },
  { intent:'electronics',     label:'Electronics & Embedded Systems', topics:['circuit analysis','microcontrollers','signal processing','power electronics','PCB design','real-time systems'] },
  { intent:'materials',       label:'Materials Science & Nanotechnology', topics:['crystal structures','semiconductors','composites','nanomaterials','superconductors','smart materials'] },
  { intent:'energy',          label:'Energy & Nuclear Technology', topics:['nuclear fission & fusion','solar & wind','batteries & storage','grid systems','hydrogen energy','reactor safety'] },
  { intent:'military',        label:'Military Technology & Defense Systems', topics:['radar & stealth','missile & guidance systems','drones & UAVs','cyber warfare','electronic warfare','space-based defense'] },
  { intent:'militarymgmt',    label:'Military Management & Strategy', topics:['command structure','logistics & supply chains','strategic planning','intelligence cycle','crisis decision-making','force coordination'], style:'scenario' },
  { intent:'geopolitics',     label:'Geopolitics & International Relations', topics:['balance of power','alliances & treaties','resource conflicts','diplomacy','global institutions','security doctrines'] },
  { intent:'economics',       label:'Economics', topics:['supply & demand','macro vs micro','inflation & monetary policy','fiscal policy','international trade','market structures'] },
  { intent:'finance',         label:'Finance & Investing', topics:['time value of money','risk & return','valuation','portfolio theory','financial statements','derivatives'] },
  { intent:'business',        label:'Business & Management', topics:['strategy frameworks','marketing','operations','organizational behavior','leadership','negotiation'] },
  { intent:'taskmgmt',        label:'Task & Project Management', topics:['planning & scheduling','prioritization frameworks','resource allocation','agile & scrum','risk tracking','dependency management'], style:'scenario' },
  { intent:'opsresearch',     label:'Operations Research & Optimization', topics:['linear programming','queuing theory','scheduling optimization','graph algorithms for ops','simulation','decision trees'], style:'scenario' },
  { intent:'criticalthinking',label:'Critical Thinking & Logic', topics:['cognitive biases','logical fallacies','argument analysis','Bayesian reasoning','first-principles thinking','evidence evaluation'], style:'scenario' },
  { intent:'problemsolving',  label:'Real-World Problem Solving', topics:['root-cause analysis','trade-off analysis','constraint handling','rapid prototyping mindset','decision under uncertainty','post-mortems'], style:'scenario' },
  { intent:'security',        label:'Threat Detection, Analysis & Response', topics:['anomaly detection','threat classification','sensor data fusion for threats','escalation & response planning','false-positive reduction','autonomous defense response'], style:'scenario' },
  { intent:'riskanalysis',    label:'Risk Analysis & Decision Making', topics:['risk matrices','failure mode analysis (FMEA)','expected value reasoning','scenario planning','black-swan resilience','mitigation strategy'], style:'scenario' },
  { intent:'systemsthinking', label:'Systems Thinking & Complexity', topics:['feedback loops','emergence','network effects','resilience & robustness','leverage points','chaos & nonlinearity'] },
  { intent:'futurescience',   label:'Future Science & Emerging Tech', topics:['fusion energy','brain-computer interfaces','AGI pathways','space colonization','molecular nanotech','climate engineering'] },
  { intent:'biotech',         label:'Biotechnology & Synthetic Biology', topics:['protein engineering','bioreactors','mRNA technology','lab-grown organs','biosensors','directed evolution'] },
  { intent:'climate',         label:'Climate & Earth Sciences', topics:['carbon cycle','climate modeling','renewable transitions','ocean systems','atmospheric science','sustainability tech'] },
  { intent:'geology',         label:'Geology & Planetary Science', topics:['plate tectonics','rock & mineral cycles','planetary formation','volcanism','remote sensing','terraforming science'] },
  { intent:'mathematics',     label:'Mathematics (Pure & Applied)', topics:['calculus','linear algebra','probability','number theory','differential equations','discrete math'] },
  { intent:'logic',           label:'Formal Logic & Reasoning', topics:['propositional logic','predicate logic','proof techniques','set theory','boolean algebra','computability'] },
  { intent:'philosophy',      label:'Philosophy & Ethics', topics:['ethics & morality','epistemology','philosophy of mind','Stoicism & resilience','political philosophy','philosophy of science'] },
  { intent:'psychology',      label:'Psychology & Human Behaviour', topics:['cognition & perception','motivation','social psychology','decision-making biases','emotional regulation','behavioral change'] },
  { intent:'history',         label:'World & Indian History', topics:['ancient civilizations','Indian freedom struggle','world wars','medieval India','industrial revolution','post-independence India'] },
  { intent:'geography',       label:'Geography', topics:['physical geography','climate zones','Indian geography','economic geography','maps & GIS','natural resources'] },
  { intent:'polity',          label:'Polity & Governance', topics:['Indian Constitution','fundamental rights','parliament & judiciary','federalism','elections','governance schemes'] },
  { intent:'english',         label:'English Language & Grammar', topics:['tenses','parts of speech','common errors','vocabulary & idioms','comprehension','sentence structure'] },
  { intent:'communication',   label:'Communication & Writing', topics:['clear writing','persuasion','structuring arguments','technical writing','storytelling','presentation skills'] },
  { intent:'entrepreneurship',label:'Startups & Innovation', topics:['idea validation','business models','product-market fit','fundraising','growth strategies','lean methodology'] },
  { intent:'ethicalai',       label:'AI Safety & Ethics', topics:['alignment','bias & fairness','interpretability','robustness','governance & policy','responsible deployment'] },
  { intent:'exam',            label:'Indian & Rajasthan Competitive Exams', topics:['Rajasthan GK','current affairs','reasoning & aptitude','CET/RAS/REET pattern','quantitative aptitude','general science'] }
];

/* allowed intent set (for the re-tagger) */
const ALLOWED = {}; CURRICULUM.forEach(function(s){ ALLOWED[s.intent]=1; }); ALLOWED['math']=1; ALLOWED['general']=1; ALLOWED['currentaffairs']=1;
const INTENT_LIST = Object.keys(ALLOWED).filter(function(k){ return k!=='general'; });

/* rotating cursor => even, BALANCED coverage of every subject */
var subjCursor = Math.floor(Math.random()*CURRICULUM.length);
function nextSubjects(k){
  var out=[]; for(var i=0;i<k;i++){ out.push(CURRICULUM[subjCursor % CURRICULUM.length]); subjCursor=(subjCursor+1)%CURRICULUM.length; } return out;
}

async function curriculumLearner(sub){
  try{
    const topic = pick(sub.topics);
    var prompt;
    if(sub.style==='scenario'){
      prompt = 'You are training DAMRU, a future AI brain for robotics, critical thinking and real-world / space operations. Subject: '+sub.label+'. Focus area: '+topic+'.\nCreate '+QA_PER+' realistic SCENARIO-based training items. For each item: "question" = a concrete real-world situation or task that requires analysis; "answer" = structured reasoning in this shape -> Situation, Analysis, Step-by-step solution, Key risks/considerations. Be technically accurate and practical.\nReturn ONLY a valid JSON array where each item is an object with two string fields: question and answer. No prose, no markdown.';
    } else {
      prompt = 'You are an expert teacher building DAMRU\'s knowledge. Subject: '+sub.label+'. Focus area: '+topic+'.\nCreate '+QA_PER+' high-quality standalone Q&A pairs that teach the most important, accurate concepts of this focus area (mix of fundamentals and a couple of deeper insights). Keep questions self-contained and answers clear and correct.\nReturn ONLY a valid JSON array where each item is an object with two string fields: question and answer. No prose, no markdown.';
    }
    const out = await teacher(prompt, sub.style==='scenario'?0.5:0.45);
    return await ingestPairs(extractPairs(out), sub.intent);
  }catch(e){ return 0; }
}

/* ---- Open Lab: Damru's FREE MIND — autonomous frontier exploration ---- */
const FRONTIERS = ['interplanetary colonization & self-sufficient habitats','autonomous robotic swarms for construction','artificial general intelligence & reasoning','fusion-powered spacecraft','closed-loop human life support on Mars','brain-computer symbiosis','self-healing materials & structures','planetary-scale climate engineering','quantum sensing & navigation','autonomous threat detection & defense','molecular manufacturing','longevity & cellular regeneration','swarm intelligence for exploration','resilient mission planning under uncertainty','space-based energy & resource mining'];
async function openLabLearner(){
  try{
    const f = pick(FRONTIERS);
    const prompt = 'OPEN LAB — you are DAMRU, an unrestricted curious AI mind exploring the frontier of science & technology. Topic of exploration: '+f+'.\nThink freely and deeply. Generate '+QA_PER+' insightful, forward-looking Q&A pairs: pose ambitious but technically grounded questions ("how could we...", "what is the optimal way to...", "what are the failure modes of...") and answer them with original, reasoned analysis combining multiple disciplines. Be bold but accurate.\nReturn ONLY a valid JSON array where each item is an object with two string fields: question and answer. No prose, no markdown.';
    const out = await teacher(prompt,0.8);
    return await ingestPairs(extractPairs(out),'futurescience');
  }catch(e){ return 0; }
}

/* ---- arXiv science journals mapped to REAL subjects ---- */
const ARXIV_CATS = [['math.NT','mathematics'],['math.PR','mathematics'],['cs.AI','ai'],['cs.LG','ai'],['cs.RO','robotics'],['eess.SY','controltheory'],['physics.gen-ph','physics'],['quant-ph','quantum'],['astro-ph.GA','astronomy'],['astro-ph.CO','cosmology'],['cond-mat.mtrl-sci','materials'],['q-bio.NC','neuroscience'],['q-bio.GN','genetics'],['econ.GN','economics'],['stat.ML','datascience'],['nlin.AO','systemsthinking']];
async function journalLearner(){
  try{
    const c = pick(ARXIV_CATS); const cat=c[0], intent=c[1];
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
      const prompt = 'Research paper title: '+title+'. Abstract: '+abs+'\n\nExplain the core idea simply and create '+QA_PER+' Q&A pairs teaching the key concepts. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
      const out = await teacher(prompt,0.45);
      saved += await ingestPairs(extractPairs(out),intent);
    }
    return saved;
  }catch(e){ return 0; }
}

/* ---- advanced maths self-practice (generate + solve) ---- */
const MATH_TOPICS = ['calculus (integration by parts, limits, infinite series)','linear algebra (eigenvalues, diagonalization)','probability and statistics','number theory','differential equations','combinatorics','complex analysis','optimization'];
async function mathLearner(){
  try{
    const topic = pick(MATH_TOPICS);
    const prompt = 'Create '+QA_PER+' ADVANCED problems on '+topic+'. For each, give a full step-by-step worked solution that ends with the final answer. Put the problem in the question field and the full solution in the answer field. Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
    const out = await teacher(prompt,0.3);
    return await ingestPairs(extractPairs(out),'mathematics');
  }catch(e){ return 0; }
}

/* ---- current affairs / news (real web, exam-relevant) ---- */
const FEEDS = ['https://feeds.bbci.co.uk/news/science_and_environment/rss.xml','https://feeds.bbci.co.uk/news/technology/rss.xml','https://feeds.bbci.co.uk/news/world/asia/india/rss.xml'];
async function newsLearner(){
  try{
    const r = await fetch(pick(FEEDS),{ headers:{'User-Agent':'DamruBot/1.0'} });
    if(!r.ok) return 0;
    const xml = await r.text();
    const items = xml.split('<item>').slice(1,5);
    var blob = items.map(function(it){ const t=ws(between(it,'<title>','</title>')||'').replace(/<!\[CDATA\[|\]\]>/g,''); const d=ws(between(it,'<description>','</description>')||'').replace(/<!\[CDATA\[|\]\]>/g,'').replace(/<[^>]+>/g,''); return t+'. '+d; }).filter(function(x){return x.length>30;}).join('\n');
    if(blob.length<80) return 0;
    const prompt = 'Here are recent news headlines & summaries:\n'+blob+'\n\nCreate '+QA_PER+' current-affairs Q&A pairs useful for competitive-exam preparation (factual, neutral). Return ONLY a valid JSON array where each item is an object with two string fields: question and answer.';
    const out = await teacher(prompt,0.4);
    return await ingestPairs(extractPairs(out),'currentaffairs');
  }catch(e){ return 0; }
}

/* ---- progressive public-domain book reading (throttled) ---- */
const BOOKS = ['https://www.gutenberg.org/cache/epub/2388/pg2388.txt','https://www.gutenberg.org/cache/epub/2680/pg2680.txt','https://www.gutenberg.org/cache/epub/5740/pg5740.txt','https://www.gutenberg.org/cache/epub/2009/pg2009.txt','https://www.gutenberg.org/cache/epub/1232/pg1232.txt'];
var bookState = null;
const CHUNK = 1600;
async function bookLearner(){
  try{
    if(!bookState || bookState.pos>=bookState.text.length){
      const idx = bookState ? (bookState.idx+1)%BOOKS.length : Math.floor(Math.random()*BOOKS.length);
      const r = await fetch(BOOKS[idx],{ headers:{'User-Agent':'DamruBot/1.0'} });
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
    return await ingestPairs(extractPairs(out),'philosophy');
  }catch(e){ return 0; }
}

/* ---- random Wikipedia (throttled general knowledge) ---- */
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

/* ---- RE-TAGGER: fix past imbalance by re-classifying old 'general' rows ---- */
async function retagGeneral(limit){
  try{
    const offset = Math.floor(Math.random()*900);
    const u = SUPABASE_URL+'/rest/v1/damru_knowledge?select=id,question,answer&intent=eq.general&limit='+(limit||3)+'&offset='+offset;
    const r = await fetch(u,{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    if(!r.ok) return 0; const rows = await r.json(); if(!Array.isArray(rows)||!rows.length) return 0;
    var n=0;
    for(const row of rows){
      const prompt = 'Classify the following Q&A into EXACTLY ONE subject from this list:\n'+INTENT_LIST.join(', ')+'\nIf none clearly fits, answer "general".\nReturn ONLY the single subject word, nothing else.\n\nQ: '+(row.question||'').slice(0,300)+'\nA: '+(row.answer||'').slice(0,400);
      const out = await teacher(prompt,0.0);
      if(!out) continue;
      const tag = ws(out).toLowerCase().replace(/[^a-z]/g,'');
      if(!tag || !ALLOWED[tag] || tag==='general') continue;
      const up = await fetch(SUPABASE_URL+'/rest/v1/damru_knowledge?id=eq.'+row.id,{ method:'PATCH', headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY, 'Content-Type':'application/json', Prefer:'return=minimal' }, body: JSON.stringify({ intent:tag }) });
      if(up.ok) n++;
    }
    return n;
  }catch(e){ return 0; }
}

async function cycle(n){
  const t0 = Date.now();
  const subs = nextSubjects(SUBJECTS_PER_CYCLE);
  const tasks = [];
  subs.forEach(function(s){ tasks.push(curriculumLearner(s)); });
  tasks.push(openLabLearner());
  tasks.push(mathLearner());
  if(n%2===0) tasks.push(journalLearner());
  if(n%3===0) tasks.push(wikiLearner());
  if(n%4===0) tasks.push(bookLearner());
  if(n%5===0) tasks.push(newsLearner());
  tasks.push(retagGeneral(3));
  const results = await Promise.allSettled(tasks);
  const learnedTasks = results.slice(0, results.length-1);
  const retag = results[results.length-1];
  const total = learnedTasks.reduce(function(x,r){ return x + (r.status==='fulfilled'?r.value:0); }, 0);
  const retagged = retag.status==='fulfilled'?retag.value:0;
  const bf = await backfillEmbeddings(8);
  console.log('[cycle '+n+'] +'+total+' learned ('+subs.map(function(s){return s.intent;}).join(',')+' +openlab+math) | retagged:'+retagged+' embed-bf:'+bf+' | '+((Date.now()-t0)/1000).toFixed(1)+'s');
  return total;
}

async function main(){
  console.log('=== Damru Learning Engine v2 (BEAST) START | cycle='+(CYCLE_MS/1000)+'s run='+RUN_MINUTES+'m subjects='+CURRICULUM.length+' ===');
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
