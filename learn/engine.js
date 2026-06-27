/* ============================================================
   DAMRU AI — Continuous Learning Engine v3 ("DEEP BEAST")
   ------------------------------------------------------------
   Upgrades over v2:
   * DEPTH LADDER (0->100): every subject x topic is taught across
     6 mastery levels (Foundation -> PhD/Research), driven by a
     persistent cursor (Supabase damru_state). So Damru no longer
     learns a topic from 1-2 questions — it climbs basic->PhD.
   * SELF-THINKING (curiosityLearner): Damru looks at what it just
     learned and AUTO-GENERATES its own deeper questions, then
     answers them. Questions "are born in its mind".
   * CODE MASTERY: codeLearner writes programs in many languages,
     ACTUALLY EXECUTES them (Piston API) to verify output, and also
     practises DEBUGGING (find+fix bugs). Covers web & app dev too.
   * OPEN LAB (openLabLearner): free, unrestricted frontier
     exploration with chained questions.
   * Feeding (curriculum/arxiv/news/book/wiki) keeps running in
     parallel, and retagGeneral keeps fixing old imbalance.
   ============================================================ */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_KEY;
const GEMINI_KEY = process.env.GEMINI_KEY||'';
const GOOGLE_CSE_KEY = process.env.GOOGLE_CSE_KEY||'';
const GOOGLE_CSE_CX = process.env.GOOGLE_CSE_CX||'';
const YOUTUBE_KEY = process.env.YOUTUBE_KEY||'';
if(!SUPABASE_URL||!SUPABASE_KEY){ console.error('Missing SUPABASE_URL / SUPABASE_KEY'); process.exit(1); }

const CYCLE_MS = parseInt(process.env.CYCLE_MS||'60000',10);
const RUN_MINUTES = parseInt(process.env.RUN_MINUTES||'330',10);
const QA_PER = parseInt(process.env.QA_PER||'5',10);
const LADDER_PER_CYCLE = parseInt(process.env.LADDER_PER_CYCLE||'2',10);
const PISTON_URL = process.env.PISTON_URL||'https://emkc.org/api/v2/piston';

const FREE_MODELS = ['deepseek/deepseek-chat-v3-0324:free','meta-llama/llama-3.3-70b-instruct:free','google/gemini-2.0-flash-exp:free','qwen/qwen-2.5-72b-instruct:free'];

const sleep = function(ms){ return new Promise(function(r){ setTimeout(r,ms); }); };
function ws(x){ return (x||'').replace(/\s+/g,' ').trim(); }
function between(s,open,close){ var i=s.indexOf(open); if(i<0) return null; var j=s.indexOf(close,i+open.length); if(j<0) return null; return s.slice(i+open.length,j); }
function pick(a){ return a[Math.floor(Math.random()*a.length)]; }

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

/* Teacher: Pollinations primary (keyless) -> OpenRouter booster. */
async function teacher(prompt,temp){
  temp = (temp===undefined?0.6:temp);
  if(GEMINI_KEY){
    try{
      const gurl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key='+GEMINI_KEY;
      const gopt = { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ contents:[{ parts:[{ text:prompt }] }], generationConfig:{ temperature:temp, maxOutputTokens:2048 } }) };
      const gr = await fetch(gurl, gopt);
      if(gr.ok){ const gj = await gr.json().catch(function(){return null;}); const gt = gj&&gj.candidates&&gj.candidates[0]&&gj.candidates[0].content&&gj.candidates[0].content.parts&&gj.candidates[0].content.parts[0]&&gj.candidates[0].content.parts[0].text; if(gt) return gt; }
    }catch(e){}
  }
  try{
    const r = await fetch('https://text.pollinations.ai/openai', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model:'openai', messages:[{role:'user',content:prompt}], temperature:temp }) });
    if(r.ok){ const j = await r.json().catch(function(){return null;}); const t = j&&j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content; if(t) return t; }
  }catch(e){}
  if(OPENROUTER_KEY){
    for(const model of FREE_MODELS){
      try{
        const r = await fetch('https://openrouter.ai/api/v1/chat/completions', { method:'POST', headers:{'Authorization':'Bearer '+OPENROUTER_KEY,'Content-Type':'application/json','X-Title':'Damru Learn'}, body: JSON.stringify({ model:model, messages:[{role:'user',content:prompt}], temperature:temp, max_tokens:2000 }) });
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
function extractObj(text){
  if(!text) return null;
  var t = text.replace(/```json/gi,'```').replace(/```/g,'');
  var s=t.indexOf('{'), e=t.lastIndexOf('}');
  if(s>=0&&e>s){ try{ return JSON.parse(t.slice(s,e+1)); }catch(err){} }
  return null;
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
async function ingestOne(q,a,intent){
  q=(q||'').trim(); a=(a||'').trim();
  if(q.length<8||a.length<20) return 0;
  if(await exists(q)) return 0;
  return (await saveQA(q,a,intent))?1:0;
}

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

/* ---- persistent STATE (depth-ladder cursor) in Supabase damru_state ---- */
async function loadState(){
  try{
    const r=await fetch(SUPABASE_URL+'/rest/v1/damru_state?id=eq.1&select=data',{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    if(r.ok){ const j=await r.json(); if(Array.isArray(j)&&j[0]&&j[0].data) return j[0].data; }
  }catch(e){}
  return null;
}
async function saveState(data){
  try{
    const r=await fetch(SUPABASE_URL+'/rest/v1/damru_state?id=eq.1',{ method:'PATCH', headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY, 'Content-Type':'application/json', Prefer:'return=minimal' }, body: JSON.stringify({ data:data, updated_at:new Date().toISOString() }) });
    if(r.ok) return true;
  }catch(e){}
  return false;
}

/* ============================================================
   CURRICULUM — subjects, each with its OWN intent tag.
   ============================================================ */
const CURRICULUM = [
  { intent:'coding',          label:'Programming & Software Engineering', topics:['clean code & design patterns','async & concurrency','memory management','testing & debugging','API design','version control & CI/CD'] },
  { intent:'webdev',          label:'Web Development', topics:['HTML/CSS layout','JavaScript & DOM','React & state','REST & GraphQL APIs','performance & caching','PWAs & security'] },
  { intent:'appdev',          label:'Mobile & App Development', topics:['Android with Kotlin','iOS with Swift','Flutter & Dart','React Native','app architecture (MVVM/Clean)','app store deployment'] },
  { intent:'devops',          label:'DevOps & Cloud', topics:['Docker & containers','Kubernetes','CI/CD pipelines','AWS/GCP cloud','infrastructure as code','monitoring & logging'] },
  { intent:'gamedev',         label:'Game Development', topics:['game loops','physics engines','graphics & shaders','Unity/Godot basics','game AI','optimization'] },
  { intent:'dsa',             label:'Data Structures & Algorithms', topics:['arrays & hashing','trees & graphs','dynamic programming','sorting & searching','greedy & backtracking','complexity analysis'] },
  { intent:'systems',         label:'Operating Systems & Computer Architecture', topics:['processes & threads','scheduling','virtual memory','file systems','CPU pipelines & caches','concurrency primitives'] },
  { intent:'databases',       label:'Databases & SQL', topics:['relational design & normalization','indexing','transactions & ACID','query optimization','NoSQL models','vector databases'] },
  { intent:'networking',      label:'Computer Networks & Internet', topics:['TCP/IP & OSI','routing & switching','DNS & HTTP','TLS & security','congestion control','CDNs'] },
  { intent:'cybersecurity',   label:'Cybersecurity & Cryptography', topics:['threat modeling','encryption & hashing','authentication','OWASP vulnerabilities','network defense','incident response'] },
  { intent:'ai',              label:'Artificial Intelligence & Machine Learning', topics:['neural networks','transformers & LLMs','reinforcement learning','training & optimization','embeddings & RAG','model evaluation'] },
  { intent:'datascience',     label:'Data Science & Statistics', topics:['probability distributions','hypothesis testing','regression','feature engineering','data visualization','experiment design'] },
  { intent:'quantumcomputing',label:'Quantum Computing', topics:['qubits & superposition','quantum gates','entanglement','Shor & Grover algorithms','quantum error correction','NISQ devices'] },
  { intent:'physics',         label:'Physics (Classical & Modern)', topics:['mechanics & dynamics','electromagnetism','thermodynamics','relativity','optics & waves','nuclear physics'] },
  { intent:'quantum',         label:'Quantum Physics & Quantum Fluctuation', topics:['wavefunction & uncertainty','quantum field theory','vacuum & quantum fluctuations','tunneling','Casimir effect','decoherence'] },
  { intent:'chemistry',       label:'Chemistry', topics:['atomic structure & bonding','thermochemistry','organic reactions','electrochemistry','chemical kinetics','periodic trends'] },
  { intent:'biology',         label:'Biology', topics:['cell biology','evolution','ecology','human anatomy','microbiology','plant biology'] },
  { intent:'genetics',        label:'Genetics & Biotechnology', topics:['DNA & RNA','gene expression','CRISPR editing','heredity','genomics','synthetic biology'] },
  { intent:'neuroscience',    label:'Neuroscience & the Brain', topics:['neurons & synapses','brain regions','memory & learning','neuroplasticity','consciousness','brain-computer interfaces'] },
  { intent:'medicine',        label:'Medicine & Human Physiology', topics:['cardiovascular system','immune system','pharmacology','diagnostics','disease mechanisms','public health'] },
  { intent:'lifescience',     label:'Future Life Sciences & Longevity', topics:['aging biology','regenerative medicine','gene therapy','synthetic organs','space biology','biohacking science'] },
  { intent:'astronomy',       label:'Astronomy & Astrophysics', topics:['stars & stellar evolution','galaxies','black holes','exoplanets','telescopes','planetary science'] },
  { intent:'cosmology',       label:'Cosmology & the Universe', topics:['Big Bang','dark matter & dark energy','cosmic inflation','CMB radiation','fate of the universe','multiverse'] },
  { intent:'space',           label:'Space Technology & Rocketry', topics:['rocket propulsion','orbital mechanics','satellites','reusable launch','spacecraft design','ISRO & global missions'] },
  { intent:'spacerobotics',   label:'Space Robotics & Interplanetary Missions', topics:['Mars rover operations','autonomous space navigation','interplanetary trajectory planning','sample collection robotics','swarm space robots','in-situ resource utilization'], style:'scenario' },
  { intent:'lifesupport',     label:'Human Life Support Systems', topics:['closed-loop air recycling','water reclamation','radiation shielding','food production in space','thermal regulation','redundancy design'], style:'scenario' },
  { intent:'robotics',        label:'Robotics & Control', topics:['kinematics & dynamics','actuators & sensors','SLAM & navigation','manipulation & grasping','human-robot interaction','fault tolerance'], style:'scenario' },
  { intent:'controltheory',   label:'Control Theory & Dynamical Systems', topics:['PID control','state-space models','stability analysis','feedback & feedforward','optimal control','adaptive control'], style:'scenario' },
  { intent:'automation',      label:'Automation & Mechatronics', topics:['PLC & industrial automation','sensor fusion','motor control','pneumatics & hydraulics','robotic assembly','predictive maintenance'] },
  { intent:'electronics',     label:'Electronics & Embedded Systems', topics:['circuit analysis','microcontrollers','signal processing','power electronics','PCB design','real-time systems'] },
  { intent:'materials',       label:'Materials Science & Nanotechnology', topics:['crystal structures','semiconductors','composites','nanomaterials','superconductors','smart materials'] },
  { intent:'energy',          label:'Energy & Nuclear Technology', topics:['nuclear fission & fusion','solar & wind','batteries & storage','grid systems','hydrogen energy','reactor safety'] },
  { intent:'military',        label:'Military Technology & Defense Systems', topics:['radar & stealth','missile & guidance','drones & UAVs','cyber warfare','electronic warfare','space-based defense'] },
  { intent:'militarymgmt',    label:'Military Management & Strategy', topics:['command structure','logistics & supply chains','strategic planning','intelligence cycle','crisis decision-making','force coordination'], style:'scenario' },
  { intent:'geopolitics',     label:'Geopolitics & International Relations', topics:['balance of power','alliances & treaties','resource conflicts','diplomacy','global institutions','security doctrines'] },
  { intent:'economics',       label:'Economics', topics:['supply & demand','macro vs micro','inflation & monetary policy','fiscal policy','international trade','market structures'] },
  { intent:'finance',         label:'Finance & Investing', topics:['time value of money','risk & return','valuation','portfolio theory','financial statements','derivatives'] },
  { intent:'business',        label:'Business & Management', topics:['strategy frameworks','marketing','operations','organizational behavior','leadership','negotiation'] },
  { intent:'taskmgmt',        label:'Task & Project Management', topics:['planning & scheduling','prioritization','resource allocation','agile & scrum','risk tracking','dependency management'], style:'scenario' },
  { intent:'opsresearch',     label:'Operations Research & Optimization', topics:['linear programming','queuing theory','scheduling optimization','graph algorithms','simulation','decision trees'], style:'scenario' },
  { intent:'criticalthinking',label:'Critical Thinking & Logic', topics:['cognitive biases','logical fallacies','argument analysis','Bayesian reasoning','first-principles thinking','evidence evaluation'], style:'scenario' },
  { intent:'problemsolving',  label:'Real-World Problem Solving', topics:['root-cause analysis','trade-off analysis','constraint handling','rapid prototyping','decision under uncertainty','post-mortems'], style:'scenario' },
  { intent:'security',        label:'Threat Detection, Analysis & Response', topics:['anomaly detection','threat classification','sensor data fusion','escalation & response','false-positive reduction','autonomous defense'], style:'scenario' },
  { intent:'riskanalysis',    label:'Risk Analysis & Decision Making', topics:['risk matrices','failure mode analysis','expected value reasoning','scenario planning','black-swan resilience','mitigation strategy'], style:'scenario' },
  { intent:'systemsthinking', label:'Systems Thinking & Complexity', topics:['feedback loops','emergence','network effects','resilience','leverage points','chaos & nonlinearity'] },
  { intent:'futurescience',   label:'Future Science & Emerging Tech', topics:['fusion energy','brain-computer interfaces','AGI pathways','space colonization','molecular nanotech','climate engineering'] },
  { intent:'biotech',         label:'Biotechnology & Synthetic Biology', topics:['protein engineering','bioreactors','mRNA technology','lab-grown organs','biosensors','directed evolution'] },
  { intent:'climate',         label:'Climate & Earth Sciences', topics:['carbon cycle','climate modeling','renewable transitions','ocean systems','atmospheric science','sustainability tech'] },
  { intent:'geology',         label:'Geology & Planetary Science', topics:['plate tectonics','rock & mineral cycles','planetary formation','volcanism','remote sensing','terraforming science'] },
  { intent:'mathematics',     label:'Mathematics (Pure & Applied)', topics:['calculus','linear algebra','probability','number theory','differential equations','discrete math'] },
  { intent:'logic',           label:'Formal Logic & Reasoning', topics:['propositional logic','predicate logic','proof techniques','set theory','boolean algebra','computability'] },
  { intent:'philosophy',      label:'Philosophy & Ethics', topics:['ethics & morality','epistemology','philosophy of mind','Stoicism & resilience','political philosophy','philosophy of science'] },
  { intent:'psychology',      label:'Psychology & Human Behaviour', topics:['cognition & perception','motivation','social psychology','decision biases','emotional regulation','behavioral change'] },
  { intent:'history',         label:'World & Indian History', topics:['ancient civilizations','Indian freedom struggle','world wars','medieval India','industrial revolution','post-independence India'] },
  { intent:'geography',       label:'Geography', topics:['physical geography','climate zones','Indian geography','economic geography','maps & GIS','natural resources'] },
  { intent:'polity',          label:'Polity & Governance', topics:['Indian Constitution','fundamental rights','parliament & judiciary','federalism','elections','governance schemes'] },
  { intent:'english',         label:'English Language & Grammar', topics:['tenses','parts of speech','common errors','vocabulary & idioms','comprehension','sentence structure'] },
  { intent:'communication',   label:'Communication & Writing', topics:['clear writing','persuasion','structuring arguments','technical writing','storytelling','presentations'] },
  { intent:'entrepreneurship',label:'Startups & Innovation', topics:['idea validation','business models','product-market fit','fundraising','growth','lean methodology'] },
  { intent:'ethicalai',       label:'AI Safety & Ethics', topics:['alignment','bias & fairness','interpretability','robustness','governance','responsible deployment'] },
  { intent:'exam',            label:'Indian & Rajasthan Competitive Exams', topics:['Rajasthan GK','current affairs','reasoning & aptitude','CET/RAS/REET pattern','quantitative aptitude','general science'] }
];

/* depth bands: 0 -> 100, basic -> PhD */
const LEVELS = [
  { tag:'L1', band:'Foundation (0-15)',     desc:'absolute basics: definitions, intuition and real-life analogies for a complete beginner. No jargon without explaining it.' },
  { tag:'L2', band:'Basic (15-35)',         desc:'core concepts with simple worked examples and the standard terminology.' },
  { tag:'L3', band:'Intermediate (35-55)',  desc:'underlying mechanisms, problem-solving, and how concepts connect to each other.' },
  { tag:'L4', band:'Advanced (55-75)',      desc:'deep theory, edge cases, derivations and advanced techniques used by professionals.' },
  { tag:'L5', band:'Expert (75-90)',        desc:'specialist depth: optimization, trade-offs, pitfalls and current best practices.' },
  { tag:'L6', band:'PhD / Research (90-100)',desc:'frontier: open problems, cutting-edge research, rigorous proofs and novel synthesis.' }
];

const ALLOWED = {}; CURRICULUM.forEach(function(s){ ALLOWED[s.intent]=1; }); ALLOWED['math']=1; ALLOWED['general']=1; ALLOWED['currentaffairs']=1; ALLOWED['curiosity']=1;
const INTENT_LIST = Object.keys(ALLOWED).filter(function(k){ return k!=='general'; });

/* ---- in-memory ladder cursor (loaded from / saved to damru_state) ---- */
var ladder = { s:0, t:0, l:0 };
function advanceLadder(){
  ladder.l++;
  if(ladder.l>=LEVELS.length){ ladder.l=0; ladder.t++; }
  const sub=CURRICULUM[ladder.s%CURRICULUM.length];
  if(ladder.t>=sub.topics.length){ ladder.t=0; ladder.s=(ladder.s+1)%CURRICULUM.length; }
}

/* ---- DEPTH-LADDER learner: teaches one (subject,topic,level) cell 0->100 ---- */
async function ladderLearner(){
  try{
    const sub = CURRICULUM[ladder.s%CURRICULUM.length];
    const topic = sub.topics[ladder.t%sub.topics.length];
    const lvl = LEVELS[ladder.l%LEVELS.length];
    var prompt;
    if(sub.style==='scenario'){
      prompt = 'You are training DAMRU, a future AI brain for robotics & real-world operations. Subject: '+sub.label+' | Topic: '+topic+' | Mastery level: '+lvl.band+' — '+lvl.desc+'\nCreate '+QA_PER+' SCENARIO-based training items AT THIS EXACT LEVEL. Each: "question"=a concrete situation/task; "answer"=Situation, Analysis, Step-by-step solution, Key risks. Match the difficulty to the level.\nReturn ONLY a valid JSON array of objects with string fields question and answer. No prose, no markdown.';
    } else {
      prompt = 'You are an expert teacher building DAMRU\'s mastery of a subject from 0 to 100 (beginner to PhD). Subject: '+sub.label+' | Topic: '+topic+' | Mastery level: '+lvl.band+' — '+lvl.desc+'\nCreate '+QA_PER+' high-quality standalone Q&A pairs that teach THIS TOPIC AT THIS EXACT LEVEL (do not drift easier or harder). Accurate, self-contained, progressively building understanding.\nReturn ONLY a valid JSON array of objects with string fields question and answer. No prose, no markdown.';
    }
    const out = await teacher(prompt, sub.style==='scenario'?0.5:0.45);
    const n = await ingestPairs(extractPairs(out), sub.intent);
    advanceLadder();
    return n;
  }catch(e){ advanceLadder(); return 0; }
}

/* ---- SELF-THINKING: Damru generates its own deeper questions & answers them ---- */
async function curiosityLearner(){
  try{
    const offset = Math.floor(Math.random()*900);
    const u = SUPABASE_URL+'/rest/v1/damru_knowledge?select=question,answer,intent&limit=4&offset='+offset;
    const r = await fetch(u,{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    var seed='';
    if(r.ok){ const rows=await r.json(); if(Array.isArray(rows)) seed=rows.map(function(x){ return '- '+(x.question||''); }).join('\n'); }
    const prompt = 'You are DAMRU, a curious self-improving AI. Here are some things you already know:\n'+(seed||'(general knowledge)')+'\n\nLet new questions be BORN IN YOUR MIND: think about what you still do not fully understand, what connects these ideas, and what a brilliant mind would ask NEXT to go deeper. Generate '+QA_PER+' original, insightful follow-up questions AND answer each one accurately and thoroughly.\nReturn ONLY a valid JSON array of objects with string fields question and answer. No prose, no markdown.';
    const out = await teacher(prompt,0.75);
    return await ingestPairs(extractPairs(out),'curiosity');
  }catch(e){ return 0; }
}

/* ---- OPEN LAB: free, unrestricted frontier exploration ---- */
const FRONTIERS = ['interplanetary colonization & self-sufficient habitats','autonomous robotic swarms for construction','artificial general intelligence & reasoning','fusion-powered spacecraft','closed-loop human life support on Mars','brain-computer symbiosis','self-healing materials','planetary-scale climate engineering','quantum sensing & navigation','autonomous threat detection & defense','molecular manufacturing','longevity & cellular regeneration','swarm intelligence for exploration','resilient mission planning under uncertainty','space-based energy & resource mining'];
async function openLabLearner(){
  try{
    const f = pick(FRONTIERS);
    const prompt = 'OPEN LAB — you are DAMRU, an unrestricted curious mind exploring the frontier. Topic: '+f+'.\nThink freely and deeply. Generate '+QA_PER+' forward-looking Q&A pairs: pose ambitious yet technically grounded questions ("how could we...", "what is the optimal way to...", "what are the failure modes of...") and answer with original cross-disciplinary reasoning. Bold but accurate.\nReturn ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.85);
    return await ingestPairs(extractPairs(out),'futurescience');
  }catch(e){ return 0; }
}

/* ---- CODE MASTERY: write + EXECUTE (Piston) + DEBUG, many languages ---- */
const CODE_LANGS = [ {l:'python',name:'Python'},{l:'javascript',name:'JavaScript'},{l:'typescript',name:'TypeScript'},{l:'c',name:'C'},{l:'c++',name:'C++'},{l:'java',name:'Java'},{l:'go',name:'Go'},{l:'rust',name:'Rust'},{l:'ruby',name:'Ruby'},{l:'php',name:'PHP'},{l:'bash',name:'Bash'},{l:'kotlin',name:'Kotlin'} ];
const CODE_TASKS = ['reverse a string','check if a number is prime','compute factorial of n','find the maximum in a list','print FizzBuzz up to 20','sort an array (any algorithm)','count vowels in a string','print Fibonacci up to n terms','check if a string is a palindrome','sum the digits of a number','binary search in a sorted array','compute GCD of two numbers'];
var _runtimes=null;
async function getRuntimes(){
  if(_runtimes) return _runtimes;
  try{ const r=await fetch(PISTON_URL+'/runtimes'); if(r.ok){ _runtimes=await r.json(); } }catch(e){}
  return _runtimes||[];
}
async function pistonVersion(lang){
  const rt=await getRuntimes(); if(!Array.isArray(rt)) return null;
  const hit=rt.find(function(x){ return x.language===lang || (x.aliases&&x.aliases.indexOf(lang)>=0); });
  return hit?hit.version:null;
}
async function runCode(lang,code){
  try{
    const version=await pistonVersion(lang); if(!version) return null;
    const r=await fetch(PISTON_URL+'/execute',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ language:lang, version:version, files:[{ content:code }] }) });
    if(!r.ok) return null; const j=await r.json();
    const run=j.run||{}; const comp=j.compile||{};
    return { out:(run.stdout||'').trim(), err:((comp.stderr||'')+(run.stderr||'')).trim(), code:run.code };
  }catch(e){ return null; }
}
async function codeLearner(){
  try{
    const lang=pick(CODE_LANGS);
    if(Math.random()<0.6){
      /* EXECUTION mode: generate -> actually run -> learn verified code */
      const task=pick(CODE_TASKS);
      const gen=await teacher('Write a COMPLETE, runnable '+lang.name+' program to: '+task+'. Print the result to standard output. For Java the public class MUST be named Main. Return ONLY JSON: {"code":"...","explanation":"..."} with the full source in code.',0.3);
      const obj=extractObj(gen); if(!obj||!obj.code) return 0;
      const res=await runCode(lang.l,obj.code);
      if(!res) return 0;
      var ok = (res.code===0 || (res.out&&res.out.length>0)) && !(res.err&&!res.out);
      const q='How do you '+task+' in '+lang.name+'? Show working, executed code.';
      var a=(obj.explanation||'')+'\n\n```'+lang.l+'\n'+obj.code+'\n```\n\n';
      if(ok){ a+='Verified output when executed:\n```\n'+(res.out||'(no stdout)')+'\n```'; }
      else { a+='Note: running this revealed an error to learn from:\n```\n'+(res.err||'runtime error')+'\n```\nAlways test and handle such cases.'; }
      return await ingestOne(q,a,'coding');
    } else {
      /* DEBUGGING mode: find + fix a bug */
      const gen=await teacher('Create an instructive DEBUGGING exercise in '+lang.name+'. Provide a short snippet with ONE realistic bug. Return ONLY JSON: {"buggy_code":"...","bug":"one-line description","fixed_code":"...","why":"short explanation"}.',0.5);
      const o=extractObj(gen); if(!o||!o.buggy_code||!o.fixed_code) return 0;
      const q='Find and fix the bug in this '+lang.name+' code:\n```'+lang.l+'\n'+o.buggy_code+'\n```';
      const a='Bug: '+(o.bug||'')+'\n\nWhy it happens: '+(o.why||'')+'\n\nFixed code:\n```'+lang.l+'\n'+o.fixed_code+'\n```';
      return await ingestOne(q,a,'coding');
    }
  }catch(e){ return 0; }
}

/* ---- arXiv journals mapped to REAL subjects ---- */
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
      const prompt = 'Research paper title: '+title+'. Abstract: '+abs+'\n\nExplain the core idea simply and create '+QA_PER+' Q&A pairs teaching the key concepts. Return ONLY a valid JSON array of objects with string fields question and answer.';
      const out = await teacher(prompt,0.45);
      saved += await ingestPairs(extractPairs(out),intent);
    }
    return saved;
  }catch(e){ return 0; }
}

/* ---- maths self-practice ---- */
const MATH_TOPICS = ['calculus (integration by parts, limits, infinite series)','linear algebra (eigenvalues, diagonalization)','probability and statistics','number theory','differential equations','combinatorics','complex analysis','optimization'];
async function mathLearner(){
  try{
    const topic = pick(MATH_TOPICS);
    const prompt = 'Create '+QA_PER+' ADVANCED problems on '+topic+'. For each, give a full step-by-step worked solution ending with the final answer. Problem in the question field, full solution in the answer field. Return ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.3);
    return await ingestPairs(extractPairs(out),'mathematics');
  }catch(e){ return 0; }
}

/* ---- current affairs / news (real web) ---- */
const FEEDS = ['https://feeds.bbci.co.uk/news/science_and_environment/rss.xml','https://feeds.bbci.co.uk/news/technology/rss.xml','https://feeds.bbci.co.uk/news/world/asia/india/rss.xml'];
async function newsLearner(){
  try{
    const r = await fetch(pick(FEEDS),{ headers:{'User-Agent':'DamruBot/1.0'} });
    if(!r.ok) return 0;
    const xml = await r.text();
    const items = xml.split('<item>').slice(1,5);
    var blob = items.map(function(it){ const t=ws(between(it,'<title>','</title>')||'').replace(/<!\[CDATA\[|\]\]>/g,''); const d=ws(between(it,'<description>','</description>')||'').replace(/<!\[CDATA\[|\]\]>/g,'').replace(/<[^>]+>/g,''); return t+'. '+d; }).filter(function(x){return x.length>30;}).join('\n');
    if(blob.length<80) return 0;
    const prompt = 'Recent news headlines & summaries:\n'+blob+'\n\nCreate '+QA_PER+' current-affairs Q&A pairs useful for competitive-exam prep (factual, neutral). Return ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.4);
    return await ingestPairs(extractPairs(out),'currentaffairs');
  }catch(e){ return 0; }
}

/* ---- public-domain book reading (throttled) ---- */
const BOOKS = ['https://www.gutenberg.org/cache/epub/2388/pg2388.txt','https://www.gutenberg.org/cache/epub/2680/pg2680.txt','https://www.gutenberg.org/cache/epub/5740/pg5740.txt','https://www.gutenberg.org/cache/epub/2009/pg2009.txt','https://www.gutenberg.org/cache/epub/1232/pg1232.txt'];
var bookState = null; const CHUNK = 1600;
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
    const prompt = 'From the following book passage, create '+QA_PER+' clear standalone Q&A pairs that teach its key ideas. Return ONLY a valid JSON array of objects with string fields question and answer.\n\nPassage:\n'+chunk;
    const out = await teacher(prompt,0.5);
    return await ingestPairs(extractPairs(out),'philosophy');
  }catch(e){ return 0; }
}

/* ---- random Wikipedia (throttled) ---- */
async function wikiLearner(){
  try{
    const r = await fetch('https://en.wikipedia.org/api/rest_v1/page/random/summary', { headers:{'User-Agent':'DamruBot/1.0'} });
    if(!r.ok) return 0;
    const j = await r.json();
    const topic = j.title||''; const extract = j.extract||'';
    if(extract.length<80) return 0;
    const prompt = 'Topic: '+topic+'. Summary: '+extract+'\n\nCreate '+QA_PER+' factual Q&A pairs that teach this topic clearly. Return ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.5);
    return await ingestPairs(extractPairs(out),'general');
  }catch(e){ return 0; }
}

/* ---- RE-TAGGER: re-classify old 'general' rows into real subjects ---- */
async function retagGeneral(limit){
  try{
    const offset = Math.floor(Math.random()*900);
    const u = SUPABASE_URL+'/rest/v1/damru_knowledge?select=id,question,answer&intent=eq.general&limit='+(limit||3)+'&offset='+offset;
    const r = await fetch(u,{ headers:{ apikey:SUPABASE_KEY, Authorization:'Bearer '+SUPABASE_KEY } });
    if(!r.ok) return 0; const rows = await r.json(); if(!Array.isArray(rows)||!rows.length) return 0;
    var n=0;
    for(const row of rows){
      const prompt = 'Classify the following Q&A into EXACTLY ONE subject from this list:\n'+INTENT_LIST.join(', ')+'\nIf none clearly fits, answer "general". Return ONLY the single subject word.\n\nQ: '+(row.question||'').slice(0,300)+'\nA: '+(row.answer||'').slice(0,400);
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

/* ---- REAL WEB SEARCH learner (Google Custom Search) ---- */
async function webSearchLearner(){
  if(!GOOGLE_CSE_KEY||!GOOGLE_CSE_CX) return 0;
  try{
    const sub = pick(CURRICULUM); const topic = pick(sub.topics);
    const q = topic+' '+sub.label+' explained in depth';
    const surl = 'https://www.googleapis.com/customsearch/v1?key='+GOOGLE_CSE_KEY+'&cx='+GOOGLE_CSE_CX+'&num=6&q='+encodeURIComponent(q);
    const r = await fetch(surl, { headers:{ Accept:'application/json' } });
    if(!r.ok) return 0;
    const j = await r.json();
    const items = (j&&j.items)||[];
    if(!items.length) return 0;
    var blob = items.map(function(it){ return '- '+(it.title||'')+': '+(it.snippet||''); }).join('\n');
    const prompt = 'You are DAMRU learning '+sub.label+' ('+topic+') from REAL web search results.\nResults:\n'+blob+'\n\nUsing these as grounding, create '+QA_PER+' accurate, self-contained Q&A pairs that teach this topic from foundations toward advanced understanding. Add correct established facts where helpful; do not invent sources.\nReturn ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.45);
    return await ingestPairs(extractPairs(out), sub.intent);
  }catch(e){ return 0; }
}

/* ---- YOUTUBE learner (YouTube Data API v3) ---- */
async function youtubeLearner(){
  if(!YOUTUBE_KEY) return 0;
  try{
    const sub = pick(CURRICULUM); const topic = pick(sub.topics);
    const q = topic+' '+sub.label+' tutorial lecture';
    const yurl = 'https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=6&relevanceLanguage=en&key='+YOUTUBE_KEY+'&q='+encodeURIComponent(q);
    const r = await fetch(yurl, { headers:{ Accept:'application/json' } });
    if(!r.ok) return 0;
    const j = await r.json();
    const items = (j&&j.items)||[];
    if(!items.length) return 0;
    var blob = items.map(function(it){ var s=it.snippet||{}; return '- '+(s.title||'')+': '+(s.description||''); }).join('\n');
    const prompt = 'You are DAMRU learning '+sub.label+' ('+topic+') by surveying educational video titles and descriptions.\nVideos:\n'+blob+'\n\nInfer the key concepts these cover and create '+QA_PER+' accurate standalone Q&A pairs teaching this topic clearly. Use correct established knowledge; do not fabricate claims about specific videos.\nReturn ONLY a valid JSON array of objects with string fields question and answer.';
    const out = await teacher(prompt,0.5);
    return await ingestPairs(extractPairs(out), sub.intent);
  }catch(e){ return 0; }
}

async function cycle(n){
  const t0 = Date.now();
  const tasks = [];
  const cells=[];
  for(var i=0;i<LADDER_PER_CYCLE;i++){ const sub=CURRICULUM[ladder.s%CURRICULUM.length]; cells.push(sub.intent+':L'+(ladder.l+1)); tasks.push(ladderLearner()); }
  tasks.push(curiosityLearner());
  tasks.push(openLabLearner());
  tasks.push(mathLearner());
  if(n%2===0) tasks.push(codeLearner());
  if(n%2===1) tasks.push(journalLearner());
  if(n%3===0) tasks.push(wikiLearner());
  if(n%4===0) tasks.push(bookLearner());
  if(n%2===1) tasks.push(webSearchLearner());
  if(n%3===1) tasks.push(youtubeLearner());
  if(n%5===0) tasks.push(newsLearner());
  tasks.push(retagGeneral(3));
  const results = await Promise.allSettled(tasks);
  const retag = results[results.length-1];
  const learned = results.slice(0,results.length-1).reduce(function(x,r){ return x+(r.status==='fulfilled'?r.value:0); },0);
  const retagged = retag.status==='fulfilled'?retag.value:0;
  await saveState(ladder);
  const bf = await backfillEmbeddings(8);
  console.log('[cycle '+n+'] +'+learned+' learned (ladder:'+cells.join(',')+' +think+lab+code) retag:'+retagged+' bf:'+bf+' '+((Date.now()-t0)/1000).toFixed(1)+'s');
  return learned;
}

async function main(){
  console.log('=== Damru Engine v3 (DEEP BEAST) START | cycle='+(CYCLE_MS/1000)+'s run='+RUN_MINUTES+'m subjects='+CURRICULUM.length+' levels='+LEVELS.length+' ===');
  const st = await loadState();
  if(st && typeof st.s==='number'){ ladder=st; console.log('resumed ladder @ subject '+ladder.s+'/'+CURRICULUM.length+' topic '+ladder.t+' level L'+(ladder.l+1)); }
  else { ladder={ s:Math.floor(Math.random()*CURRICULUM.length), t:0, l:0 }; console.log('no saved state (create damru_state table to persist) — starting fresh ladder'); }
  try{ await getEx(); console.log('embeddings model ready'); }catch(e){ console.log('embeddings load failed (retry per-call):',e&&e.message); }
  const deadline = Date.now()+RUN_MINUTES*60*1000;
  var n = 0, grand = 0;
  while(Date.now()<deadline){
    const t = Date.now();
    grand += await cycle(++n);
    const wait = Math.max(0, CYCLE_MS-(Date.now()-t));
    if(Date.now()+wait>=deadline) break;
    await sleep(wait);
  }
  console.log('=== Damru Engine END | cycles='+n+' total_learned='+grand+' ===');
}
main();
