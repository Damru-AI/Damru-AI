#!/usr/bin/env python3
"""One-time Colab BGE-M3 backfill for the Supabase HOT vector cache.

It intentionally caps at 10k rows. Millions remain in HF/FAISS; putting all
10M x 1024 vectors in Supabase free tier would require tens of GB.
"""
from __future__ import annotations
import hashlib, os, json
from datetime import datetime, timezone
import numpy as np, requests

HF_TOKEN=os.getenv("HF_TOKEN","")
SB_URL=os.getenv("SUPABASE_URL","").rstrip("/")
SB_KEY=os.getenv("SUPABASE_SERVICE_KEY","")
HF_REPO=os.getenv("HF_REPO","Damaru-ai/damru-knowledge")
MODEL=os.getenv("EMBED_MODEL","BAAI/bge-m3")
MAX_ROWS=max(100,min(int(os.getenv("HOT_BACKFILL_ROWS","10000")),20000))
BATCH=max(1,min(int(os.getenv("BACKFILL_BATCH","8")),32))
H={"apikey":SB_KEY,"Authorization":"Bearer "+SB_KEY,"Content-Type":"application/json"}

def neg_id(q):
    return -(int(hashlib.sha256(q.encode()).hexdigest()[:15],16)+1)
def halfvec(v):
    return "["+",".join("%.7g"%float(x) for x in v)+"]"
def main():
    if not(HF_TOKEN and SB_URL and SB_KEY): raise RuntimeError("HF_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY required")
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds=load_dataset(HF_REPO,data_files="data/*.parquet",split="train",streaming=True,token=HF_TOKEN)
    seen=set(); rows=[]
    for x in ds:
        q=(x.get("question") or "").strip(); a=(x.get("answer") or "").strip()
        key=" ".join(q.lower().split())
        if len(q)<8 or len(a)<20 or key in seen: continue
        seen.add(key); rows.append(x)
        if len(rows)>=MAX_ROWS: break
    print("selected",len(rows),"unique HF rows")
    model=SentenceTransformer(MODEL)
    done=0
    for pos in range(0,len(rows),BATCH):
        part=rows[pos:pos+BATCH]
        texts=[(x.get("question","")+"\n"+x.get("answer","")[:1200]).strip() for x in part]
        vecs=model.encode(texts,batch_size=BATCH,normalize_embeddings=True,convert_to_numpy=True).astype("float32")
        if vecs.shape[1]!=1024: raise RuntimeError("Expected BGE-M3 1024 dims, got %s"%(vecs.shape,))
        payload=[]
        for x,v in zip(part,vecs):
            q=(x.get("question") or "").strip(); payload.append({
              "knowledge_id":neg_id(q),"question":q[:2000],"answer":(x.get("answer") or "")[:6000],
              "intent":(x.get("intent") or "general")[:80],"lang":x.get("lang") or "en",
              "source":"hf-backfill","created_at":x.get("created_at"),"embedding":halfvec(v)})
        r=requests.post(SB_URL+"/rest/v1/damru_hot_vectors",headers={**H,"Prefer":"return=minimal,resolution=merge-duplicates"},params={"on_conflict":"knowledge_id"},json=payload,timeout=120)
        r.raise_for_status(); done+=len(part); print("upserted",done,"/",len(rows),flush=True)
    print(json.dumps({"ok":True,"model":MODEL,"backfilled":done,"at":datetime.now(timezone.utc).isoformat()},indent=2))
if __name__=="__main__": main()
