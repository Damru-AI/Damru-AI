#!/usr/bin/env python3
"""
Damru SYNAPTIC MEMORY LAYER  (brain-like associative recall)
============================================================
NOT a biological-neuron replica -- Damru's "thinking" still lives in the
transformer weights. This builds the FUNCTIONAL equivalent of synaptic bonds:
a concept graph where

  * NODES    = concepts (the row's intent + salient keywords)
  * EDGES    = associations between concepts that appear together
  * WEIGHTS  = Hebbian -> "concepts that fire together, wire together"
               (every co-occurrence strengthens the bond; rare bonds pruned)

At inference the model can do ASSOCIATIVE RECALL: map a query to its concept
nodes, walk the strongest synapses, and pull in related concepts as extra RAG
context -- so Damru "remembers" linked ideas instead of treating every question
in isolation.

This is a BATCH builder: it streams the knowledge dataset (bounded RAM),
accumulates Hebbian weights, prunes, and pushes a compact graph to HF. A second
mode (--recall "...") demonstrates query-time traversal.

Env:
  HF_TOKEN        (required)
  HF_REPO         source dataset   (default Damaru-ai/damru-knowledge)
  GRAPH_REPO      where to store    (default = HF_REPO)
  GRAPH_PATH      file in repo       (default synapses/graph.json)
  MAX_ROWS        rows to scan        (default 2000000)
  TOP_NODES       keep N strongest concepts        (default 40000)
  EDGES_PER_NODE  keep M strongest synapses / node (default 24)
  MIN_EDGE_W      drop synapses weaker than this   (default 2)
  KW_PER_ROW      keywords extracted per row       (default 6)
"""
import os
import re
import io
import json
import math
import time
import heapq
from collections import defaultdict, Counter

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
GRAPH_REPO = os.environ.get("GRAPH_REPO", HF_REPO)
GRAPH_PATH = os.environ.get("GRAPH_PATH", "synapses/graph.json")
MAX_ROWS = int(os.environ.get("MAX_ROWS") or "2000000")
TOP_NODES = int(os.environ.get("TOP_NODES") or "40000")
EDGES_PER_NODE = int(os.environ.get("EDGES_PER_NODE") or "24")
MIN_EDGE_W = int(os.environ.get("MIN_EDGE_W") or "2")
KW_PER_ROW = int(os.environ.get("KW_PER_ROW") or "6")

_STOP = set("""a an the of to in on at for and or but if is are was were be been being
this that these those it its as by with from into over under then than so such
what which who whom whose how why when where can could should would will shall
may might must do does did done have has had not no yes you your we our they them
he she his her him i me my mine ours yours their s t am pm via using use used
get got make made one two three new also more most very much many some any each
about above below between within without per into onto upon""".split())

_WORD = re.compile(r"[a-zA-Z][a-zA-Z+#.\-]{2,}")


def _keywords(text, k):
    """Cheap, dependency-free salient-term extraction."""
    toks = [w.lower().strip(".-") for w in _WORD.findall(text or "")]
    toks = [w for w in toks if len(w) >= 3 and w not in _STOP]
    if not toks:
        return []
    # frequency within the row, longer terms lightly favoured
    score = Counter()
    for w in toks:
        score[w] += 1 + min(len(w), 12) / 24.0
    return [w for w, _ in score.most_common(k)]


def _concepts(row):
    """A row's active concepts = its intent + top keywords from Q (and a few
    from A). These are the 'neurons' that fire for this memory."""
    out = []
    intent = (row.get("intent") or "").strip().lower()
    if intent:
        out.append("intent:" + intent)
    q = row.get("question") or ""
    a = row.get("answer") or ""
    out.extend(_keywords(q, KW_PER_ROW))
    out.extend(_keywords(a, max(2, KW_PER_ROW // 2)))
    # de-dup, preserve order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[: KW_PER_ROW + 3]


def _stream_rows():
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, split="train", streaming=True)
    n = 0
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        yield ex
        n += 1
        if n >= MAX_ROWS:
            break


def build():
    assert HF_TOKEN, "HF_TOKEN required"
    node_fire = Counter()                  # how often each concept fires
    edge_w = defaultdict(int)              # Hebbian synapse weights
    rows = 0
    t0 = time.time()
    for ex in _stream_rows():
        cs = _concepts(ex)
        for c in cs:
            node_fire[c] += 1
        # co-fire -> strengthen every pair (undirected, sorted key)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                edge_w[key] += 1
        rows += 1
        if rows % 100000 == 0:
            print("  scanned %d rows | nodes=%d edges=%d | %.0fs"
                  % (rows, len(node_fire), len(edge_w), time.time() - t0),
                  flush=True)
            # bound memory: periodically prune the weakest transient edges
            if len(edge_w) > 8_000_000:
                edge_w = defaultdict(int, {k: v for k, v in edge_w.items()
                                           if v >= MIN_EDGE_W})

    # keep only the strongest concept nodes
    keep = set(c for c, _ in node_fire.most_common(TOP_NODES))
    print("keeping %d / %d concept nodes" % (len(keep), len(node_fire)),
          flush=True)

    # build adjacency, prune by weight, cap fan-out per node
    adj = defaultdict(list)
    for (a, b), w in edge_w.items():
        if w < MIN_EDGE_W or a not in keep or b not in keep:
            continue
        adj[a].append((b, w))
        adj[b].append((a, w))

    nodes = {}
    edges = {}
    for c in keep:
        nbrs = heapq.nlargest(EDGES_PER_NODE, adj.get(c, []),
                              key=lambda x: x[1])
        if not nbrs and node_fire[c] < 2:
            continue
        nodes[c] = node_fire[c]
        edges[c] = [[b, w] for b, w in nbrs]

    graph = {
        "meta": {
            "rows_scanned": rows,
            "nodes": len(nodes),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hebbian": True,
            "edges_per_node": EDGES_PER_NODE,
            "min_edge_w": MIN_EDGE_W,
        },
        "nodes": nodes,
        "edges": edges,
    }
    _push(graph)
    print("SYNAPTIC GRAPH built: %d nodes, ~%d synapses, from %d rows."
          % (len(nodes), sum(len(v) for v in edges.values()) // 2, rows),
          flush=True)
    return graph


def _push(graph):
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    raw = json.dumps(graph, ensure_ascii=False).encode("utf-8")
    for attempt in range(8):
        try:
            api.upload_file(path_or_fileobj=io.BytesIO(raw),
                            path_in_repo=GRAPH_PATH, repo_id=GRAPH_REPO,
                            repo_type="dataset")
            print("  pushed %s (%.2f MB)" % (GRAPH_PATH, len(raw) / 1e6),
                  flush=True)
            return
        except Exception as e:
            s = str(e)
            if attempt == 7:
                raise
            wait = 1900 if ("per hour" in s or "repository commits" in s) \
                else min(120, 8 * (2 ** attempt))
            print("  push retry in %ds (%s)" % (wait, s[:70]), flush=True)
            time.sleep(wait)


def _load_graph():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(GRAPH_REPO, GRAPH_PATH, repo_type="dataset",
                        token=HF_TOKEN)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def recall(query, hops=2, top=12):
    """Associative recall: map query -> concept nodes, spread activation across
    the strongest synapses for `hops`, return the most-activated related
    concepts. Use these as extra retrieval keys / RAG context at inference."""
    g = _load_graph()
    edges = g.get("edges", {})
    seeds = [c for c in _keywords(query, 8) if c in edges]
    intent = "intent:" + (query or "").strip().lower()
    if intent in edges:
        seeds.append(intent)
    activation = defaultdict(float)
    frontier = {s: 1.0 for s in seeds}
    for _ in range(max(1, hops)):
        nxt = defaultdict(float)
        for node, energy in frontier.items():
            for b, w in edges.get(node, []):
                # normalise by log-weight so hubs don't dominate
                gain = energy * (math.log1p(w) / 6.0)
                activation[b] += gain
                nxt[b] += gain * 0.5            # decay across hops
        frontier = nxt
    for s in seeds:
        activation.pop(s, None)
    ranked = heapq.nlargest(top, activation.items(), key=lambda x: x[1])
    return [c for c, _ in ranked]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--recall":
        print("associated concepts ->", recall(" ".join(sys.argv[2:])))
    else:
        build()
