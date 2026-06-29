#!/usr/bin/env python3
"""
Damru BLOOM dedup filter -- the "middle filter" that NEVER lets a duplicate /
copied Q&A through, at 20M+ scale, in a tiny file.

WHY A BLOOM FILTER (not a Python set / sqlite)?
  A set of 40M sha1 hashes ~= 3 GB RAM. A bloom filter for 60M items at 1%%
  false-positive rate is only ~72 MB and answers "have I seen this?" in
  microseconds. We persist it on HuggingFace, load it at the start of every
  run, and re-save it at the end -> dedup state survives across all runs and
  across both ingestion tracks.

Pure standard-library (hashlib + gzip). No external deps, so it is trivial to
unit-test locally and cannot break the harvester install.

Key API:
  bf = BloomFilter(capacity=60_000_000, error_rate=0.01)
  if bf.add(normalize(question)):   # True  -> brand new, KEEP
      ...                            # False -> already seen, SKIP
  raw = bf.to_bytes(); BloomFilter.from_bytes(raw)
"""
import io
import gzip
import json
import math
import hashlib


def normalize(text):
    """Canonical key for a question: lowercase + whitespace-collapsed.
    Matches the spirit of store._hash so the two tracks agree on duplicates."""
    return " ".join((text or "").lower().split())


class BloomFilter:
    def __init__(self, capacity=60_000_000, error_rate=0.01,
                 num_bits=None, num_hashes=None):
        if num_bits and num_hashes:
            self.m = int(num_bits)
            self.k = int(num_hashes)
        else:
            cap = max(1, int(capacity))
            m = -cap * math.log(error_rate) / (math.log(2) ** 2)
            self.m = max(8, ((int(m) + 7) // 8) * 8)        # whole bytes
            self.k = max(1, int(round((self.m / cap) * math.log(2))))
        self.bits = bytearray(self.m // 8)
        self.n = 0                                          # approx items added

    def _indexes(self, key):
        data = key.encode("utf-8") if isinstance(key, str) else key
        h = hashlib.sha256(data).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:16], "big") | 1             # odd -> good stride
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, key):
        """Set bits for key. Return True if it was (probably) NOT present before
        (i.e. brand new -> keep), False if already seen (-> skip)."""
        was_new = False
        for idx in self._indexes(key):
            byte, mask = idx >> 3, 1 << (idx & 7)
            if not (self.bits[byte] & mask):
                self.bits[byte] |= mask
                was_new = True
        if was_new:
            self.n += 1
        return was_new

    def __contains__(self, key):
        return all(self.bits[idx >> 3] & (1 << (idx & 7))
                   for idx in self._indexes(key))

    def to_bytes(self):
        header = json.dumps({"m": self.m, "k": self.k, "n": self.n}).encode("utf-8")
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6) as g:
            g.write(len(header).to_bytes(4, "big"))
            g.write(header)
            g.write(self.bits)
        return out.getvalue()

    @classmethod
    def from_bytes(cls, raw):
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as g:
            hlen = int.from_bytes(g.read(4), "big")
            header = json.loads(g.read(hlen).decode("utf-8"))
            bits = bytearray(g.read())
        bf = cls(num_bits=header["m"], num_hashes=header["k"])
        bf.bits = bits
        bf.n = int(header.get("n", 0))
        return bf


if __name__ == "__main__":
    # tiny self-test
    bf = BloomFilter(capacity=100000, error_rate=0.01)
    assert bf.add(normalize("What is Newton's second law?")) is True
    assert bf.add(normalize("what   is NEWTON'S second law?")) is False  # same key
    raw = bf.to_bytes()
    bf2 = BloomFilter.from_bytes(raw)
    assert normalize("What is Newton's second law?") in bf2
    print("bloom self-test OK | size=%.1f KB | m=%d k=%d" % (len(raw) / 1024, bf2.m, bf2.k))
