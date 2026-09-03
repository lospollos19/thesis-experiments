#!/usr/bin/env python3
"""
Join the SIL and HIL campaign results and report the SIL/HIL gap per operator (RQ1).

Usage: analyze_campaign.py results_sil.json results_hil.json

Verdict per mutant per env: KILLED_OUTPUT / KILLED_ERROR / SURVIVED / DIM_ERR.
"killed" = any KILLED_*/DIM_ERR (fault observed: divergence, crash, or shape change).

Per operator we report:
  killed_sil, killed_hil,
  GAP   = survived in SIL but killed in HIL  <- the RQ1 blind spot
  sil+  = killed in SIL but survived in HIL  (e.g. OOB segfault only on CPU)
  equiv = survived in both (equivalent-mutant candidates, to vet by hand)
"""
import json, sys
from collections import defaultdict

def killed(v): return v.startswith("KILLED") or v == "DIM_ERR"

def load(p):
    """Accept both a JSON array (old runs) and JSONL (one verdict per line)."""
    txt = open(p).read().strip()
    if txt.startswith("["):
        rows = json.loads(txt)
    else:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
    return {m["id"]: m for m in rows}

sil, hil = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(sil) & set(hil))

rows = defaultdict(lambda: dict(n=0, ks=0, kh=0, gap=0, silp=0, equiv=0, race=False))
gap_ids = []
for i in ids:
    s, h = sil[i], hil[i]
    op = s["operator"]
    r = rows[op]; r["n"] += 1; r["race"] = s["race_dependent"]
    ks, kh = killed(s["verdict"]), killed(h["verdict"])
    r["ks"] += ks; r["kh"] += kh
    if not ks and kh:   r["gap"]   += 1; gap_ids.append(i)
    elif ks and not kh: r["silp"]  += 1
    elif not ks and not kh: r["equiv"] += 1

hdr = f"{'operator':34} {'n':>3} {'kSIL':>4} {'kHIL':>4} {'GAP':>4} {'sil+':>4} {'equiv':>5} race"
print(hdr); print("-"*len(hdr))
tot = defaultdict(int)
for op in sorted(rows, key=lambda o: (not rows[o]["race"], o)):
    r = rows[op]
    for k in ("n","ks","kh","gap","silp","equiv"): tot[k]+=r[k]
    print(f"{op:34} {r['n']:>3} {r['ks']:>4} {r['kh']:>4} {r['gap']:>4} {r['silp']:>4} {r['equiv']:>5} {'YES' if r['race'] else ''}")
print("-"*len(hdr))
print(f"{'TOTAL':34} {tot['n']:>3} {tot['ks']:>4} {tot['kh']:>4} {tot['gap']:>4} {tot['silp']:>4} {tot['equiv']:>5}")

print(f"\nSIL/HIL GAP = {tot['gap']} mutants killed on HIL but survived on SIL (structural blind spots).")
if gap_ids:
    print("gap mutants:", ", ".join(gap_ids))
print("\nH1 check: the gap should concentrate on sync_removal (the only race-dependent operator present).")
