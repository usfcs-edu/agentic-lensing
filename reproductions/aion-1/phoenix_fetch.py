"""Standalone GZ-DECaLS cutout fetcher for phoenix (2nd IP, parallel campaign).
Cache-key-compatible with _ls_cutout.py so files rsync straight into the main
cache. Fetches the DISTRACTOR slice (the main box does mergers+spirals)."""
import hashlib, io, time, os, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from astropy.io import fits

CACHE = "/raid/benson/aion_campaign/ls_cutouts"
os.makedirs(CACHE, exist_ok=True)
BASE = "https://www.legacysurvey.org/viewer/cutout.fits"

def key(ra, dec, layer="ls-dr10", size=160, pixscale=0.262):
    return hashlib.md5(f"{ra:.6f}_{dec:.6f}_{layer}_{size}_{pixscale}".encode()).hexdigest()

def fetch_one(ra, dec, retries=6, timeout=45):
    cp = os.path.join(CACHE, key(ra, dec) + ".npy")
    if os.path.exists(cp): return True
    url = f"{BASE}?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&size=160&bands=griz"
    for a in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content[:6] == b"SIMPLE":
                with fits.open(io.BytesIO(r.content)) as h:
                    arr = np.asarray(h[0].data, dtype=np.float32)
                if arr.ndim == 3 and arr.shape[0] == 4:
                    np.save(cp, arr); return True
                return False
            if r.status_code == 429:
                w = float(r.headers.get("Retry-After", 0)) or (1.5 ** a)
                time.sleep(min(w, 8) + random.uniform(0, 1)); continue
            if r.status_code in (404, 500) and b"no overlap" in r.content[:200].lower():
                return False
            time.sleep(1.5 * (a + 1) + random.uniform(0, 1))
        except Exception:
            time.sleep(1.5 * (a + 1) + random.uniform(0, 1))
    return False

t = pd.read_parquet("/raid/benson/aion_campaign/targets.parquet")
t = t[t.label == "distractor"].reset_index(drop=True)
coords = list(zip(t.ra.astype(float), t.dec.astype(float)))
print(f"phoenix campaign: {len(coords)} distractor cutouts", flush=True)
done = {"n": 0, "ok": 0}
def job(i):
    ok = fetch_one(*coords[i]); done["n"] += 1; done["ok"] += int(ok)
    if done["n"] % 200 == 0: print(f"  phoenix {done['n']}/{len(coords)} ok={done['ok']}", flush=True)
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(job, range(len(coords))))
print(f"PHOENIX_CAMPAIGN_OK {done['ok']}/{len(coords)}", flush=True)
