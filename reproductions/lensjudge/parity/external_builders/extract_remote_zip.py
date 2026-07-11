#!/usr/bin/env python
"""Extract selected files from a remote zip via HTTP range requests.
Usage: extract_remote_zip.py URL pattern [outdir]
Extracts every member whose name contains `pattern` (or matches regex) to outdir (default stdout print).
"""
import re, struct, sys, zlib, os, urllib.request

def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def get_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])

def central_directory(url):
    size = get_size(url)
    tail_len = min(66000, size)
    tail = fetch_range(url, size - tail_len, size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    use64 = False
    total_entries = cd_size = cd_offset = None
    if eocd_pos >= 0:
        eocd = tail[eocd_pos:eocd_pos + 22]
        total_entries = struct.unpack("<H", eocd[10:12])[0]
        cd_size = struct.unpack("<I", eocd[12:16])[0]
        cd_offset = struct.unpack("<I", eocd[16:20])[0]
        if cd_offset == 0xFFFFFFFF or total_entries == 0xFFFF or cd_size == 0xFFFFFFFF:
            use64 = True
    else:
        use64 = True
    if use64:
        loc_pos = tail.rfind(b"PK\x06\x07")
        eocd64_off = struct.unpack("<Q", tail[loc_pos + 8:loc_pos + 16])[0]
        eocd64 = fetch_range(url, eocd64_off, eocd64_off + 55)
        total_entries = struct.unpack("<Q", eocd64[32:40])[0]
        cd_size = struct.unpack("<Q", eocd64[40:48])[0]
        cd_offset = struct.unpack("<Q", eocd64[48:56])[0]
    cd = fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    pos = 0
    entries = []
    while pos < len(cd):
        if cd[pos:pos+4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd[pos+10:pos+12])[0]
        csize = struct.unpack("<I", cd[pos+20:pos+24])[0]
        usize = struct.unpack("<I", cd[pos+24:pos+28])[0]
        name_len = struct.unpack("<H", cd[pos+28:pos+30])[0]
        extra_len = struct.unpack("<H", cd[pos+30:pos+32])[0]
        comment_len = struct.unpack("<H", cd[pos+32:pos+34])[0]
        lho = struct.unpack("<I", cd[pos+42:pos+46])[0]
        name = cd[pos+46:pos+46+name_len].decode("utf-8", "replace")
        extra = cd[pos+46+name_len:pos+46+name_len+extra_len]
        # zip64 extra field
        ep = 0
        vals_needed = [v == 0xFFFFFFFF for v in (usize, csize, lho)]
        while ep + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[ep:ep+4])
            if hid == 0x0001:
                data = extra[ep+4:ep+4+hsz]
                dp = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", data[dp:dp+8])[0]; dp += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", data[dp:dp+8])[0]; dp += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", data[dp:dp+8])[0]; dp += 8
            ep += 4 + hsz
        entries.append((name, method, csize, usize, lho))
        pos += 46 + name_len + extra_len + comment_len
    return entries

def extract(url, entry):
    name, method, csize, usize, lho = entry
    # local header: 30 bytes fixed + name + extra
    lh = fetch_range(url, lho, lho + 29)
    assert lh[:4] == b"PK\x03\x04", lh[:4]
    nlen = struct.unpack("<H", lh[26:28])[0]
    elen = struct.unpack("<H", lh[28:30])[0]
    data_start = lho + 30 + nlen + elen
    raw = fetch_range(url, data_start, data_start + csize - 1)
    if method == 0:
        return raw
    elif method == 8:
        return zlib.decompress(raw, -15)
    raise RuntimeError(f"unsupported method {method}")

if __name__ == "__main__":
    url, pattern = sys.argv[1], sys.argv[2]
    outdir = sys.argv[3] if len(sys.argv) > 3 else None
    rx = re.compile(pattern)
    entries = central_directory(url)
    matched = [e for e in entries if rx.search(e[0])]
    sys.stderr.write(f"{len(matched)} matches\n")
    for e in matched:
        data = extract(url, e)
        if outdir:
            path = os.path.join(outdir, e[0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            sys.stderr.write(f"wrote {path}\n")
        else:
            print(f"===== {e[0]} =====")
            sys.stdout.write(data.decode("utf-8", "replace"))
            print()
