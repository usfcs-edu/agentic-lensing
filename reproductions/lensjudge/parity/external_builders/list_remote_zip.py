#!/usr/bin/env python
"""List contents of a remote zip via HTTP range requests (EOCD + central directory)."""
import struct, sys, urllib.request

def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def get_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])

def list_zip(url):
    size = get_size(url)
    tail_len = min(66000, size)
    tail = fetch_range(url, size - tail_len, size - 1)
    # find EOCD
    eocd_pos = tail.rfind(b"PK\x05\x06")
    names = []
    use64 = False
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
        if loc_pos < 0:
            raise RuntimeError("no zip64 EOCD locator")
        eocd64_off = struct.unpack("<Q", tail[loc_pos + 8:loc_pos + 16])[0]
        eocd64 = fetch_range(url, eocd64_off, eocd64_off + 55)
        assert eocd64[:4] == b"PK\x06\x06"
        total_entries = struct.unpack("<Q", eocd64[32:40])[0]
        cd_size = struct.unpack("<Q", eocd64[40:48])[0]
        cd_offset = struct.unpack("<Q", eocd64[48:56])[0]
    cd = fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    pos = 0
    while pos < len(cd) and len(names) < total_entries:
        if cd[pos:pos+4] != b"PK\x01\x02":
            break
        name_len = struct.unpack("<H", cd[pos+28:pos+30])[0]
        extra_len = struct.unpack("<H", cd[pos+30:pos+32])[0]
        comment_len = struct.unpack("<H", cd[pos+32:pos+34])[0]
        name = cd[pos+46:pos+46+name_len].decode("utf-8", "replace")
        names.append(name)
        pos += 46 + name_len + extra_len + comment_len
    return names

if __name__ == "__main__":
    url = sys.argv[1]
    for n in list_zip(url):
        print(n)
