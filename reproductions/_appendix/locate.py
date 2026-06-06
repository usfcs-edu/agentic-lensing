#!/usr/bin/env python3
"""Print the page (1-based) of every figure/table caption in a PDF.

Uses PyMuPDF text extraction (pdfplumber is not installed in the lens venv).
Helps locate which page a "Figure N" / "Table N" lives on before cropping.

    /raid/benson/.venvs/lens/bin/python3 locate.py <pdf> [--grep TERM]
"""
import argparse
import re
import fitz

# Caption starts: "Figure 4.", "Fig. 4", "Table 2.", "TABLE 2", "Figure B1.4" ...
CAP = re.compile(r"^\s*(Figure|Fig\.?|Table|TABLE|FIG\.?)\s*[A-Z]?\d", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--grep", help="only show caption lines containing this substring")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    print(f"{args.pdf}: {doc.page_count} pages, page0 rect {doc[0].rect}")
    for pno in range(doc.page_count):
        for line in doc[pno].get_text().splitlines():
            s = line.strip()
            if CAP.match(s):
                if args.grep and args.grep.lower() not in s.lower():
                    continue
                print(f"  p{pno + 1:>2} (idx {pno}): {s[:100]}")


if __name__ == "__main__":
    main()
