#!/usr/bin/env python3
"""Render or crop a region of a PDF page to PNG using PyMuPDF (fitz).

The only working rasterizer on this aarch64 host (no ghostscript / poppler /
imagemagick). Used to extract published figures/tables from papers/*.pdf and to
crop reproduced tables out of each report's compiled main.pdf, for the
side-by-side comparison appendices.

Run with the lens venv:
    /raid/benson/.venvs/lens/bin/python3 crop_pdf.py <pdf> <page> <out.png> \
        [--bbox x0 y0 x1 y1] [--dpi 300]

PAGE is 1-based (the number you'd cite, = fitz page index + 1).
BBOX is in PDF points (origin top-left; 72 pt = 1 inch). Omit --bbox to render
the whole page (useful for eyeballing where to crop). Page size is printed so
you can pick a box; e.g. a letter page is 612 x 792 pt.
"""
import argparse
import fitz


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("page", type=int, help="1-based page number")
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="crop box in PDF points (top-left origin); omit for full page")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    idx = args.page - 1
    if not (0 <= idx < doc.page_count):
        raise SystemExit(f"page {args.page} out of range 1..{doc.page_count}")
    page = doc[idx]
    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    clip = fitz.Rect(*args.bbox) if args.bbox else None
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(args.out)
    print(f"page rect (pt): {page.rect}")
    print(f"wrote {args.out}: {pix.width}x{pix.height}px @ {args.dpi}dpi"
          + (f"  clip={args.bbox}" if args.bbox else "  (full page)"))


if __name__ == "__main__":
    main()
