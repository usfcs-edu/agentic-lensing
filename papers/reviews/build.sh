#!/usr/bin/env bash
# Build .docx and .pdf from each review .md in this directory.
#   ./build.sh            # build all
#   ./build.sh 01-foo.md  # build one
set -euo pipefail
cd "$(dirname "$0")"

PDF_OPTS=(
  --pdf-engine=xelatex
  -V geometry:margin=1in
  -V fontsize=11pt
  -V mainfont="Charter"
  -V sansfont="Helvetica Neue"
  -V monofont="Menlo"
  -V linkcolor=NavyBlue
  -V colorlinks=true
  --syntax-highlighting=tango
  --include-in-header=.header.tex
)

files=("$@")
if [ ${#files[@]} -eq 0 ]; then files=(*.md); fi

for f in "${files[@]}"; do
  base="${f%.md}"
  [ "$base" = "README" ] && continue
  echo "==> $f"
  pandoc "$f" -o "$base.docx" --reference-doc=.reference.docx
  pandoc "$f" -o "$base.pdf" "${PDF_OPTS[@]}"
  pages=$(pdfinfo "$base.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  printf '    %-46s %5s words   %s pages\n' "$base" "$(wc -w < "$f" | tr -d ' ')" "${pages:-?}"
done
