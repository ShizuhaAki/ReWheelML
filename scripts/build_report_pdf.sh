#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/doc/report"
OUT_MD="${1:-$REPORT_DIR/combined.md}"

mkdir -p "$(dirname "$OUT_MD")"
: >"$OUT_MD"

append_markdown() {
  local file="$1"
  {
    echo
    echo "<!-- BEGIN $(realpath --relative-to="$ROOT_DIR" "$file") -->"
    echo
    # Section files live in doc/report/sections and refer to ../assets/.
    # The combined Markdown is written to doc/report by default, so rewrite
    # those image links to assets/.
    sed 's#](../assets/#](assets/#g' "$file"
    echo
    echo "<!-- END $(realpath --relative-to="$ROOT_DIR" "$file") -->"
  } >>"$OUT_MD"
}

append_markdown "$REPORT_DIR/main.md"

while IFS= read -r section; do
  append_markdown "$section"
done < <(find "$REPORT_DIR/sections" -maxdepth 1 -type f -name '*.md' | sort)

echo "wrote $OUT_MD"
