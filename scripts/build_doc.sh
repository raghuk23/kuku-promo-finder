#!/usr/bin/env bash
# Render the solution doc to PDF and report its page count (must be <= 2).
set -euo pipefail
cd "$(dirname "$0")/.."
weasyprint docs/solution.html docs/solution.pdf
python3 -c "import pypdf,sys; n=len(pypdf.PdfReader('docs/solution.pdf').pages); print(f'docs/solution.pdf: {n} page(s)'); sys.exit(0 if n<=2 else 1)"
