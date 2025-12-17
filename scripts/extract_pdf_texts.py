"""
Extract text from all PDFs under docs/ into UTF-8 txt files.

Usage:
  python scripts/extract_pdf_texts.py [--output-dir docs/pdf_text]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PyPDF2 import PdfReader

DOCS_DIR = Path("docs")


def extract_pdf(pdf_path: Path, out_dir: Path) -> None:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            chunks.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to read page {idx} of {pdf_path.name}: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pdf_path.stem + ".txt")
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    dest = out_path.resolve()
    base = Path.cwd().resolve()
    try:
        rel = dest.relative_to(base)
    except ValueError:
        rel = dest
    print(f"[ok] {pdf_path.name} -> {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from all PDFs in docs/.")
    parser.add_argument("--output-dir", default=DOCS_DIR / "pdf_text", help="Directory to store extracted txt files.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        return

    for pdf_path in pdfs:
        extract_pdf(pdf_path, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
