"""Loading and chunking the knowledge base, two ways.

naive: what most first versions do. Read the raw file, cut every 1,000
characters. Front matter ends up in chunks, tables get cut away from their
header row, and a chunk doesn't say which article or section it came from.

fixed: split on Markdown headings first, write each table row out as a
self-describing line ("Team: Monthly billing $12; ..."), and start every chunk
with its breadcrumb ("Plans and pricing > Price per user") so both the
retriever and the model know where a fact lives.
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def parse_front_matter(raw: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if not m:
        return {}, raw
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, raw[m.end():]


def load_kb(kb_dir: Path = KB_DIR) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        meta, body = parse_front_matter(raw)
        docs.append(Document(page_content=body.strip(), metadata={
            "source": path.name, "title": meta.get("title", path.stem), "updated": meta.get("updated", ""), "raw": raw}))
    return docs


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def serialize_tables(text: str) -> str:
    """Rewrite Markdown tables as one line per row, each carrying its column names."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            header = [c.strip() for c in lines[i].strip("|").split("|")]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                pairs = "; ".join(f"{h} {c}" if h else c for h, c in zip(header[1:], cells[1:]))
                out.append(f"- {cells[0]}: {pairs}.")
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def split_naive(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = []
    for d in docs:
        for j, piece in enumerate(splitter.split_text(d.metadata["raw"])):
            chunks.append(Document(page_content=piece, metadata={
                "id": f"{d.metadata['source']}#{j}", "source": d.metadata["source"],
                "title": d.metadata["title"], "section": f"chunk {j + 1}", "updated": d.metadata["updated"], "anchor": ""}))
    return chunks


def split_fixed(docs: list[Document]) -> list[Document]:
    by_heading = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)
    # Sections in a help center are short; only long ones get cut, on paragraph and list boundaries.
    sizer = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n- ", "\n", ". ", " "])
    chunks = []
    for d in docs:
        for sec in by_heading.split_text(d.page_content):
            heads = [sec.metadata[h] for h in ("h2", "h3") if h in sec.metadata]
            section = " > ".join(heads) or "Overview"
            body = serialize_tables(sec.page_content)
            for j, piece in enumerate(sizer.split_text(body)):
                crumb = " > ".join([d.metadata["title"], *heads])
                chunks.append(Document(page_content=f"{crumb}\n{piece}", metadata={
                    "id": f"{d.metadata['source']}#{slug(section)}-{j}", "source": d.metadata["source"],
                    "title": d.metadata["title"], "section": section, "updated": d.metadata["updated"],
                    "anchor": slug(heads[-1]) if heads else ""}))
    return chunks
