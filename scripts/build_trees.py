#!/usr/bin/env python3
"""Build tree JSON files from HTML pages and transcripts.

Usage:
    uv run python scripts/build_trees.py [--output-dir data/trees] [--generate-summaries]

Generates hierarchical tree structures for PageIndex-style tree search.
Each page/transcript gets its own JSON file in data/trees/ plus a _manifest.json.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Add project root so we can import the tree parser
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.md_tree_parser import md_string_to_tree

# ── Configuration ─────────────────────────────────────────────────────────────

SITE_PAGES = [
    ("index", "https://aiforimpact.github.io/"),
    ("spring26", "https://aiforimpact.github.io/spring26.html"),
    ("fall25", "https://aiforimpact.github.io/fall25.html"),
    ("spring25", "https://aiforimpact.github.io/spring25.html"),
    ("fall24", "https://aiforimpact.github.io/fall24.html"),
    ("spring24", "https://aiforimpact.github.io/spring24.html"),
    ("fall23", "https://aiforimpact.github.io/fall23.html"),
]

BIO_LABELS = {"lead professor:", "co-instructor:", "instructor:", "course ta:",
              "ta:", "course instructor:", "professor:", "guest speaker:",
              "speaker:", "mentor:", "judge:", "panelist:", "moderator:"}

TRANSCRIPT_DIR = Path(__file__).parent.parent / "data" / "transcripts"


# ── HTML → Markdown ──────────────────────────────────────────────────────────


def html_to_markdown(url):
    """Fetch an HTML page and convert to clean Markdown."""
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")
    title = soup.title.string.strip() if soup.title else url.split("/")[-1]

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    lines = [f"# {title}", ""]
    main = soup.find("main") or soup.find("body")
    if not main:
        return "\n".join(lines), title

    # Track seen bio names to avoid duplicates
    seen_bios = set()

    for el in main.descendants:
        if not hasattr(el, "name") or el.name is None:
            continue

        # Headings
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            text = el.get_text(strip=True)
            if text and text != title:
                lines.append("")
                lines.append(f"{'#' * level} {text}")
                lines.append("")

        # Bio cards — convert to ### Name\nRole, Org format
        elif el.name in ("div", "article") and _is_bio_card(el):
            bio_md = _bio_card_to_md(el, seen_bios)
            if bio_md:
                lines.append(bio_md)

        # Paragraphs and list items (skip if inside a card we already processed)
        elif el.name in ("p", "li") and not _has_card_ancestor(el):
            text = el.get_text(strip=True)
            if text and len(text) > 10:
                prefix = "- " if el.name == "li" else ""
                lines.append(f"{prefix}{text}")
                lines.append("")

    return "\n".join(lines), title


def _is_bio_card(el):
    """Check if element looks like a bio/speaker card."""
    classes = " ".join(el.get("class", []))
    if any(c in classes for c in ("card", "speaker", "col-md-3", "col-lg-3")):
        # Must have at least a name and some text
        text_parts = [t.strip() for t in el.stripped_strings]
        return len(text_parts) >= 2
    return False


def _has_card_ancestor(el):
    """Check if element is inside a bio card."""
    for parent in el.parents:
        if parent.name in ("div", "article") and _is_bio_card(parent):
            return True
    return False


def _bio_card_to_md(card, seen_bios):
    """Convert a bio card element to Markdown."""
    text_parts = [t.strip() for t in card.stripped_strings]
    if not text_parts or len(text_parts) < 2:
        return ""

    name_idx = 0
    if text_parts[0].lower().rstrip(":") + ":" in BIO_LABELS or text_parts[0].lower() in BIO_LABELS:
        name_idx = 1
        if len(text_parts) < 3:
            return ""

    name = text_parts[name_idx]
    if name.lower() in seen_bios or len(name) < 2:
        return ""
    seen_bios.add(name.lower())

    label = text_parts[0] if name_idx > 0 else ""
    role = " ".join(text_parts[name_idx + 1:])

    parts = [f"### {name}"]
    if label:
        parts.append(f"{label.rstrip(':').strip()}")
    if role:
        parts.append(role)
    parts.append("")
    return "\n".join(parts)


# ── Transcripts → Markdown ───────────────────────────────────────────────────


def transcript_to_markdown(txt_path):
    """Convert a transcript .txt file to Markdown with section headings."""
    lines = txt_path.read_text().splitlines()
    meta = {}
    content_start = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            kv = line[2:].strip()
            if ": " in kv:
                k, v = kv.split(": ", 1)
                meta[k.lower()] = v
            elif not meta:
                meta["title"] = kv
        else:
            content_start = i
            break

    title = meta.get("title", txt_path.stem)
    vid_id = txt_path.stem
    content_lines = lines[content_start:]

    md_lines = [
        f"# {title}",
        f"Video: https://www.youtube.com/watch?v={vid_id}",
        "",
    ]

    # Inject ## Part N headings every ~500 words
    word_count = 0
    part_num = 0
    for line in content_lines:
        if not line.strip():
            continue
        words = len(line.split())
        if word_count >= 500:
            part_num += 1
            md_lines.append("")
            md_lines.append(f"## Part {part_num}")
            md_lines.append("")
            word_count = 0
        md_lines.append(line)
        word_count += words

    return "\n".join(md_lines), title, vid_id


# ── Summary generation ───────────────────────────────────────────────────────


def _count_tokens_approx(text):
    """Rough token count (~0.75 words per token)."""
    return len(text.split()) * 4 // 3


def generate_summaries(tree_data, client, model="google/gemini-2.0-flash-001"):
    """Walk tree and generate summaries for nodes with substantial text."""
    structure = tree_data["structure"]
    _summarize_nodes(structure, client, model)
    return tree_data


def _summarize_nodes(nodes, client, model):
    """Recursively summarize nodes with text > 200 tokens."""
    for node in nodes:
        text = node.get("text", "")
        if _count_tokens_approx(text) > 200:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": (
                        f"Summarize this section in 1-2 sentences. "
                        f"Focus on key people, topics, and facts.\n\n{text[:3000]}"
                    )}],
                    temperature=0.0,
                    max_tokens=150,
                )
                node["summary"] = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"    Warning: summary failed for '{node.get('title', '?')}': {e}")
        if node.get("children"):
            _summarize_nodes(node["children"], client, model)


def generate_doc_description(tree_data, client, model="google/gemini-2.0-flash-001"):
    """Generate a one-line description for a document."""
    # Collect top-level headings and some text
    titles = []
    sample_text = ""
    for node in tree_data["structure"]:
        titles.append(node["title"])
        if node.get("text") and len(sample_text) < 500:
            sample_text += node["text"][:200] + " "
    context = f"Document: {tree_data['doc_name']}\nSections: {', '.join(titles)}\nSample: {sample_text[:500]}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": (
                f"Write a one-line description (under 100 words) of what this document covers.\n\n{context}"
            )}],
            temperature=0.0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Warning: doc description failed for '{tree_data['doc_name']}': {e}")
        return ""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Build tree JSONs from HTML pages + transcripts")
    parser.add_argument("--output-dir", type=Path, default=Path("data/trees"))
    parser.add_argument("--generate-summaries", action="store_true",
                        help="Generate LLM summaries (requires OPENROUTER_API_KEY)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"documents": []}

    # Optional: load .env for API key
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    client = None
    if args.generate_summaries:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("Error: --generate-summaries requires OPENROUTER_API_KEY")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # ── HTML pages ────────────────────────────────────────────────────────
    print("Processing HTML pages...")
    for name, url in SITE_PAGES:
        print(f"  {name}: {url}")
        md_text, title = html_to_markdown(url)
        tree = md_string_to_tree(md_text, name)

        if args.generate_summaries and client:
            print(f"    Generating summaries...")
            generate_summaries(tree, client)
            desc = generate_doc_description(tree, client)
        else:
            desc = _auto_description(name, title, "page")

        out_path = args.output_dir / f"{name}.json"
        out_path.write_text(json.dumps(tree, indent=2))
        print(f"    -> {out_path} ({_count_nodes(tree['structure'])} nodes)")

        manifest["documents"].append({
            "doc_name": name,
            "type": "page",
            "source_url": url,
            "title": title,
            "description": desc,
            "file": f"{name}.json",
        })

    # ── Transcripts ───────────────────────────────────────────────────────
    print("\nProcessing transcripts...")
    index_path = TRANSCRIPT_DIR / "index.json"
    transcript_index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for txt_file in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        vid_id = txt_file.stem
        print(f"  {vid_id}: {txt_file.name}")
        md_text, title, _ = transcript_to_markdown(txt_file)
        tree = md_string_to_tree(md_text, vid_id)

        if args.generate_summaries and client:
            print(f"    Generating summaries...")
            generate_summaries(tree, client)
            desc = generate_doc_description(tree, client)
        else:
            meta = transcript_index.get(vid_id, {}).get("meta", {})
            desc = _auto_description(vid_id, title, "transcript", meta)

        out_path = args.output_dir / f"{vid_id}.json"
        out_path.write_text(json.dumps(tree, indent=2))
        print(f"    -> {out_path} ({_count_nodes(tree['structure'])} nodes)")

        manifest["documents"].append({
            "doc_name": vid_id,
            "type": "transcript",
            "source_url": f"https://www.youtube.com/watch?v={vid_id}",
            "title": title,
            "description": desc,
            "file": f"{vid_id}.json",
        })

    # ── Write manifest ────────────────────────────────────────────────────
    manifest_path = args.output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path} ({len(manifest['documents'])} documents)")
    print("Done!")


def _count_nodes(tree):
    """Count total nodes in a tree."""
    count = len(tree)
    for node in tree:
        if node.get("children"):
            count += _count_nodes(node["children"])
    return count


def _auto_description(name, title, doc_type, meta=None):
    """Generate a simple description without LLM."""
    if doc_type == "page":
        sem_match = re.match(r"(spring|fall)(\d{2})", name)
        if sem_match:
            semester = f"{sem_match.group(1).title()} 20{sem_match.group(2)}"
            return f"MIT AI Studio course page for {semester}. Lists instructors, speakers, mentors, and course details."
        if name == "index":
            return "MIT AI Studio main page. Overview of the course, current semester info, and speaker/mentor directory."
        return f"MIT AI Studio page: {title}"
    elif doc_type == "transcript":
        duration = meta.get("duration_str", "") if meta else ""
        channel = meta.get("channel", "") if meta else ""
        desc = f"Video transcript: {title}."
        if channel:
            desc += f" Channel: {channel}."
        if duration:
            desc += f" Duration: {duration}."
        return desc
    return title


if __name__ == "__main__":
    main()
