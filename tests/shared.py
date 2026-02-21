"""Shared benchmark data, crawling, and scoring helpers for RAG retrieval tests.

Extracted from test_eval.py so that multiple test files (hybrid search, PageIndex,
BM25-only, LLM-reranking) can reuse the same benchmark queries and infrastructure.
"""

import json
import re
import sqlite3
import struct
from pathlib import Path
from urllib.parse import urljoin

import requests
import sqlite_vec
from bs4 import BeautifulSoup
from fastembed import TextEmbedding

# ── Benchmark queries at 5 difficulty levels ─────────────────────────────────

BENCHMARK_QUERIES = [
    # L1: Easy -- direct keyword match
    {"q": "Who is Ramesh Raskar?", "keywords": ["ramesh", "raskar", "professor", "mit"],
     "level": "L1", "note": "Direct name lookup"},
    {"q": "Are there any course videos?", "keywords": ["video", "youtube"],
     "level": "L1", "note": "Content type lookup"},
    {"q": "What is the course about?",
     "keywords": ["ai", "venture", "prototyping", "hands-on"],
     "level": "L1", "note": "Course description from overview"},

    # L2: Medium -- requires semantic understanding
    {"q": "How do I register for the course?", "keywords": ["register", "questionnaire", "apply", "form"],
     "level": "L2", "note": "Actionable info"},
    {"q": "What is the course format for Fall 2025?",
     "keywords": ["venture studio", "agentic", "demo day"],
     "level": "L2", "note": "Semester-specific overview"},

    # L3: Hard -- needs cross-page reasoning
    {"q": "What venture capital firms are involved as speakers or mentors?",
     "keywords": ["khosla", "lux", "link ventures", "two lanterns", "pillar", "e14"],
     "level": "L3", "note": "Aggregate across bios -- VC firms in Role field",
     "top_k": 15},
    {"q": "Who were the venture capital speakers in Fall 2023?",
     "keywords": ["khosla", "lux", "link", "e14", "pillar"],
     "level": "L3", "note": "Semester-specific VC list -- needs many bios from one page",
     "top_k": 15},
    {"q": "How has the course name changed across semesters?", "keywords": ["web3", "venture", "agentic", "foundations"],
     "level": "L3", "note": "Cross-page temporal"},
    {"q": "What changed between the 2025 and 2026 versions of the course?",
     "keywords": ["agentic", "autonomous", "venture studio", "foundations", "spring 2026", "spring 2025"],
     "level": "L3", "note": "Cross-page temporal comparison"},

    # L4: Very hard -- needs transcript content
    {"q": "What is NANDA and why does Raskar think it matters?",
     "keywords": ["nanda", "network", "agent", "decentralized", "internet"],
     "level": "L4", "note": "Only in video transcripts"},
    {"q": "What did Raskar say about privacy and decentralized AI?",
     "keywords": ["privacy", "decentralized", "data", "machine learning"],
     "level": "L4", "note": "Deep transcript content"},

    # L5: Expert -- nuanced cross-source reasoning
    {"q": "What companies like Mitsubishi were judges or partners at Demo Day?",
     "keywords": ["mitsubishi", "judge", "partner", "corporate"],
     "level": "L5", "note": "Specific detail in transcript"},
    {"q": "How does Raskar compare the internet evolution to the agentic web?",
     "keywords": ["internet", "worldwide web", "mainframe", "intranet", "agentic"],
     "level": "L5", "note": "Conceptual from transcript"},
    {"q": "What healthcare or medical examples did speakers discuss?",
     "keywords": ["health", "medical", "patient", "hospital", "diabetic", "chest"],
     "level": "L5", "note": "Cross-source thematic"},
]

# Subset: "hard" queries (L3+) that hybrid search struggles with
HARD_QUERIES = [bq for bq in BENCHMARK_QUERIES if bq["level"] in ("L3", "L4", "L5")]

SITE_PAGES = [
    "https://aiforimpact.github.io/",
    "https://aiforimpact.github.io/spring26.html",
    "https://aiforimpact.github.io/fall25.html",
    "https://aiforimpact.github.io/spring25.html",
    "https://aiforimpact.github.io/fall24.html",
    "https://aiforimpact.github.io/spring24.html",
    "https://aiforimpact.github.io/fall23.html",
]

BIO_LABELS = {"lead professor:", "co-instructor:", "instructor:", "course ta:",
              "ta:", "course instructor:", "professor:", "guest speaker:",
              "speaker:", "mentor:", "judge:", "panelist:", "moderator:"}


# ── Crawling helpers ─────────────────────────────────────────────────────────

def get_youtube_title(video_id):
    """Fetch YouTube video title via oEmbed API."""
    try:
        resp = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("title", ""), data.get("author_name", "")
    except Exception:
        pass
    return "", ""


def crawl_page(url):
    """Crawl a single page and extract structured chunks (bios, videos, text)."""
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")
    title = soup.title.string.strip() if soup.title else url.split("/")[-1]
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    chunks = []
    seen_names = set()

    for card in soup.select(".card, .speaker-card, .col-md-3, .col-lg-3"):
        img = card.find("img")
        text_parts = [t.strip() for t in card.stripped_strings]
        if not text_parts or len(text_parts) < 2:
            continue
        name_idx = 0
        if text_parts[0].lower().rstrip(":") + ":" in BIO_LABELS or text_parts[0].lower() in BIO_LABELS:
            name_idx = 1
            if len(text_parts) < 3:
                continue
        name = text_parts[name_idx]
        label = text_parts[0] if name_idx > 0 else ""
        role = " ".join(text_parts[name_idx + 1:])
        img_url = urljoin(url, img["src"]) if img and img.get("src") else None
        name_key = name.lower().strip()
        if name_key in seen_names or len(name) < 2:
            continue
        seen_names.add(name_key)
        bio_text = f"Speaker/Mentor: {name}."
        if label:
            bio_text += f" Position: {label.rstrip(':').strip()}."
        bio_text += f" Role: {role}."
        if img_url:
            bio_text += f" Photo: {img_url}"
        chunks.append({"content": bio_text, "section_title": f"Bio: {name}",
                        "content_type": "bio", "metadata": json.dumps({"name": name})})

    # Generate a people-summary chunk grouping all bios on this page
    bio_chunks = [c for c in chunks if c["content_type"] == "bio"]
    if bio_chunks:
        semester = ""
        page_file = url.rstrip("/").split("/")[-1].replace(".html", "")
        sem_match = re.match(r"(spring|fall)(\d{2})", page_file)
        if sem_match:
            semester = f"{sem_match.group(1).title()} 20{sem_match.group(2)}"

        by_label = {}
        for c in bio_chunks:
            content = c["content"]
            name_match = re.search(r"Speaker/Mentor: (.+?)\.", content)
            role_match = re.search(r"Role: (.+?)\.(?:\s|$)", content)
            label_match = re.search(r"Position: (.+?)\.", content)
            name = name_match.group(1) if name_match else "?"
            role = role_match.group(1) if role_match else ""
            label = label_match.group(1) if label_match else "Speaker"
            entry = f"{name} ({role})" if role else name
            by_label.setdefault(label, []).append(entry)
        summary_parts = []
        for label, people in by_label.items():
            summary_parts.append(f"{label}s: " + ", ".join(people))
        header = f"People and organizations for {title}"
        if semester:
            header += f" ({semester})"
        summary = header + ":\n" + "\n".join(summary_parts)
        section = f"People: {semester or title[:50]}"
        chunks.append({"content": summary, "section_title": section,
                        "content_type": "text", "metadata": "{}"})

    seen_videos = set()
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if "youtube" not in src and "youtu.be" not in src:
            continue
        match = re.search(r'(?:embed/|v=|youtu\.be/)([a-zA-Z0-9_-]{11})', src)
        if not match:
            continue
        vid_id = match.group(1)
        if vid_id in seen_videos:
            continue
        seen_videos.add(vid_id)
        real_title, author = get_youtube_title(vid_id)
        video_title = real_title or iframe.get("title", "Course Video")
        content = f"Video: {video_title}."
        if author:
            content += f" By: {author}."
        content += f" YouTube URL: https://www.youtube.com/watch?v={vid_id}."
        chunks.append({"content": content, "section_title": f"Video: {video_title[:60]}",
                        "content_type": "video",
                        "metadata": json.dumps({"video_id": vid_id, "title": video_title})})

    bio_img_urls = set()
    for c in chunks:
        if c["content_type"] == "bio":
            meta = json.loads(c["metadata"])
            if meta.get("image_url"):
                bio_img_urls.add(meta["image_url"])
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        if not src or src.startswith("data:") or not alt:
            continue
        img_url = urljoin(url, src)
        if img_url in bio_img_urls:
            continue
        parent_text = img.parent.get_text(strip=True)[:200] if img.parent else ""
        chunks.append({"content": f"Image: {alt}. URL: {img_url}. Context: {parent_text}",
                        "section_title": f"Image: {alt[:50]}", "content_type": "image", "metadata": "{}"})

    main = soup.find("main") or soup.find("body")
    if main:
        current_section = "Overview"
        current_text = []
        for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
            if el.name in ["h1", "h2", "h3"]:
                if current_text:
                    text = "\n".join(current_text).strip()
                    if len(text) > 30:
                        chunks.append({"content": f"{current_section}\n\n{text}",
                                        "section_title": current_section, "content_type": "text", "metadata": "{}"})
                current_section = el.get_text(strip=True)
                current_text = []
            else:
                text = el.get_text(strip=True)
                if text and len(text) > 5:
                    current_text.append(text)
        if current_text:
            text = "\n".join(current_text).strip()
            if len(text) > 30:
                chunks.append({"content": f"{current_section}\n\n{text}",
                                "section_title": current_section, "content_type": "text", "metadata": "{}"})
    return {"url": url, "title": title, "chunks": chunks}


def load_transcripts():
    """Load pre-downloaded transcripts from data/transcripts/*.txt."""
    transcript_dir = Path(__file__).parent.parent / "data" / "transcripts"
    chunks = []
    for txt_file in sorted(transcript_dir.glob("*.txt")):
        lines = txt_file.read_text().splitlines()
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
        title = meta.get("title", txt_file.stem)
        vid_id = txt_file.stem
        content_lines = lines[content_start:]
        current_chunk, current_words, chunk_num = [], 0, 0
        for line in content_lines:
            if not line.strip():
                continue
            current_chunk.append(line)
            current_words += len(line.split())
            if current_words >= 500:
                chunk_num += 1
                chunks.append({
                    "content": f"Video transcript: {title}\nhttps://www.youtube.com/watch?v={vid_id}\n\n" + "\n".join(current_chunk),
                    "section_title": f"Transcript: {title[:45]} (part {chunk_num})",
                    "content_type": "transcript",
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "page_title": title, "metadata": "{}"})
                current_chunk, current_words = [], 0
        if current_chunk:
            chunk_num += 1
            chunks.append({
                "content": f"Video transcript: {title}\nhttps://www.youtube.com/watch?v={vid_id}\n\n" + "\n".join(current_chunk),
                "section_title": f"Transcript: {title[:45]} (part {chunk_num})",
                "content_type": "transcript",
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "page_title": title, "metadata": "{}"})
    return chunks


# ── Embedding and DB helpers ─────────────────────────────────────────────────

def serialize_embedding(emb):
    """Serialize a float embedding list to bytes for sqlite-vec."""
    return struct.pack(f"{len(emb)}f", *emb)


def build_rag_db(db_path):
    """Build full RAG database: crawl site + load transcripts + embed + store.

    Returns (conn, emb_model, all_chunks).
    """
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()
    cur.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, page_title TEXT, section_title TEXT, content TEXT, content_type TEXT, metadata TEXT)")
    cur.execute("CREATE VIRTUAL TABLE vec_documents USING vec0(embedding float[384] distance_metric=cosine)")
    cur.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(content, page_title, section_title, content='documents', content_rowid='id')")
    cur.execute("CREATE TRIGGER docs_ai AFTER INSERT ON documents BEGIN INSERT INTO documents_fts(rowid, content, page_title, section_title) VALUES (new.id, new.content, new.page_title, new.section_title); END")
    conn.commit()

    all_chunks = []
    for url in SITE_PAGES:
        page = crawl_page(url)
        for c in page["chunks"]:
            c["url"] = url
            c["page_title"] = page["title"]
        all_chunks.extend(page["chunks"])

    all_chunks.extend(load_transcripts())

    emb_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    for i in range(0, len(all_chunks), 32):
        batch = all_chunks[i:i + 32]
        texts = [c["content"] for c in batch]
        embs = [e.tolist() for e in emb_model.embed(texts)]
        for chunk, emb in zip(batch, embs):
            cur.execute("INSERT INTO documents (url,page_title,section_title,content,content_type,metadata) VALUES (?,?,?,?,?,?)",
                (chunk.get("url", ""), chunk.get("page_title", ""), chunk.get("section_title", ""),
                 chunk["content"], chunk["content_type"], chunk.get("metadata", "{}")))
            rid = cur.lastrowid
            cur.execute("INSERT INTO vec_documents (rowid, embedding) VALUES (?,?)", (rid, serialize_embedding(emb)))
        conn.commit()

    return conn, emb_model, all_chunks


# ── Search functions ─────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being do does did have has had "
    "will would shall should may might can could of in on at to for with "
    "and or not no by from as it its this that these those i we you he she "
    "they what which who whom how why when where there here am is".split()
)


def _fts5_query(query):
    """Convert a natural-language query to an OR-joined FTS5 query."""
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    terms = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    if not terms:
        terms = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return " OR ".join(f'"{t}"' for t in terms)


def hybrid_search(conn, emb_model, query, top_k=5, kw=0.3, sw=0.7):
    """Hybrid BM25 + semantic vector search."""
    qe = list(emb_model.embed([query]))[0].tolist()
    candidate_k = max(50, top_k * 5)

    try:
        cur = conn.cursor()
        cur.execute("SELECT rowid, bm25(documents_fts) FROM documents_fts WHERE documents_fts MATCH ? LIMIT ?",
                     (_fts5_query(query), candidate_k))
        bm25_raw = {r[0]: r[1] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        bm25_raw = {}

    cur = conn.cursor()
    cur.execute("SELECT rowid, distance FROM vec_documents WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                 (serialize_embedding(qe), candidate_k))
    sem_raw = {r[0]: r[1] for r in cur.fetchall()}

    def norm(scores, hib=True):
        if not scores:
            return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mn == mx:
            return {k: 1.0 for k in scores}
        if hib:
            return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
        return {k: (mx - v) / (mx - mn) for k, v in scores.items()}

    bm25_n = norm(bm25_raw, hib=False)
    sem_n = norm(sem_raw, hib=False)
    all_ids = set(bm25_n) | set(sem_n)
    if not all_ids:
        return []

    cur = conn.cursor()
    ph = ",".join("?" * len(all_ids))
    cur.execute(f"SELECT id, content, content_type FROM documents WHERE id IN ({ph})", list(all_ids))
    docs = {r[0]: {"content": r[1], "content_type": r[2]} for r in cur.fetchall()}

    results = []
    for did in all_ids:
        score = kw * bm25_n.get(did, 0) + sw * sem_n.get(did, 0)
        doc = docs.get(did, {"content": "", "content_type": "text"})
        results.append({"id": did, "content": doc["content"], "content_type": doc["content_type"], "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)

    # Content-type diversity: cap any single type at (top_k - 1)
    max_per_type = max(top_k - 1, 1)
    diverse = []
    type_counts = {}
    for r in results:
        ct = r.get("content_type", "text")
        type_counts[ct] = type_counts.get(ct, 0) + 1
        if type_counts[ct] <= max_per_type:
            diverse.append(r)
        if len(diverse) >= top_k:
            break

    return diverse[:top_k]


def bm25_only_search(conn, query, top_k=5):
    """BM25-only keyword search (no embeddings)."""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT rowid, bm25(documents_fts) FROM documents_fts WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
            (_fts5_query(query), top_k))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []

    if not rows:
        return []

    results = []
    for rowid, bm25_score in rows:
        cur2 = conn.cursor()
        cur2.execute("SELECT id, content, content_type FROM documents WHERE id = ?", (rowid,))
        row = cur2.fetchone()
        if row:
            results.append({"id": row[0], "content": row[1], "content_type": row[2], "score": -bm25_score})
    return results


def semantic_only_search(conn, emb_model, query, top_k=5):
    """Pure semantic vector search (no BM25)."""
    qe = list(emb_model.embed([query]))[0].tolist()
    cur = conn.cursor()
    cur.execute("SELECT rowid, distance FROM vec_documents WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_embedding(qe), top_k))
    rows = cur.fetchall()
    if not rows:
        return []

    results = []
    for rowid, distance in rows:
        cur2 = conn.cursor()
        cur2.execute("SELECT id, content, content_type FROM documents WHERE id = ?", (rowid,))
        row = cur2.fetchone()
        if row:
            results.append({"id": row[0], "content": row[1], "content_type": row[2], "score": 1.0 - distance})
    return results


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_retrieval(results, keywords):
    """Score retrieval results by keyword coverage (0.0 to 1.0)."""
    if not results:
        return 0.0
    content = " ".join(r["content"] for r in results).lower()
    hits = sum(1 for kw in keywords if kw.lower() in content)
    return hits / len(keywords)
