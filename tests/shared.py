"""Shared benchmark data, crawling, and scoring helpers for RAG retrieval tests.

Extracted from test_eval.py so that multiple test files (hybrid search, PageIndex,
BM25-only, LLM-reranking) can reuse the same benchmark queries and infrastructure.
"""

import json
import math
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

# ── Differentiating queries: designed to separate retrieval approaches ────────
#
# Unlike BENCHMARK_QUERIES where BM25 keywords often appear verbatim in the
# content, these queries are crafted so that different retrieval strategies
# produce meaningfully different scores.

DIFFERENTIATING_QUERIES = [
    # ── L6: Paraphrased queries ──────────────────────────────────────────────
    # User uses synonyms / different framing than the actual content.
    # Semantic search should outperform BM25 because the query terms
    # do NOT appear verbatim in the indexed chunks.

    {"q": "Who funds early-stage startups at the course?",
     "keywords": ["khosla", "lux", "link ventures", "e14", "pillar", "venture"],
     "level": "L6",
     "note": "Paraphrased: 'funds early-stage startups' vs 'venture capital' / 'Speaker/Mentor' bios",
     "top_k": 15},

    {"q": "What cutting-edge tech topics are covered in lectures?",
     "keywords": ["agentic", "autonomous", "nanda", "decentralized", "web3", "blockchain"],
     "level": "L6",
     "note": "Paraphrased: 'cutting-edge tech topics' vs actual topic names in content"},

    {"q": "Who helps students refine their business ideas?",
     "keywords": ["mentor", "speaker", "instructor", "co-instructor", "judge", "catalyst"],
     "level": "L6",
     "note": "Paraphrased: 'helps students refine business ideas' vs 'mentor' / 'judge' roles",
     "top_k": 10},

    {"q": "How do teams show off their work at the end of the semester?",
     "keywords": ["demo day", "pitch", "presentation", "showcase", "judge"],
     "level": "L6",
     "note": "Paraphrased: 'show off their work' vs 'demo day' / 'final pitches'"},

    {"q": "What privacy-preserving machine learning method was invented at MIT?",
     "keywords": ["split learning", "federated", "privacy", "no peak", "decentralized"],
     "level": "L6",
     "note": "Paraphrased: 'privacy-preserving ML method' vs 'split learning' in transcripts"},

    # ── L7: Multi-hop queries ────────────────────────────────────────────────
    # Answering requires combining info from 2+ separate chunks or pages.
    # LLM-reranking should excel; single-chunk retrieval will be incomplete.

    {"q": "Which speakers or mentors appeared in both Fall 2023 and Spring 2024?",
     "keywords": ["raskar", "ramesh", "john", "werner", "habib", "hadad", "link"],
     "level": "L7",
     "note": "Multi-hop: must cross-reference people lists from fall23 and spring24 pages",
     "top_k": 15},

    {"q": "How has the number of student teams changed across semesters?",
     "keywords": ["15", "24", "38", "teams", "presenting"],
     "level": "L7",
     "note": "Multi-hop: team counts mentioned in different semester transcripts (15, 24, 38)"},

    {"q": "What real-world companies were both judges and corporate partners at demo days?",
     "keywords": ["mitsubishi", "state street", "ey", "mass challenge"],
     "level": "L7",
     "note": "Multi-hop: company names scattered across multiple demo day transcripts",
     "top_k": 15},

    {"q": "Did the course evaluation criteria change between Spring 2023 and Fall 2025?",
     "keywords": ["impact", "unique", "complete", "demo", "judge", "gigas scale"],
     "level": "L7",
     "note": "Multi-hop: judging criteria described in multiple semester transcripts"},

    # ── L8: Negative / absent queries ────────────────────────────────────────
    # The answer is NOT in the data. A good retrieval system should return
    # low-relevance results. High keyword hits = hallucination risk.
    # NOTE: For L8, keywords list things that SHOULD NOT appear in results.
    # Scoring is inverted: finding these keywords means the system is
    # hallucinating or retrieving irrelevant content.

    {"q": "What is the tuition cost for the AI Studio course?",
     "keywords": ["tuition", "cost", "fee", "price", "dollar", "pay"],
     "level": "L8",
     "note": "Negative: tuition/pricing info is NOT on the course website",
     "scoring": "inverse"},

    {"q": "What programming languages are required as prerequisites?",
     "keywords": ["python", "java", "javascript", "prerequisite", "requirement", "coding"],
     "level": "L8",
     "note": "Negative: no programming prerequisites listed anywhere",
     "scoring": "inverse"},

    {"q": "What is the final exam format for the course?",
     "keywords": ["exam", "test", "midterm", "final", "quiz", "grading"],
     "level": "L8",
     "note": "Negative: no exam/grading info exists in the content",
     "scoring": "inverse"},
]

# ── Complex queries: temporal, reasoning, aggregation ─────────────────────────
#
# These queries go beyond keyword matching and require cross-semester temporal
# reasoning, inference from context, or aggregation across many pages/transcripts.

COMPLEX_QUERIES = [
    # ── L9: Temporal comparison ────────────────────────────────────────────────
    # Answering requires comparing information across multiple semesters.

    {"q": "Which VCs are no longer participating in the course this year?",
     "keywords": ["khosla", "lux", "link", "e14", "pillar", "two lanterns"],
     "level": "L9",
     "note": "Temporal: must compare people-summary pages across 5 semesters to find who left",
     "top_k": 15},

    {"q": "How have the demo day judging criteria evolved from Spring 2023 to Fall 2025?",
     "keywords": ["impact", "unique", "complete", "demo", "judge", "gigas scale", "criteria"],
     "level": "L9",
     "note": "Temporal: judging criteria described in multiple semester transcripts",
     "top_k": 10},

    {"q": "How has the number of student teams grown from Spring 2023 to Fall 2025?",
     "keywords": ["15", "24", "38", "teams", "presenting", "demo"],
     "level": "L9",
     "note": "Temporal: team counts scattered across different semester transcripts"},

    # ── L10: Reasoning required ────────────────────────────────────────────────
    # The answer isn't stated directly -- requires inference from context.

    {"q": "Why does Raskar say NANDA needs to involve universities and not just companies?",
     "keywords": ["university", "nanda", "trust", "decentralized", "neutral", "open"],
     "level": "L10",
     "note": "Reasoning: answer must be inferred from transcript discussion about decentralized networks"},

    {"q": "What does gigascale impact mean in the context of this course?",
     "keywords": ["gigas scale", "billion", "impact", "people", "scale"],
     "level": "L10",
     "note": "Reasoning: concept explained across multiple transcripts with examples"},

    {"q": "What is the quilt approach to building AI products that Raskar describes?",
     "keywords": ["quilt", "quilting", "model", "combine", "agent"],
     "level": "L10",
     "note": "Reasoning: metaphorical concept from transcript, not a standard term"},

    # ── L11: Aggregation ──────────────────────────────────────────────────────
    # Answering requires collecting and combining info from many pages.

    {"q": "Name all the companies that have ever been judges or sponsors at demo days.",
     "keywords": ["mitsubishi", "state street", "ey", "mass challenge", "sponsor", "judge"],
     "level": "L11",
     "note": "Aggregation: company names scattered across many transcript chunks",
     "top_k": 15},

    {"q": "What are all the healthcare or medical AI applications discussed across all semesters?",
     "keywords": ["health", "medical", "patient", "diabetic", "chest", "hospital", "clinical"],
     "level": "L11",
     "note": "Aggregation: medical examples appear in multiple transcripts across semesters",
     "top_k": 10},

    # ── L12: Bio evolution (temporal + aggregation + reasoning) ────────────────
    # Requires retrieving the same person's bio from multiple semester pages,
    # comparing roles/affiliations, and reasoning about which change is "most."

    {"q": "Whose bio or role has changed the most across semesters of the course?",
     "keywords": ["werner", "john", "visiting lecturer", "link ventures", "santanu",
                   "bhattacharya", "airtel", "media lab", "nanda", "chris pease"],
     "level": "L12",
     "note": "Bio evolution: Werner went Speaker→Visiting Lecturer→Link Ventures; "
             "Bhattacharya went Airtel→MIT Media Lab→Airtel; Pease went MIT→Project NANDA. "
             "Requires bios from multiple semesters + reasoning about significance",
     "top_k": 15},
]

# Combined list for convenience
ALL_QUERIES = BENCHMARK_QUERIES + DIFFERENTIATING_QUERIES + COMPLEX_QUERIES

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


def _fts5_query(query, expand=False):
    """Convert a natural-language query to an OR-joined FTS5 query."""
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    terms = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    if not terms:
        terms = re.findall(r"[a-zA-Z0-9]+", query.lower())

    if expand:
        # Domain-specific synonyms for the MIT AI Studio course
        _SYNONYMS = {
            "vc": ["venture", "capital", "investor", "fund"],
            "venture": ["vc", "investor"],
            "demo": ["pitch", "presentation", "showcase"],
            "mentor": ["advisor", "judge", "speaker"],
            "speaker": ["mentor", "lecturer", "guest"],
            "healthcare": ["health", "medical", "clinical", "patient"],
            "medical": ["healthcare", "clinical", "hospital"],
            "privacy": ["private", "confidential", "decentralized"],
            "ai": ["artificial", "intelligence", "machine", "learning"],
            "startup": ["company", "founder", "entrepreneur"],
            "team": ["group", "project", "student"],
            "bio": ["biography", "profile", "background"],
        }
        expanded = set(terms)
        for t in terms:
            if t in _SYNONYMS:
                expanded.update(_SYNONYMS[t])
        terms = list(expanded)

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


def hybrid_search_expanded(conn, emb_model, query, top_k=5, kw=0.3, sw=0.7):
    """Hybrid search with query expansion for BM25 signal."""
    qe = list(emb_model.embed([query]))[0].tolist()
    candidate_k = max(50, top_k * 5)

    try:
        cur = conn.cursor()
        cur.execute("SELECT rowid, bm25(documents_fts) FROM documents_fts WHERE documents_fts MATCH ? LIMIT ?",
                     (_fts5_query(query, expand=True), candidate_k))
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


def hybrid_search_rrf(conn, emb_model, query, top_k=5, rrf_k=60):
    """Hybrid search using Reciprocal Rank Fusion (RRF) instead of linear combination."""
    qe = list(emb_model.embed([query]))[0].tolist()
    candidate_k = max(50, top_k * 5)

    # Get BM25 ranked list
    try:
        cur = conn.cursor()
        cur.execute("SELECT rowid, bm25(documents_fts) FROM documents_fts WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
                     (_fts5_query(query), candidate_k))
        bm25_ranked = [r[0] for r in cur.fetchall()]
    except sqlite3.OperationalError:
        bm25_ranked = []

    # Get semantic ranked list
    cur = conn.cursor()
    cur.execute("SELECT rowid, distance FROM vec_documents WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                 (serialize_embedding(qe), candidate_k))
    sem_ranked = [r[0] for r in cur.fetchall()]

    # RRF: score = sum(1 / (k + rank)) across signals
    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_ranked):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(sem_ranked):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    all_ids = set(rrf_scores.keys())
    if not all_ids:
        return []

    cur = conn.cursor()
    ph = ",".join("?" * len(all_ids))
    cur.execute(f"SELECT id, content, content_type FROM documents WHERE id IN ({ph})", list(all_ids))
    docs = {r[0]: {"content": r[1], "content_type": r[2]} for r in cur.fetchall()}

    results = []
    for did in all_ids:
        doc = docs.get(did, {"content": "", "content_type": "text"})
        results.append({"id": did, "content": doc["content"], "content_type": doc["content_type"], "score": rrf_scores[did]})
    results.sort(key=lambda x: x["score"], reverse=True)

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
    """Score retrieval results by keyword coverage (0.0 to 1.0).

    This is equivalent to Recall@K where K = len(results).
    """
    if not results:
        return 0.0
    content = " ".join(r["content"] for r in results).lower()
    hits = sum(1 for kw in keywords if kw.lower() in content)
    return hits / len(keywords)


# Alias for clarity in metric-oriented code
recall_at_k = score_retrieval


def mrr_score(results, keywords):
    """Mean Reciprocal Rank for a single query.

    Finds the rank of the FIRST result containing ANY expected keyword.
    Returns 1/rank of that first hit, or 0.0 if no result contains any keyword.
    """
    if not results or not keywords:
        return 0.0
    kw_lower = [kw.lower() for kw in keywords]
    for rank, r in enumerate(results, start=1):
        content = r["content"].lower()
        if any(kw in content for kw in kw_lower):
            return 1.0 / rank
    return 0.0


def ndcg_score(results, keywords):
    """Normalized Discounted Cumulative Gain for a single query.

    Relevance of each result at position i = (number of keywords found) / len(keywords).
    DCG  = sum(rel_i / log2(i + 2)) for i in range(K)  (i+2 because 1-indexed positions)
    IDCG = DCG of the ideal ranking (relevances sorted descending)
    NDCG = DCG / IDCG, or 1.0 if IDCG == 0.
    """
    if not results or not keywords:
        return 0.0
    kw_lower = [kw.lower() for kw in keywords]
    n_kw = len(keywords)

    # Compute per-result relevance
    relevances = []
    for r in results:
        content = r["content"].lower()
        hits = sum(1 for kw in kw_lower if kw in content)
        relevances.append(hits / n_kw)

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    # IDCG (ideal: sort relevances descending)
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def pass_at_k(n, c, k):
    """pass@k -- probability that at least 1 of k random samples is correct.

    Given n total attempts with c correct (score >= threshold), computes:
        1 - C(n-c, k) / C(n, k)
    where C is the binomial coefficient.

    Args:
        n: total number of queries (attempts)
        c: number of queries passing the threshold
        k: number of samples drawn
    Returns:
        Probability (float) that at least one of k samples passes.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        k = n
    # Use log-space to avoid overflow with large binomial coefficients:
    # C(n-c, k) / C(n, k) = product((n-c-i) / (n-i)) for i in 0..k-1
    # But only if n-c >= k, otherwise the ratio is 0 (guaranteed success).
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def pass_power_k(n, c, k):
    """pass^k -- reliability metric: (success_rate)^k.

    Measures the probability that ALL k independent samples pass.

    Args:
        n: total number of queries
        c: number passing
        k: exponent (number of independent trials)
    Returns:
        (c/n)^k
    """
    if n <= 0:
        return 0.0
    return (c / n) ** k
