"""Markdown tree parser — vendored from PageIndex (MIT License).

Extracts heading structure from Markdown text and builds a hierarchical tree.
Based on VectifyAI/PageIndex `pageindex/page_index_md.py`.
https://github.com/VectifyAI/PageIndex

Zero external dependencies.
"""

import re

# Matches Markdown headings: # H1, ## H2, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def extract_nodes_from_markdown(lines):
    """Extract heading nodes from markdown lines.

    Returns list of dicts: {"level": int, "title": str, "line_idx": int}
    Skips headings inside fenced code blocks (``` ... ```).
    """
    nodes = []
    in_code_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            nodes.append({
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
                "line_idx": i,
            })
    return nodes


def extract_node_text_content(node_list, lines):
    """Fill in the text content between each heading and the next.

    Mutates each node in node_list to add a "text" field containing the
    non-heading content lines between this node and the next node (or EOF).
    """
    for i, node in enumerate(node_list):
        start = node["line_idx"] + 1
        end = node_list[i + 1]["line_idx"] if i + 1 < len(node_list) else len(lines)
        text_lines = []
        in_code = False
        for j in range(start, end):
            stripped = lines[j].strip()
            if stripped.startswith("```"):
                in_code = not in_code
            if stripped and not _HEADING_RE.match(stripped):
                text_lines.append(lines[j])
        node["text"] = "\n".join(text_lines).strip()
    return node_list


def build_tree_from_nodes(node_list):
    """Build a nested tree from a flat list of heading nodes using a stack.

    Returns a list of root-level nodes, each with a "children" list.
    """
    root = {"level": 0, "title": "root", "children": [], "text": ""}
    stack = [root]

    for node in node_list:
        tree_node = {
            "level": node["level"],
            "title": node["title"],
            "text": node.get("text", ""),
            "children": [],
        }
        # Pop stack until we find a parent with lower level
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if not stack:
            stack = [root]
        stack[-1]["children"].append(tree_node)
        stack.append(tree_node)

    return root["children"]


def write_node_id(tree, counter=None):
    """Assign sequential 4-digit IDs to all nodes in the tree (DFS).

    Mutates each node to add a "node_id" field ("0001", "0002", ...).
    Returns the counter value after processing.
    """
    if counter is None:
        counter = [0]
    for node in tree:
        counter[0] += 1
        node["node_id"] = f"{counter[0]:04d}"
        if node.get("children"):
            write_node_id(node["children"], counter)
    return counter[0]


def md_string_to_tree(md_text, doc_name):
    """Convert a markdown string to a tree structure.

    Args:
        md_text: Markdown text content.
        doc_name: Name/identifier for this document.

    Returns:
        {"doc_name": str, "structure": [...]} where structure is a nested
        list of nodes with: node_id, level, title, text, children.
    """
    lines = md_text.split("\n")
    nodes = extract_nodes_from_markdown(lines)

    if not nodes:
        # No headings found — wrap entire text as a single root node
        return {
            "doc_name": doc_name,
            "structure": [{
                "node_id": "0001",
                "level": 1,
                "title": doc_name,
                "text": md_text.strip(),
                "children": [],
            }],
        }

    extract_node_text_content(nodes, lines)
    tree = build_tree_from_nodes(nodes)
    write_node_id(tree)

    # Capture any text before the first heading as preamble
    first_line = nodes[0]["line_idx"]
    if first_line > 0:
        preamble = "\n".join(lines[:first_line]).strip()
        if preamble:
            tree[0]["text"] = preamble + "\n\n" + tree[0]["text"] if tree[0]["text"] else preamble

    return {"doc_name": doc_name, "structure": tree}
