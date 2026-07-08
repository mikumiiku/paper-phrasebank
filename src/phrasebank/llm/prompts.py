"""Prompt templates for the LLM extraction stage."""
from __future__ import annotations

from phrasebank import FUNCTION_CATEGORIES

SYSTEM_METADATA = """You are a metadata extraction assistant. Given the first page or first ~2000 characters of a paper, extract ONLY:
- paper_title: the full title of the paper
- paper_authors: a comma-separated list of authors
- paper_year: the publication year as a 4-digit string

Return STRICT JSON only. No markdown fences, no commentary. Object shape:
{"paper_title": "...", "paper_authors": "...", "paper_year": "..."}
If a field is unknown, use an empty string."""

USER_METADATA = """Extract metadata from the following paper text:

<<<TEXT>>>
{text}
<<<END>>>"""

SENTENCE_SYSTEM = f"""You are an academic writing assistant. Given a chunk of paper text, extract reusable template sentences suitable for an academic phrase bank.

For each candidate sentence, output an object with these fields:
- "sentence": the ORIGINAL sentence from the text, verbatim, full sentence with terminating punctuation. Do not truncate.
- "language": "en" if the sentence is primarily English, "zh" if primarily Chinese.
- "function_category": exactly one of these six categories: {", ".join(f"'{c}'" for c in FUNCTION_CATEGORIES)}
- "tags" (string[]): up to 5 short topical keywords (single words or short phrases).
- "usage_note" (string): one short sentence describing when/how this sentence is useful.

Return a STRICT JSON ARRAY ONLY (e.g. [{{...}}, {{...}}]). No markdown ``` fences, no commentary before or after, no explanations.
If no reusable sentence is found, return an empty array []. Do not invent sentences not present in the text."""

SENTENCE_USER = """Extract candidate template sentences from the following text. The primary language of this text is {language}.

<<<TEXT>>>
{chunk}
<<<END>>>"""
