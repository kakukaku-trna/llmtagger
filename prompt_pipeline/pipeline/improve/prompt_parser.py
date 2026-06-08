"""Prompt structure parser.

Parses a Markdown-formatted detection prompt into a structured object:
    - sections (## headers with body text)
    - rules (numbered list items that act as judgment criteria)
    - output_format (the JSON output specification)

Used by topk_modifier to precisely locate which rules to modify rather
than sending the full prompt text and hoping the LLM finds the right place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────

@dataclass
class PromptRule:
    """A single numbered judgment rule extracted from the prompt."""
    index: int          # 1-based position in the rules list
    text: str           # Full rule text (may be multi-line)
    section: str        # Parent section name (## heading)
    keywords: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.index}. {self.text}"


@dataclass
class PromptSection:
    """A ## section within the prompt."""
    title: str
    body: str
    rules: List[PromptRule] = field(default_factory=list)


@dataclass
class ParsedPrompt:
    """Structured representation of a detection prompt."""
    raw: str
    sections: List[PromptSection] = field(default_factory=list)
    output_format: str = ""
    # All rules across all sections, in order of appearance
    all_rules: List[PromptRule] = field(default_factory=list)

    def get_section(self, title_pattern: str) -> Optional[PromptSection]:
        for s in self.sections:
            if re.search(title_pattern, s.title, re.IGNORECASE):
                return s
        return None

    def top_k_rules(self, k: int) -> List[PromptRule]:
        """Return first K rules (default: most structurally central ones)."""
        return self.all_rules[:k]

    def rule_by_index(self, index: int) -> Optional[PromptRule]:
        for r in self.all_rules:
            if r.index == index:
                return r
        return None

    def replace_rule(self, index: int, new_text: str) -> str:
        """Return a new prompt string with rule `index` replaced by new_text."""
        prompt = self.raw
        for rule in self.all_rules:
            if rule.index == index:
                old_line = f"{rule.index}. {rule.text}"
                new_line = f"{rule.index}. {new_text}"
                prompt = prompt.replace(old_line, new_line, 1)
                break
        return prompt

    def summary(self) -> str:
        lines = [f"Sections: {len(self.sections)}  Rules: {len(self.all_rules)}"]
        for r in self.all_rules:
            lines.append(f"  [{r.section}] {r.index}. {r.text[:60]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────

def parse_prompt(text: str) -> ParsedPrompt:
    """Parse a Markdown prompt into a ParsedPrompt structure."""
    parsed = ParsedPrompt(raw=text)
    sections = _split_sections(text)
    rule_counter = 1

    for title, body in sections:
        rules = _extract_rules(body, rule_counter)
        rule_counter += len(rules)
        section = PromptSection(title=title, body=body, rules=rules)
        parsed.sections.append(section)
        parsed.all_rules.extend(rules)

        # Detect output format section
        if re.search(r"输出|output|format|格式", title, re.IGNORECASE):
            parsed.output_format = body.strip()

    # If no sections, treat the whole text as one unnamed section
    if not parsed.sections:
        rules = _extract_rules(text, 1)
        section = PromptSection(title="main", body=text, rules=rules)
        parsed.sections.append(section)
        parsed.all_rules = rules

    return parsed


def load_and_parse(prompt_path: str) -> ParsedPrompt:
    """Load a prompt .md file from disk and parse it."""
    from pathlib import Path
    text = Path(prompt_path).read_text(encoding="utf-8")
    return parse_prompt(text)


# ─────────────────────────────────────────────────────────────
# Diff helper: show what changed between two prompt versions
# ─────────────────────────────────────────────────────────────

def diff_rules(old: ParsedPrompt, new: ParsedPrompt) -> List[Tuple[int, str, str]]:
    """Return list of (rule_index, old_text, new_text) for changed rules."""
    old_map = {r.index: r.text for r in old.all_rules}
    new_map = {r.index: r.text for r in new.all_rules}
    diffs = []
    for idx in sorted(set(old_map) | set(new_map)):
        a = old_map.get(idx, "")
        b = new_map.get(idx, "")
        if a != b:
            diffs.append((idx, a, b))
    return diffs


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split prompt by ## headings. Returns list of (title, body) tuples."""
    pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("main", text)]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))
    return sections


def _extract_rules(text: str, start_index: int) -> List[PromptRule]:
    """Extract numbered list items from a text block."""
    # Match: "1. ", "1、", "（1）" style numberings
    pattern = re.compile(
        r"(?:^|\n)\s*(?:(\d+)[\.、。）\)]\s*|（(\d+)）\s*)(.+?)(?=\n\s*(?:\d+[\.、。）\)]|（\d+）)|\Z)",
        re.DOTALL,
    )
    rules = []
    for m in pattern.finditer(text):
        num_str = m.group(1) or m.group(2) or ""
        rule_text = m.group(3).strip()
        rule_text = re.sub(r"\s+", " ", rule_text)
        if not rule_text:
            continue
        idx = int(num_str) if num_str.isdigit() else start_index + len(rules)
        section = _infer_section(text, m.start())
        rules.append(PromptRule(
            index=idx,
            text=rule_text,
            section=section,
            keywords=_extract_keywords(rule_text),
        ))
    return rules


def _extract_keywords(text: str) -> List[str]:
    """Extract Chinese/English content words as naive keywords."""
    words = re.findall(r"[一-鿿]{2,}|[A-Za-z]{4,}", text)
    return list(dict.fromkeys(words))[:6]


def _infer_section(text: str, pos: int) -> str:
    """Find the closest preceding ## heading for a position in the text."""
    before = text[:pos]
    m = re.findall(r"#{1,3}\s+(.+)", before)
    return m[-1].strip() if m else "main"
