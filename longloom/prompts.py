"""Prompt templates and a brace-safe renderer.

Placeholders use ``{name}`` syntax, but rendering is done with literal string
replacement (not ``str.format``), so chunk/merge content containing braces,
JSON, or code never breaks rendering — a common failure mode with
f-string-style prompt templates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = [
    "PromptTemplates",
    "DEFAULT_PROMPTS",
    "VERBATIM_MAP_PROMPT",
    "VERBATIM_MERGE_PROMPT",
    "VERBATIM_REPAIR_PROMPT",
    "render_prompt",
]

# Extraction preset prompts: deterministic no-loss extraction. The map phase
# COPIES fact-bearing lines verbatim instead of summarizing (summarization is
# where facts get dropped); the merge phase deduplicates exact copies only.
VERBATIM_MAP_PROMPT = (
    "Copy VERBATIM every line from the chunk below that contains any concrete "
    "fact: numbers, identifiers (INC-, TKT-, tr-, ERR_, FR-, PO-...), email addresses, "
    "URLs, hostnames, IP addresses, versions, dates, money amounts, "
    "key=value or key: value pairs, percentages, measurements, geographic names "
    "(states, cities, mandis, districts), and organization/person names.\n"
    "If a sentence wraps onto the next line, copy the continuation too so the "
    "fact stays complete.\n"
    "Rules:\n"
    "- Copy lines exactly as written. Do NOT paraphrase, reformat, or compose.\n"
    "- Do NOT drop any fact-bearing line. Do NOT add commentary.\n"
    "- One line of output per line of input (prefix with '- ').\n\n"
    "CHUNK {chunk_index}/{total_chunks}:\n{chunk_content}"
)
VERBATIM_MERGE_PROMPT = (
    "Below are verbatim fact lines extracted from chunks of one document.\n"
    "Combine them into a single list.\n"
    "Rules:\n"
    "- Remove EXACT duplicate lines only (same text after trimming).\n"
    "- Keep every unique line, copied exactly. Do NOT paraphrase, merge, "
    "reformat, or drop any line. Do NOT add commentary.\n\n"
    "CHUNK RESULTS:\n{chunk_results}"
)
VERBATIM_REPAIR_PROMPT = (
    "The list below is too long ({current_tokens} tokens) for the "
    "{max_tokens}-token limit. Remove only entries that are redundant "
    "substrings of other entries (e.g. keep the full line, drop a line that "
    "adds no new identifier/number/email/URL). Never invent or alter values.\n\n"
    "CURRENT RESULT:\n{merged_result}\n\n"
    "Return the trimmed list, every remaining line copied exactly."
)


@dataclass
class PromptTemplates:
    map: str = (
        "You are processing a chunk of a larger document. "
        "Your task is to {task} this chunk.\n\n"
        "CHUNK {chunk_index}/{total_chunks}:\n{chunk_content}\n\n"
        "Provide a concise result for this chunk that can be combined with "
        "other chunks.\nFocus on: {task_focus}"
    )
    merge: str = (
        "You are combining results from multiple document chunks.\n"
        "Original task: {task}\n\n"
        "CHUNK RESULTS:\n{chunk_results}\n\n"
        "Combine these into a single coherent result.\n"
        "Focus on: {task_focus}\n"
        "Remove redundancy while preserving all key information."
    )
    repair: str = (
        "The previous merge result was too long ({current_tokens} tokens) for "
        "the context window ({max_tokens} tokens).\n"
        "Please condense the following result while preserving all essential "
        "information for the task: {task}\n\n"
        "CURRENT RESULT:\n{merged_result}\n\n"
        "Provide a condensed version that fits within {max_tokens} tokens.\n"
        "Focus on: {task_focus}"
    )
    conflicts: str = (
        "Review the report below and identify CONTRADICTORY statements: the "
        "same subject carrying different values, dates, quantities or claims.\n"
        "Output one line per contradiction, formatted exactly:\n"
        "CONFLICT: <statement A> || <statement B>\n"
        "If there are no contradictions, output exactly: NONE\n"
        "Do not invent conflicts. Do not add commentary.\n\n"
        "REPORT:\n{final_result}"
    )

    @classmethod
    def for_extraction(cls) -> "PromptTemplates":
        """No-loss extraction preset: verbatim map + keep-every-line merge."""
        return cls(
            map=VERBATIM_MAP_PROMPT,
            merge=VERBATIM_MERGE_PROMPT,
            repair=VERBATIM_REPAIR_PROMPT,
        )


DEFAULT_PROMPTS = PromptTemplates()


def render_prompt(template: str, variables: Dict[str, object]) -> str:
    """Replace ``{name}`` placeholders literally; content braces are safe."""
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", str(value))
    return out