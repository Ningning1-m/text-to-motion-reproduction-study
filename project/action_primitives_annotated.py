"""Rule-based text to action-primitive conversion with line-by-line comments."""

import json  # Serialize Python dictionaries as JSON.
import re  # Match action phrases with regular expressions.
from pathlib import Path  # Build platform-independent filesystem paths.


# Resolve the repository root from this file's location.
ROOT = Path(__file__).resolve().parents[1]

# Input file containing one natural-language prompt per line.
PROMPTS_FILE = ROOT / "project" / "prompts_primitives.txt"

# Output file containing the parsed action sequences.
OUTPUT_FILE = ROOT / "project" / "results" / "action_primitives.json"

# Each tuple maps a phrase pattern to one canonical primitive name.
PATTERNS = [
    (r"\bwalk(?:s)?\s+forward\b", "WALK_FORWARD"),
    (r"\bwalk(?:s)?\s+backward\b", "WALK_BACKWARD"),
    (r"\bwalk(?:s)?\s+to\s+the\s+left\b", "WALK_LEFT"),
    (r"\bwalk(?:s)?\s+to\s+the\s+right\b", "WALK_RIGHT"),
    (r"\bturn(?:s)?\s+(?:to\s+the\s+)?left\b", "TURN_LEFT"),
    (r"\bturn(?:s)?\s+(?:to\s+the\s+)?right\b", "TURN_RIGHT"),
    (r"\bturn(?:s)?\s+around\b", "TURN_AROUND"),
    (r"\braise(?:s)?\s+(?:the\s+)?right\s+hand\b", "RAISE_RIGHT_HAND"),
    (r"\braise(?:s)?\s+(?:the\s+)?left\s+hand\b", "RAISE_LEFT_HAND"),
    (r"\braise(?:s)?\s+both\s+arms\b", "RAISE_BOTH_ARMS"),
    (r"\bstand(?:s)?\s+up\b", "STAND_UP"),
    (r"\bsit(?:s)?\s+down\b", "SIT_DOWN"),
    (r"\bjump(?:s)?\b", "JUMP"),
]


def parse_prompt(prompt):
    """Return recognized action primitives in their original text order."""
    actions = []  # Store matches found in the prompt.

    for pattern, action_type in PATTERNS:
        # Search for every occurrence of the current phrase pattern.
        for match in re.finditer(pattern, prompt.lower()):
            actions.append(
                {
                    "type": action_type,  # Canonical machine-readable action.
                    "text_span": match.group(0),  # Phrase matched in the prompt.
                    "position": match.start(),  # Character offset for sorting.
                }
            )

    # Regex patterns are checked by category, so restore sentence order.
    actions.sort(key=lambda item: item["position"])

    # Position is an internal implementation detail, not output data.
    for action in actions:
        action.pop("position")

    return actions  # Return the ordered list of primitives.


def main():
    """Read prompts, parse them, and write one JSON record per prompt."""
    if not PROMPTS_FILE.exists():
        # Fail early with a useful path if the input file is missing.
        raise FileNotFoundError(PROMPTS_FILE)

    # Read non-empty lines and discard surrounding whitespace.
    prompts = [
        line.strip()
        for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records = []  # Store the final JSON records.

    for index, prompt in enumerate(prompts):
        # Convert zero-based indices into IDs such as C000 and C001.
        records.append(
            {
                "id": f"C{index:03d}",
                "prompt": prompt,
                "actions": parse_prompt(prompt),
            }
        )

    # Create the output directory if it does not already exist.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write human-readable UTF-8 JSON.
    OUTPUT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Report the number and destination of generated records.
    print(f"saved {len(records)} records to {OUTPUT_FILE}")


# Run main only when this file is executed directly.
if __name__ == "__main__":
    main()
