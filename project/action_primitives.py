import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_FILE = ROOT / "project" / "prompts_primitives.txt"
OUTPUT_FILE = ROOT / "project" / "results" / "action_primitives.json"

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
    actions = []

    for pattern, action_type in PATTERNS:
        for match in re.finditer(pattern, prompt.lower()):
            actions.append(
                {
                    "type": action_type,
                    "text_span": match.group(0),
                    "position": match.start(),
                }
            )

    actions.sort(key=lambda item: item["position"])

    for action in actions:
        action.pop("position")

    return actions


def main():
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(PROMPTS_FILE)

    prompts = [
        line.strip()
        for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records = []

    for index, prompt in enumerate(prompts):
        records.append(
            {
                "id": f"C{index:03d}",
                "prompt": prompt,
                "actions": parse_prompt(prompt),
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"saved {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()