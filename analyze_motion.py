from pathlib import Path
import csv
import numpy as np
import torch

from scripts.motion_process import recover_from_ric

ROOT = Path("eval_results/t2m/Comp_v6_KLD01/study_v1/joints")

PROMPTS = {
    "C000": "simple",
    "C001": "simple",
    "C002": "simple",
    "C003": "simple",
    "C004": "double",
    "C005": "double",
    "C006": "complex",
    "C007": "complex",
    "C008": "complex",
}

# HumanML3D foot joints used by this repository.
FOOT_JOINTS = [7, 8, 10, 11]


def load_positions(path):
    motion = np.load(path)

    # Shape: (1, frames, 263) -> (frames, 263)
    if motion.ndim == 3:
        motion = motion[0]

    tensor = torch.from_numpy(motion).float().unsqueeze(0)

    # Shape: (1, frames, 22, 3) -> (frames, 22, 3)
    positions = recover_from_ric(tensor, 22)[0].numpy()
    return positions


rows = []

for motion_id, complexity in PROMPTS.items():
    folder = ROOT / motion_id

    for repeat, path in enumerate(sorted(folder.glob("*.npy")), start=1):
        positions = load_positions(path)

        velocity = np.diff(positions, axis=0)
        acceleration = np.diff(velocity, axis=0)

        velocity_mean = float(np.linalg.norm(velocity, axis=-1).mean())
        acceleration_mean = float(np.linalg.norm(acceleration, axis=-1).mean())

        feet = positions[:, FOOT_JOINTS]
        foot_speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1)

        # Estimate contact frames using the lowest quarter of each foot's height.
        foot_height = feet[:-1, :, 1]
        threshold = np.percentile(foot_height, 25, axis=0, keepdims=True)
        contact_mask = foot_height <= threshold

        contact_speeds = foot_speed[contact_mask]
        foot_contact_speed = (
            float(contact_speeds.mean())
            if contact_speeds.size
            else float("nan")
        )

        rows.append({
            "id": motion_id,
            "complexity": complexity,
            "repeat": repeat,
            "file": str(path),
            "frames": positions.shape[0],
            "velocity_mean": round(velocity_mean, 6),
            "acceleration_mean": round(acceleration_mean, 6),
            "foot_contact_speed": round(foot_contact_speed, 6),
            "semantic_score": "",
            "order_correct": "",
            "notes": "",
        })

with open("auto_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"saved {len(rows)} rows to auto_metrics.csv")