r"""
Analyze diversity of Text-to-Motion generated samples.

Expected project layout:
E:\motion-research\text-to-motion\
├── project\
│   ├── prompts_diversity.txt
│   └── analyze_diversity.py
└── eval_results\
    └── t2m\
        └── Comp_v6_KLD01\
            └── exp_diversity_v1\
                └── joints\
                    ├── C000\
                    ├── C001\
                    ├── C002\
                    └── C003\

This script:
1. Reads 263-D motion representations from the official generator.
2. Recovers them to 22-joint XYZ using the official recover_from_ric().
3. Computes per-sample motion length, duration, root trajectory statistics,
   mean joint speed, and mean joint acceleration.
4. Computes pairwise diversity within each text:
   - pose_diversity: root-centered joint-position RMS distance
   - trajectory_diversity: aligned root XZ trajectory RMS distance
5. Writes CSV files and PNG figures.

Important:
The diversity metrics in this script are exploratory metrics for this project.
They are NOT the official CVPR 2022 benchmark metrics (FID, R-Precision, etc.).
"""

from __future__ import annotations

import csv
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.motion_process import recover_from_ric


# -----------------------------
# Configuration
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "exp_direction_v1"
JOINTS_DIR = (
    ROOT
    / "eval_results"
    / "t2m"
    / "Comp_v6_KLD01"
    / EXPERIMENT_NAME
    / "joints"
)

PROMPTS_FILE = ROOT / "project" / "prompts_direction.txt"

RESULTS_DIR = ROOT / "project" / "results" / EXPERIMENT_NAME
FIGURES_DIR = ROOT / "project" / "figures" / EXPERIMENT_NAME

FPS = 20.0
NUM_JOINTS = 22
RESAMPLE_LENGTH = 60
META_DIR = ROOT / "checkpoints" / "t2m" / "Comp_v6_KLD01" / "meta"
MEAN_PATH = META_DIR / "mean.npy"
STD_PATH = META_DIR / "std.npy"


# -----------------------------
# Utility functions
# -----------------------------
def read_prompts(path: Path) -> dict[str, str]:
    """Read non-empty prompt lines and map C000 -> text, C001 -> text, ..."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts: dict[str, str] = {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]

    for i, text in enumerate(lines):
        prompts[f"C{i:03d}"] = text

    return prompts


def parse_sample_id(filename: str) -> int:
    """
    Parse sample id from:
    gen_motion_00_L072.npy -> 0
    gen_motion_04_L196.npy -> 4
    """
    match = re.search(r"gen_motion_(\d+)_L\d+\.npy$", filename)
    if not match:
        raise ValueError(f"Unexpected motion filename: {filename}")
    return int(match.group(1))


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Linearly resample a temporal sequence.

    sequence:
        [T, ...]
    return:
        [target_length, ...]
    """
    if sequence.ndim < 2:
        raise ValueError(f"Sequence must have at least 2 dimensions, got {sequence.shape}")

    src_length = sequence.shape[0]

    if src_length == target_length:
        return sequence.copy()

    if src_length < 2:
        return np.repeat(sequence, target_length, axis=0)

    old_t = np.linspace(0.0, 1.0, src_length)
    new_t = np.linspace(0.0, 1.0, target_length)

    flat = sequence.reshape(src_length, -1)
    out = np.empty((target_length, flat.shape[1]), dtype=np.float32)

    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(new_t, old_t, flat[:, dim])

    return out.reshape((target_length,) + sequence.shape[1:])


def recover_motion_to_joints(
    motion_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Load [1, T, 263] motion representation and recover it to [T, 22, 3].
    """
    data = np.load(motion_path)

    if data.ndim != 3:
        raise ValueError(f"{motion_path} expected 3D array [1,T,263], got {data.shape}")

    if data.shape[0] != 1:
        raise ValueError(f"{motion_path} expected batch dimension 1, got {data.shape}")

    if data.shape[2] != 263:
        raise ValueError(f"{motion_path} expected last dimension 263, got {data.shape}")

    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(
            f"Expected mean/std shape (263,), got {mean.shape} and {std.shape}"
        )

    # The generator saves normalized motion features. Undo the normalization
    # before applying the repository's inverse-RIC recovery.
    data = data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    tensor = torch.from_numpy(data).float()

    with torch.no_grad():
        joints = recover_from_ric(tensor, NUM_JOINTS)

    joints = joints.detach().cpu().numpy()

    # Expected [1, T, 22, 3] -> [T, 22, 3]
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]

    if joints.ndim != 3 or joints.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"{motion_path} recovery produced unexpected shape {joints.shape}; "
            f"expected [T,{NUM_JOINTS},3]"
        )

    return joints.astype(np.float32)


def compute_sample_metrics(joints: np.ndarray) -> dict[str, float]:
    """
    Compute exploratory motion statistics from recovered joint positions.
    """
    frames = joints.shape[0]
    duration = frames / FPS

    # Root joint is index 0 in the HumanML3D 22-joint representation.
    root = joints[:, 0, :]  # [T, 3]
    root_xz = root[:, [0, 2]]

    # Root path and net displacement.
    if frames >= 2:
        root_steps = np.diff(root_xz, axis=0)
        root_step_lengths = np.linalg.norm(root_steps, axis=1)
        root_path_length = float(root_step_lengths.sum())
        root_displacement = float(np.linalg.norm(root_xz[-1] - root_xz[0]))

        # All-joint velocity / acceleration.
        velocity = np.diff(joints, axis=0) * FPS
        speed = np.linalg.norm(velocity, axis=-1)
        mean_joint_speed = float(speed.mean())

        if velocity.shape[0] >= 2:
            acceleration = np.diff(velocity, axis=0) * FPS
            acceleration_norm = np.linalg.norm(acceleration, axis=-1)
            mean_joint_acceleration = float(acceleration_norm.mean())
        else:
            mean_joint_acceleration = float("nan")
    else:
        root_path_length = 0.0
        root_displacement = 0.0
        mean_joint_speed = 0.0
        mean_joint_acceleration = float("nan")

    straightness = (
        root_displacement / root_path_length
        if root_path_length > 1e-8
        else 0.0
    )

    return {
        "frames": int(frames),
        "duration_sec": duration,
        "root_path_length_xz": root_path_length,
        "root_displacement_xz": root_displacement,
        "root_straightness": straightness,
        "mean_joint_speed": mean_joint_speed,
        "mean_joint_acceleration": mean_joint_acceleration,
    }


def root_center(joints: np.ndarray) -> np.ndarray:
    """Remove global root translation so we can compare body pose."""
    return joints - joints[:, 0:1, :]


def normalized_root_trajectory(joints: np.ndarray) -> np.ndarray:
    """
    Return root XZ trajectory aligned to the initial point and resampled.
    Shape: [RESAMPLE_LENGTH, 2]
    """
    root_xz = joints[:, 0, [0, 2]]
    root_xz = root_xz - root_xz[0]
    return resample_sequence(root_xz, RESAMPLE_LENGTH)


def normalized_pose(joints: np.ndarray) -> np.ndarray:
    """
    Root-centered pose sequence resampled to a fixed temporal length.
    Shape: [RESAMPLE_LENGTH, 22, 3]
    """
    centered = root_center(joints)
    return resample_sequence(centered, RESAMPLE_LENGTH)


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square Euclidean distance over all represented elements."""
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


# -----------------------------
# Main analysis
# -----------------------------
def main() -> None:
    if not MEAN_PATH.exists() or not STD_PATH.exists():
        raise FileNotFoundError(
            f"Missing normalization files: {MEAN_PATH} and/or {STD_PATH}"
        )

    mean = np.load(MEAN_PATH).astype(np.float32)
    std = np.load(STD_PATH).astype(np.float32)
    print(f"Mean path  : {MEAN_PATH}")
    print(f"Std path   : {STD_PATH}")
    print(f"Mean shape : {mean.shape}")
    print(f"Std shape  : {std.shape}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(PROMPTS_FILE)

    if not JOINTS_DIR.exists():
        raise FileNotFoundError(f"Joints directory not found: {JOINTS_DIR}")

    # Store recovered motions in memory for pairwise comparisons.
    motions: dict[str, dict[int, dict]] = {}

    sample_rows = []

    for text_id, prompt in prompts.items():
        text_dir = JOINTS_DIR / text_id

        if not text_dir.exists():
            raise FileNotFoundError(f"Missing directory for {text_id}: {text_dir}")

        files = sorted(text_dir.glob("*.npy"), key=lambda p: parse_sample_id(p.name))

        if not files:
            raise FileNotFoundError(f"No .npy files found in {text_dir}")

        motions[text_id] = {}

        for motion_path in files:
            sample_id = parse_sample_id(motion_path.name)
            joints = recover_motion_to_joints(motion_path, mean, std)
            metrics = compute_sample_metrics(joints)

            motions[text_id][sample_id] = {
                "path": motion_path,
                "joints": joints,
                "pose": normalized_pose(joints),
                "trajectory": normalized_root_trajectory(joints),
            }

            row = {
                "text_id": text_id,
                "sample_id": sample_id,
                "prompt": prompt,
                "file": str(motion_path.relative_to(ROOT)),
                **metrics,
            }
            sample_rows.append(row)

    # -------------------------
    # Save per-sample metrics
    # -------------------------
    sample_csv = RESULTS_DIR / "sample_metrics.csv"
    sample_fields = [
        "text_id",
        "sample_id",
        "prompt",
        "file",
        "frames",
        "duration_sec",
        "root_path_length_xz",
        "root_displacement_xz",
        "root_straightness",
        "mean_joint_speed",
        "mean_joint_acceleration",
    ]

    with sample_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    # -------------------------
    # Pairwise diversity
    # -------------------------
    pairwise_rows = []
    summary_rows = []

    for text_id, samples in motions.items():
        sample_ids = sorted(samples.keys())

        pose_distances = []
        trajectory_distances = []

        for sample_a, sample_b in combinations(sample_ids, 2):
            pose_d = rms_distance(
                samples[sample_a]["pose"],
                samples[sample_b]["pose"],
            )
            traj_d = rms_distance(
                samples[sample_a]["trajectory"],
                samples[sample_b]["trajectory"],
            )

            pose_distances.append(pose_d)
            trajectory_distances.append(traj_d)

            pairwise_rows.append(
                {
                    "text_id": text_id,
                    "sample_a": sample_a,
                    "sample_b": sample_b,
                    "pose_diversity": pose_d,
                    "trajectory_diversity": traj_d,
                }
            )

        durations = [
            row["duration_sec"]
            for row in sample_rows
            if row["text_id"] == text_id
        ]

        summary_rows.append(
            {
                "text_id": text_id,
                "prompt": prompts[text_id],
                "num_samples": len(sample_ids),
                "duration_mean_sec": float(np.mean(durations)),
                "duration_std_sec": float(np.std(durations)),
                "duration_min_sec": float(np.min(durations)),
                "duration_max_sec": float(np.max(durations)),
                "mean_pairwise_pose_diversity": float(np.mean(pose_distances))
                if pose_distances
                else 0.0,
                "std_pairwise_pose_diversity": float(np.std(pose_distances))
                if pose_distances
                else 0.0,
                "mean_pairwise_trajectory_diversity": float(
                    np.mean(trajectory_distances)
                )
                if trajectory_distances
                else 0.0,
                "std_pairwise_trajectory_diversity": float(
                    np.std(trajectory_distances)
                )
                if trajectory_distances
                else 0.0,
            }
        )

    pairwise_csv = RESULTS_DIR / "pairwise_diversity.csv"
    pairwise_fields = [
        "text_id",
        "sample_a",
        "sample_b",
        "pose_diversity",
        "trajectory_diversity",
    ]

    with pairwise_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    summary_csv = RESULTS_DIR / "diversity_summary.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # -------------------------
    # Figure 1: duration distribution
    # -------------------------
    labels = [row["text_id"] for row in summary_rows]
    duration_groups = [
        [
            row["duration_sec"]
            for row in sample_rows
            if row["text_id"] == text_id
        ]
        for text_id in labels
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(duration_groups, labels=labels)
    plt.ylabel("Duration (seconds)")
    plt.xlabel("Text condition")
    plt.title("Generated Motion Duration by Text")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duration_distribution.png", dpi=200)
    plt.close()

    # -------------------------
    # Figure 2: pose diversity
    # -------------------------
    pose_values = [
        row["mean_pairwise_pose_diversity"] for row in summary_rows
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, pose_values)
    plt.ylabel("Mean pairwise pose diversity")
    plt.xlabel("Text condition")
    plt.title("Pose Diversity Within Each Text")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pose_diversity.png", dpi=200)
    plt.close()

    # -------------------------
    # Figure 3: trajectory diversity
    # -------------------------
    traj_values = [
        row["mean_pairwise_trajectory_diversity"] for row in summary_rows
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, traj_values)
    plt.ylabel("Mean pairwise trajectory diversity")
    plt.xlabel("Text condition")
    plt.title("Root Trajectory Diversity Within Each Text")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trajectory_diversity.png", dpi=200)
    plt.close()

    # -------------------------
    # Figure 4+: one root trajectory plot per text
    # -------------------------
    for text_id, samples in motions.items():
        plt.figure(figsize=(7, 6))

        for sample_id in sorted(samples.keys()):
            traj = samples[sample_id]["trajectory"]
            plt.plot(
                traj[:, 0],
                traj[:, 1],
                marker=None,
                label=f"sample {sample_id}",
            )

        plt.xlabel("Root X")
        plt.ylabel("Root Z")
        plt.title(f"Root XZ Trajectories - {text_id}: {prompts[text_id]}")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{text_id}_root_trajectory.png",
            dpi=200,
        )
        plt.close()

    # -------------------------
    # Console summary
    # -------------------------
    print("=" * 72)
    print("Diversity analysis completed.")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Input      : {PROMPTS_FILE}")
    print(f"Joints dir : {JOINTS_DIR}")
    print(f"Results    : {RESULTS_DIR}")
    print(f"Figures    : {FIGURES_DIR}")
    print("=" * 72)

    for row in summary_rows:
        print(
            f"{row['text_id']} | {row['prompt']}\n"
            f"  samples={row['num_samples']}, "
            f"duration={row['duration_mean_sec']:.2f}±{row['duration_std_sec']:.2f}s, "
            f"pose_div={row['mean_pairwise_pose_diversity']:.6f}, "
            f"traj_div={row['mean_pairwise_trajectory_diversity']:.6f}"
        )


if __name__ == "__main__":
    main()
