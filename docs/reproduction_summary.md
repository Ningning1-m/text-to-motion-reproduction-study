# Reproduction Summary

## 1. Scope

This project uses the official pretrained HumanML3D model from the CVPR 2022
Text-to-Motion implementation. The main reproduced behavior is:

```text
natural-language prompt -> generated 3D human motion -> animation
```

The experiments below are additional analyses performed on the generated
motions. They are not a claim that the original paper reported exactly these
prompt sets or metrics.

## 2. Experiments

### Diversity

Five motions were generated for each prompt. The analysis compares:

- `duration_sec`: number of frames divided by 20 FPS;
- `pose_diversity`: pairwise RMS distance after root-centering;
- `trajectory_diversity`: pairwise RMS distance between initial-point-aligned
  root XZ trajectories.

The same text can produce different lengths, poses, and trajectories because
the model samples from a learned conditional motion distribution.

### Direction

The prompts were:

- `a person walks forward`
- `a person walks backward`
- `a person walks to the left`
- `a person walks to the right`

The results show variation within each direction condition. In visual
inspection, some left-conditioned samples appeared to move right, while one
right-conditioned sample appeared close to straight motion. This should be
described as directional inconsistency or weak semantic control, not as a
formal classification accuracy result.

Coordinate conventions, camera viewpoint, body facing direction, and the
difference between root trajectory and visual appearance must be controlled
before making a definitive left/right error claim.

### Length

The prompts included:

- `a person jumps`
- `a person walks forward`
- `a person walks forward and turns left`
- `a person stands up, walks forward, turns around, and sits down`

The long compositional prompt repeatedly reached 196 frames, or 9.8 seconds.
This suggests that the current sampling or model configuration reaches a
maximum-length boundary for that condition. It should not be interpreted as
proof that the model understands every sub-action equally well.

## 3. Important Data Processing Correction

The generated `.npy` files contain normalized 263-dimensional motion features,
not directly usable XYZ joint coordinates. The correct analysis pipeline is:

```text
[1, T, 263] normalized motion
        -> data * std + mean
        -> recover_from_ric()
        -> [T, 22, 3] joint coordinates
        -> metrics and figures
```

All reported CSV files were generated after applying inverse normalization
before calling `recover_from_ric()`.

## 4. What the Numbers Mean

The project metrics are useful for comparing conditions inside this study, but
they are not directly comparable to the official CVPR 2022 benchmark metrics.

- Larger pose diversity means sampled body poses differ more after removing
  global root translation.
- Larger trajectory diversity means sampled root paths differ more after
  aligning their starting points.
- Larger duration standard deviation means the generated length is less stable.
- A lower root straightness value generally indicates a less direct root path,
  but it does not by itself prove that a motion is semantically correct.

These values should be reported with the prompt set, number of samples, model
checkpoint, and coordinate-processing details.

## 5. Limitations

This study uses five samples per prompt and a small hand-written prompt set.
It does not establish general model accuracy.

The next rigorous extension would be to use a larger balanced prompt set,
define an explicit direction classifier from root trajectories, and compare
the results against annotated references or the official evaluation protocol.
