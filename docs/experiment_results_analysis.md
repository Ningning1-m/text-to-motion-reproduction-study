# Analysis of Custom Text-to-Motion Experiments

This report analyzes 12 experimental conditions across three custom experiments, with five generated motion samples for each condition. The analysis combines summary CSV files, root-trajectory plots, duration/diversity charts, and frame-by-frame video inspection. It focuses on the pretrained Text-to-Motion model's observed behavior in motion diversity, directional semantics, and the execution order of compound actions.

> **Scope of the experiments.** This project uses the official code and pretrained weights from the CVPR 2022 Text-to-Motion work to run text-conditioned 3D human-motion generation, followed by a small-scale exploratory analysis. The pose and root-trajectory distances reported here are custom metrics developed for this project; they are not the paper's official evaluation results, such as FID, R-Precision, Matching Score, Diversity, or Multimodality. Five samples were generated for each prompt. The report describes differences observed in these samples, identifies specific semantic deviations, and proposes targeted improvements. The small sample size does not support estimates of the model's general accuracy.

## 1. Analysis Pipeline and Metric Definitions

### 1.1 Analysis pipeline

The three experiments follow the same main processing pipeline:

```text
Text prompt
  -> Generate motion with the official pretrained model
  -> Load normalized motion features with shape [1, T, 263]
  -> Inverse normalization: motion * std + mean
  -> Call the official recover_from_ric()
  -> Obtain 3D joint coordinates with shape [T, 22, 3]
  -> Compute per-sample motion metrics and pairwise distances
  -> Export CSV files and PNG figures
```

`analyze_diversity.py`, `analyze_direction.py`, and `analyze_length.py` reuse similar analysis logic and distinguish the experiments mainly through their prompts and output directories. The motion features must be inverse-normalized before calling `recover_from_ric()`. Treating the normalized 263-dimensional feature vectors directly as XYZ joint coordinates would distort the analysis results.

### 1.2 Metric definitions

| Metric | How it is calculated | Interpretation and limitations |
|---|---|---|
| `duration_mean_sec` | Mean number of frames divided by 20 FPS | Describes sequence duration; it does not establish that the action is complete or semantically correct. |
| `duration_std_sec` | Population standard deviation of sample durations for one prompt (`np.std`, default `ddof=0`) | A larger value means greater duration variation among these five samples. |
| `mean_pairwise_pose_diversity` | Mean pairwise RMS distance between root-centered pose sequences, each resampled to 60 frames | A larger value means greater pose variation; it does not necessarily mean better quality. |
| `mean_pairwise_trajectory_diversity` | Mean pairwise RMS distance between root XZ trajectories, aligned at their starting points and resampled to 60 frames | A larger value means greater trajectory variation across samples; it is not a directional-accuracy score. |
| `root_path_length_xz` | Sum of frame-to-frame root displacement lengths in the XZ plane | Describes the cumulative length of the root trajectory. |
| `root_displacement_xz` | Straight-line XZ distance between the root's start and end positions | Not the same as cumulative path length. |
| `root_straightness` | Endpoint displacement divided by cumulative path length | Describes how directly the root moves; it does not determine whether the direction matches the text prompt. |

**Two types of trajectory measures must be distinguished.** `root_straightness` describes the straightness of a single trajectory, whereas `mean_pairwise_trajectory_diversity` describes the average distance between trajectories generated from the same prompt. Every sample in a group can follow an approximately straight path while the group still has high trajectory-diversity values because the samples differ in direction, lateral offset, or travel distance.

Video IDs `00`–`04` correspond to the five generated samples for each prompt and match the generation IDs in the output filenames. The analyses below combine video observations with aggregate metrics: videos are used to assess action semantics and execution order, while CSV files and trajectory plots describe duration, pose, and root-motion differences. These two forms of evidence complement one another but are not interchangeable.

---

## 2. Experiment 1: Motion Diversity

### 2.1 Setup

Five motion sequences were generated for each text prompt. The experiment compares variation in duration, body pose, and root trajectory across samples generated from the same prompt.

Prompts:

- `C000`: `a person walks forward`
- `C001`: `a person jumps`
- `C002`: `a person sits down`
- `C003`: `a person raises the right hand`

### 2.2 Quantitative results

Source: [diversity_summary.csv](../project/results/exp_diversity_v1/diversity_summary.csv).

| ID / Prompt | Samples | Mean duration ± SD (s) | Duration range (s) | Mean pose distance | Mean trajectory distance |
|---|---:|---:|---:|---:|---:|
| C000 — Walk forward | 5 | 5.80 ± 2.501 | 2.8–9.8 | 0.1402 | 0.7565 |
| C001 — Jump | 5 | 3.44 ± 0.408 | 3.0–4.0 | 0.1489 | 0.1542 |
| C002 — Sit down | 5 | 7.36 ± 3.123 | 2.0–9.8 | 0.2000 | 0.2180 |
| C003 — Raise the right hand | 5 | 2.72 ± 0.776 | 2.0–4.2 | 0.0900 | 0.0051 |

### 2.3 Video observations and figure analysis

#### C000: Walk forward

**Video observations.** All five videos show forward walking. The samples differ in their step direction, foot placement, and body pose, so the local visual orientation of the motion is not identical across samples. However, the overall displacement is forward in all five videos, and the root trajectories are approximately straight.

**Metrics and trajectory plot.** The mean trajectory distance is `0.7565`, the highest among the four conditions in this experiment; the mean pose distance is `0.1402`. The root-trajectory plot also shows differences in forward travel distance and lateral offset, with one sample moving noticeably farther than the others.

**Interpretation.** Although this condition has the largest mean trajectory distance among the four action types, this does not mean that the individual trajectories are curved. The videos and trajectory plot show that all five samples move forward along approximately straight paths. Differences in stepping, pose, forward distance, and lateral position can still produce a relatively large pairwise trajectory distance. Foot/body orientation should be distinguished from root translation: the root trajectory describes overall planar displacement and does not directly represent the direction in which the feet are stepping.

#### C001: Jump

**Video observations.** All five videos perform the corresponding jumping action.

**Metrics and trajectory plot.** The mean duration is `3.44 s`, with a duration standard deviation of approximately `0.408 s`, making the durations relatively consistent among the four conditions. The mean pose distance is `0.1489`, and the mean trajectory distance is `0.1542`. Overall root movement is limited, although the poses are not identical across samples.

**Interpretation.** All five samples perform a jump, and their durations are relatively consistent (SD approximately `0.408 s`). The mean pose distance of `0.1489` indicates that the body poses and motion sequences still vary across samples. This metric describes sample variation only; it does not directly assess how natural each jump is.

#### C002: Sit down

**Video observations.** Sample `03` does not complete the sitting action. The video shows the knees bending and the body moving downward, but the motion does not reach a clear final sitting pose. It is therefore classified as an incomplete execution of the target action.

**Metrics and trajectory plot.** The mean duration is `7.36 s`, with a standard deviation of approximately `3.123 s`, the largest duration variation among the four conditions. The shortest sample is `2.0 s` and the longest is `9.8 s`. The mean pose distance is `0.2000`, also the highest among the four conditions; the mean trajectory distance is `0.2180`.

**Interpretation.** The sit-down condition has both the largest duration variation and the highest mean pose distance. Video `03` also fails to reach the final sitting pose. Together, these observations indicate considerable variation in both duration and execution/semantic completion under this prompt. The mean pose distance is an aggregate statistic computed from all pairwise comparisons among the five samples. It is consistent with the observed failure in video `03`, but the group-level mean cannot be attributed entirely to that single sample.

#### C003: Raise the right hand

**Video observations.** All five videos complete the action of raising the right hand.

**Metrics and trajectory plot.** This condition has the shortest mean duration, `2.72 s`. Its mean pose distance is `0.0900`, the lowest among the four conditions, and its mean trajectory distance is only `0.0051`, far below the values for the other conditions. The root shows almost no relative translation variation across samples.

**Interpretation.** All five videos perform the target action, and the mean trajectory distance of only `0.0051` indicates almost no difference in root translation between samples. This is consistent with a hand-raising action that mainly involves the upper limbs and does not require substantial root movement. The mean pose distance of `0.0900` nevertheless indicates that the full-body poses are not identical. Root-level metrics describe overall translation; they cannot quantify hand-raising height or timing. To compare those details, the trajectories of joints such as the wrists and elbows should be analyzed directly.

### 2.4 Summary of the diversity experiment

1. The four action conditions show different generation patterns. Walking forward has the largest pairwise trajectory distance; sitting down has the greatest duration variation and the highest mean pose distance; raising the right hand has the most stable root trajectory.
2. Video inspection provides semantic-completion information that cannot be recovered from the aggregate values alone. All five jump samples and all five right-hand-raising samples complete their target actions. In contrast, sit-down sample `03` does not reach a clear sitting pose, although knee bending and downward body motion are visible.
3. The experiment shows that repeated sampling from the same prompt can produce variation in duration, pose, and trajectory. However, greater diversity is not sufficient evidence of higher generation quality. Action completion and motion naturalness must also be considered.

Figures:

- Summary charts: [Duration distribution](../project/figures/exp_diversity_v1/duration_distribution.png) · [Pose diversity](../project/figures/exp_diversity_v1/pose_diversity.png) · [Trajectory diversity](../project/figures/exp_diversity_v1/trajectory_diversity.png)
- Root trajectories: [C000 — Walk forward](../project/figures/exp_diversity_v1/C000_root_trajectory.png) · [C001 — Jump](../project/figures/exp_diversity_v1/C001_root_trajectory.png) · [C002 — Sit down](../project/figures/exp_diversity_v1/C002_root_trajectory.png) · [C003 — Raise the right hand](../project/figures/exp_diversity_v1/C003_root_trajectory.png)

---

## 3. Experiment 2: Directional Semantics

### 3.1 Setup

The direction experiment uses forward, backward, left, and right prompts, with five samples generated for each prompt. The left and right prompts are `walks to the left` and `walks to the right`. During video inspection, the most salient differences in these samples appear as left turns, right turns, or changes in turning direction. This report records the visible directional behavior and analyzes it separately from the root trajectory in the XZ plane: body-orientation rotation, stepping direction, and lateral root translation are related but distinct aspects of motion.

### 3.2 Quantitative results

Source: [diversity_summary.csv](../project/results/exp_direction_v1/diversity_summary.csv).

| ID / Prompt | Samples | Mean duration ± SD (s) | Duration range (s) | Mean pose distance | Mean trajectory distance |
|---|---:|---:|---:|---:|---:|
| C000 — Walk forward | 5 | 4.68 ± 1.330 | 3.2–7.0 | 0.1403 | 0.6620 |
| C001 — Walk backward | 5 | 5.60 ± 0.769 | 4.4–6.8 | 0.1080 | 0.4598 |
| C002 — Walk left (videos show turning left/right) | 5 | 5.08 ± 0.574 | 4.2–5.8 | 0.1751 | 0.9691 |
| C003 — Walk right (videos show turning left/right) | 5 | 5.88 ± 1.874 | 4.6–9.6 | 0.1278 | 0.7285 |

### 3.3 Video observations and figure analysis

#### C000: Walk forward

**Video observations.** All five videos show forward walking. The stepping direction and gait pose vary across samples, but the overall movement is forward and the root trajectories are approximately straight.

**Metrics and figures.** The mean trajectory distance is `0.6620`, and the mean duration is `4.68 s`. The trajectories show broadly consistent main displacement directions, with differences in travel distance and local offsets.

**Interpretation.** The five videos and their trajectory plots indicate that all samples generally walk forward along approximately straight paths. Variations in stepping and body pose affect the visual appearance but do not, by themselves, establish that the root moves in the wrong direction. Likewise, trajectory diversity is not direct evidence of a directional error.

#### C001: Walk backward

**Video observations.** All five videos generally show backward walking. Although the footwork or body orientation varies, the overall backward direction is recognizable, and the root trajectories are approximately straight.

**Metrics and figures.** The mean trajectory distance is `0.4598`, lower than the values for forward, left, and right movement. The duration standard deviation is approximately `0.769 s`, indicating relatively consistent durations across the five samples.

**Interpretation.** The forward and backward videos are both recognizable as the corresponding walking direction in this sample set. This suggests that the model expressed the two opposing walking intents in these generated samples. However, the trajectory plots and RMS distances do not themselves classify direction. To convert these observations into an automated metric, the coordinate system must first be defined, after which the direction of root displacement can be computed over time segments.

#### C002: Walk left (video observations show turning direction)

**Video observations.** Videos `00`, `01`, and `02` turn left, while videos `03` and `04` turn right. Thus, three of the five samples show the target leftward behavior and two show the opposite turning direction.

**Metrics and figures.** The mean trajectory distance is `0.9691`, the highest among the four directional conditions. The mean pose distance is `0.1751`, also the highest in this experiment. The root-trajectory plot shows widely dispersed paths, with substantial differences in lateral offsets and endpoint positions.

**Interpretation.** Under the leftward prompt, three videos show the target direction and two turn in the opposite direction. The group also has the highest mean trajectory distance (`0.9691`), accompanied by substantial spread in the trajectory plot. Together, the video and quantitative evidence indicate that the generated behavior is inconsistent under this prompt. The value `0.9691` measures trajectory variation across samples, not the directional error rate; the `3/5` observation applies only to these five generated samples and should not be generalized as the model's overall accuracy.

#### C003: Walk right (video observations show turning direction)

**Video observations.** Videos `00`, `02`, and `03` turn right; video `01` turns left; and video `04` first turns left and then turns right. Using consistency of the target direction throughout the sequence as the criterion, three samples maintain the target rightward behavior, one turns in the opposite direction, and one changes direction during the motion.

**Metrics and figures.** The mean trajectory distance is `0.7285`, lower than for the leftward condition but clearly nonzero. The mean duration is `5.88 s`, with a standard deviation of approximately `1.874 s`, the largest duration variation among the four direction conditions.

**Interpretation.** The rightward condition also contains inconsistent results: one sample turns in the opposite direction, and another changes from left to right. The direction switch in video `04` is especially important because a measure based only on the start and endpoint—or an average pairwise distance—may hide changes during the motion. The root trajectory should therefore be divided into temporal segments and the displacement direction of each segment calculated to identify changes in direction. The whole-sequence average distance cannot fully characterize this left-then-right pattern.

### 3.4 Summary of the direction experiment

1. Frame-by-frame inspection shows that all five forward-walking and all five backward-walking samples generally match their target directions. Variations in footwork or body appearance should not automatically be treated as errors in root-motion direction.
2. In the leftward condition, samples `00`–`02` turn left and samples `03`–`04` turn right. In the rightward condition, sample `01` turns left, while sample `04` first turns left and then turns right.
3. The leftward condition has the highest mean trajectory distance (`0.9691`), and both videos `03` and `04` turn in the opposite direction. The rightward condition includes the reversed direction in `01` and the left-then-right switch in `04`. These findings indicate inconsistency in left/right generation. Trajectory distance helps describe variation across samples but cannot replace video-level directional assessment.
4. The prompts `walks to the left/right` and the visible behavior of turning the body are not interchangeable concepts. Future automated evaluation should explicitly define whether the target is lateral translation, body-orientation rotation, or both, and should fix the coordinate axes and viewing perspective so that different motion quantities are not merged into one direction label.

Figures:

- Summary charts: [Duration distribution](../project/figures/exp_direction_v1/duration_distribution.png) · [Pose diversity](../project/figures/exp_direction_v1/pose_diversity.png) · [Trajectory diversity](../project/figures/exp_direction_v1/trajectory_diversity.png)
- Root trajectories: [C000 — Walk forward](../project/figures/exp_direction_v1/C000_root_trajectory.png) · [C001 — Walk backward](../project/figures/exp_direction_v1/C001_root_trajectory.png) · [C002 — Walk left](../project/figures/exp_direction_v1/C002_root_trajectory.png) · [C003 — Walk right](../project/figures/exp_direction_v1/C003_root_trajectory.png)

---

## 4. Experiment 3: Generation Duration and Compound Actions

### 4.1 Setup

This experiment compares generation duration and corresponding pose/trajectory variation for prompts ranging from a single action to a compound sequence. The prompts describe jumping, walking forward, walking forward and turning left, and the compound action “stand up, walk forward, turn around, and sit down.”

### 4.2 Quantitative results

Source: [diversity_summary.csv](../project/results/exp_length_v1/diversity_summary.csv).

| ID / Prompt | Samples | Mean duration ± SD (s) | Duration range (s) | Mean pose distance | Mean trajectory distance |
|---|---:|---:|---:|---:|---:|
| C000 — Jump | 5 | 3.96 ± 1.023 | 2.6–5.4 | 0.0786 | 0.1370 |
| C001 — Walk forward | 5 | 4.84 ± 0.975 | 3.6–6.4 | 0.2013 | 1.1623 |
| C002 — Walk forward and turn left | 5 | 4.32 ± 0.449 | 3.8–4.8 | 0.2004 | 1.3119 |
| C003 — Stand up, walk forward, turn around, and sit down | 5 | 9.80 ± 0.000 | 9.8–9.8 | 0.2390 | 0.8835 |

### 4.3 Video observations and figure analysis

#### C000: Jump

**Quantitative results.** The mean duration is `3.96 s`, with a standard deviation of approximately `1.023 s` and a range of `2.6–5.4 s`. The mean pose distance is `0.0786`, and the mean trajectory distance is `0.1370`; both are relatively low among the four conditions.

**Interpretation.** The relatively low pose and root-trajectory distances indicate that the five generated results are comparatively similar in the metric space used in this report. However, low pairwise distances do not imply high motion quality, nor do they prove that take-off, flight, and landing are all natural and complete. Assessing jump quality requires examining these action phases together with quantitative measurements.

#### C001: Walk forward

**Video observations.** In video `02`, the body turns to the right while the root trajectory remains approximately straight. In video `04`, the body turns to the left while the root trajectory is clearly not straight.

**Metrics and figures.** The mean trajectory distance is `1.1623`, the second highest among the four conditions in this experiment; the mean pose distance is `0.2013`. The trajectory plot shows noticeable differences in travel amplitude, lateral offset, and/or path shape across samples.

**Interpretation.** Videos `02` and `04` clearly show that body orientation and root-path geometry do not necessarily correspond one-to-one. In `02`, the body turns right although the root path remains approximately straight; in `04`, the body turns left while the root path is not straight. Therefore, path curvature alone cannot determine whether the body has turned, nor whether the turn direction matches the text prompt.

#### C002: Walk forward and turn left

**Video observations.** In video `00`, the body turns right at the end of the sequence, opposite to the requested left turn. Videos `01`, `02`, `03`, and `04` perform the requested forward-walking-then-left-turn action.

**Metrics and figures.** The mean duration is `4.32 s`, with a standard deviation of approximately `0.449 s`, so durations are relatively consistent. The mean trajectory distance is `1.3119`, the highest among the four conditions in this experiment, and the mean pose distance is `0.2004`. The root-trajectory plot also shows substantial differences between sample paths.

**Interpretation.** Although the durations are similar across the five sequences, one sample ends with a turn in the direction opposite to the prompt. This demonstrates that stable duration does not imply semantic correctness. The large trajectory distance indicates path variation but cannot automatically distinguish a valid turn from a direction error. Video inspection complements the metrics: most samples perform the target action, but one has a clear final-direction error.

#### C003: Stand up, walk forward, turn around, and sit down

**Video observations.** Videos `00`, `01`, and `02` follow the order “stand up → walk forward → turn around → sit down.” Videos `03` and `04` instead follow “stand up → turn around → walk forward → sit down,” placing the turn before forward walking.

**Metrics and figures.** All five samples have a duration of `9.8 s`, corresponding to 196 frames, and the duration standard deviation is `0`. The mean pose distance is `0.2390`, the highest among the four conditions; the mean trajectory distance is `0.8835`. The trajectory plot shows that root paths still differ even though all sequences have the same duration.

**Interpretation.** All five sequences contain 196 frames (`9.8 s`), but videos `00`–`02` follow the requested order, whereas `03` and `04` turn before walking forward. In the inference code, `opt.max_motion_length` is set to `196`, and the number of length classes is computed as `200 // opt.unit_length`, with the default `unit_length=4`. The model samples a discrete length class and multiplies it by `unit_length` to obtain the frame count, so the highest valid class corresponds to `49 × 4 = 196` frames. All five outputs reaching this upper boundary means that, for this prompt, the length estimator's sampled outputs all fall in the maximum class. Therefore, `9.8 s` reflects a boundary imposed by the current length-estimation and sampling configuration; it should not be interpreted as proof that the model autonomously planned the ideal duration for the action sequence. In this compound-action condition, `2/5` samples do not follow the specified sub-action order, indicating a clear remaining issue with temporal action sequencing.

### 4.4 Summary of the duration experiment

1. Generation duration does not increase monotonically with the number of actions described in the prompt. The mean duration for “walk forward and turn left” (`4.32 s`) is even shorter than for “walk forward” alone (`4.84 s`).
2. The compound prompt produces 196 frames (`9.8 s`) in all five samples, with zero duration variation. However, only videos `00`–`02` perform all sub-actions in the requested order; videos `03` and `04` swap the order of turning and walking forward.
3. For the forward-walking and forward-walking-plus-left-turn samples, visible body orientation and root-path shape do not always correspond directly. Root-path analysis alone may miss body rotation, while video appearance still needs a defined coordinate and orientation convention to be interpreted against trajectory metrics.
4. Future evaluation of compound actions should measure duration, whether each sub-action occurs, whether the sequence is correctly ordered, and changes in body orientation. Generated frame count alone is not sufficient to evaluate long-horizon action understanding.

Figures:

- Summary charts: [Duration distribution](../project/figures/exp_length_v1/duration_distribution.png) · [Pose diversity](../project/figures/exp_length_v1/pose_diversity.png) · [Trajectory diversity](../project/figures/exp_length_v1/trajectory_diversity.png)
- Root trajectories: [C000 — Jump](../project/figures/exp_length_v1/C000_root_trajectory.png) · [C001 — Walk forward](../project/figures/exp_length_v1/C001_root_trajectory.png) · [C002 — Walk forward and turn left](../project/figures/exp_length_v1/C002_root_trajectory.png) · [C003 — Compound action](../project/figures/exp_length_v1/C003_root_trajectory.png)

---

## 5. Overall Findings Across the Three Experiments

### 5.1 Diversity, direction, and duration measure different aspects

- **Diversity experiment.** The sit-down condition has the largest duration variation (`7.36 ± 3.123 s`), the highest mean pose distance (`0.2000`), and an incomplete final sitting pose in video `03`. The right-hand-raising condition has the lowest mean trajectory distance (`0.0051`), consistent with an action that mainly involves the upper limbs. The forward-walking condition has the largest mean trajectory distance (`0.7565`), although all five videos generally show straight forward walking; this indicates that trajectory distance is also affected by differences in movement amplitude and position across samples.
- **Direction experiment.** All five forward and backward samples generally match the requested direction. In the leftward condition, videos `03` and `04` turn in the opposite direction; in the rightward condition, video `01` turns in the opposite direction and video `04` first turns left and then right. The leftward condition has the largest mean trajectory distance (`0.9691`), alongside the visibly dispersed behavior, but this distance is not a directional-accuracy measure.
- **Duration experiment.** The mean duration for walking forward and turning left (`4.32 s`) is shorter than for walking forward alone (`4.84 s`). The compound prompt produces 196 frames (`9.8 s`) in all five samples, but videos `03` and `04` turn before walking forward. A fixed output length therefore does not guarantee the correct order of sub-actions.

### 5.2 Root trajectories alone cannot describe full-body motion

The current trajectory analysis extracts the root's XZ coordinates. This is useful for describing planar root movement, but it does not directly describe body orientation, foot direction, arm movement, or semantic action order. In the forward-walking condition of the duration experiment (`exp_length_v1/C001`), video `02` shows the body turning right while the root trajectory remains approximately straight, whereas video `04` shows the body turning left while the root trajectory is not straight. These two cases directly demonstrate that body-orientation changes and root-path geometry are not in one-to-one correspondence.

To analyze the directional deviations observed in this experiment, future work should estimate body orientation from the recovered joint coordinates and add orientation-angle changes and segment-wise trajectory-direction metrics. For the ordering errors in compound actions `03` and `04`, each sub-action—standing up, walking, turning, and sitting down—should be detected separately and checked for temporal order. This would convert failures already identified in the videos into repeatable, quantitative evaluations.

### 5.3 Conclusions and scope

Based on the video-by-video inspection, summary tables, and trajectory plots, the following conclusions apply to the model, prompts, and sampling settings used in this experiment:

1. Under the fixed pretrained model and prompt setup, generated duration, pose, and root-trajectory behavior vary across text conditions.
2. Video inspection reveals several semantic deviations: sit-down sample `03` does not complete the sitting action; some left/right samples show reversed or mixed turning directions; sample `00` in the forward-walking-plus-left-turn condition turns right at the end; and compound-action samples `03` and `04` turn before walking forward, contrary to the specified order.
3. Quantitative statistics and video inspection provide complementary evidence, but the sample size is small, and the custom pose/trajectory distances do not directly measure generation quality or directional accuracy.

These results **do not support** claims about the model's overall accuracy in general settings. They also do not support direct comparison of the custom pose/trajectory distances with the paper's official metrics, nor do they establish that a 9.8-second output means the compound action was planned successfully.

## 6. Recommendations for Further Work

1. **Increase the sample count and repeat across random seeds.** Five samples per prompt are insufficient for stable estimates of means and standard deviations. Increase the number of samples and record random seeds, checkpoints, and generation parameters.
2. **Define directional semantics and coordinate conventions.** Distinguish lateral translation, body-orientation rotation, and foot orientation. Then specify the positive and negative X/Z directions and the viewing perspective so that visual direction is not confused with root-motion direction.
3. **Add automated directional metrics.** Develop explicit direction rules based on segment-wise root displacement and changes in body orientation. Samples with mixed directions should be labeled separately rather than forced into a binary correct/incorrect category.
4. **Evaluate compound-action completion and order.** For “stand up → walk → turn → sit down,” detect whether each sub-action occurs, whether the order is correct, and whether the final sitting pose is reached. Treat duration as a supplementary measure.
5. **Analyze identified failure samples individually.** Prioritize sit-down sample `C002` video `03` in the diversity experiment; videos `03` and `04` in the leftward `C002` condition and videos `01` and `04` in the rightward `C003` condition of the direction experiment; videos `02` and `04` in the forward-walking `C001` condition and video `00` in the forward-walking-plus-left-turn `C002` condition of the duration experiment; and videos `03` and `04` in the compound-action `C003` condition. Match each video to its rows in `sample_metrics.csv` and the pairwise-distance table to examine its duration, pose, and trajectory values.
6. **Investigate why the length estimator repeatedly samples the maximum class.** In the inference code, `num_classes = 200 // opt.unit_length`, the default `unit_length=4`, and sampled classes are converted to frame counts by `m_lens = length * opt.unit_length`. The maximum valid class corresponds to 196 frames, and `opt.max_motion_length` is also set to 196 (see [`gen_motion_script.py`](../gen_motion_script.py#L92-L151) and [`base_options.py`](../options/base_options.py#L16-L20)). This explains the 196-frame upper bound. The next step is to record the length probability distribution for the compound prompt and investigate whether all five samples land in the maximum class because of a concentrated probability distribution, prompt features, or sampling variation.
7. **Keep metric definitions transparent.** Continue documenting inverse normalization, `recover_from_ric()`, root-centering, 60-frame resampling, and the RMS-distance calculation so that the analysis can be reproduced and audited.

## 7. Related Files

- [Diversity analysis script](../project/analyze_diversity.py)
- [Direction analysis script](../project/analyze_direction.py)
- [Duration analysis script](../project/analyze_length.py)
- [Diversity experiment summary](../project/results/exp_diversity_v1/diversity_summary.csv)
- [Direction experiment summary](../project/results/exp_direction_v1/diversity_summary.csv)
- [Duration experiment summary](../project/results/exp_length_v1/diversity_summary.csv)
- [Project reproduction summary](reproduction_summary.md)

---

*Note: Quantitative values in this report come from the experiment summary CSV files. Action completion, direction, and sub-action order were recorded through video-by-video inspection. All conclusions are limited to the model, prompts, and sampling settings used here and should not be generalized to the model's overall performance in other settings.*
