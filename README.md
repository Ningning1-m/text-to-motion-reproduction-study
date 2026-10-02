# Text-to-Motion Reproduction Study

This repository documents a local reproduction study based on:

> Generating Diverse and Natural 3D Human Motions from Text, CVPR 2022.

The upstream implementation and paper are available here:

- [Official implementation](https://github.com/EricGuo5513/text-to-motion)
- [Project page](https://ericguo5513.github.io/text-to-motion)
- [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)

This repository is a learning-oriented reproduction and analysis project. It
does not claim to reproduce every official benchmark result.

## What This Project Does

The official pretrained model maps a natural-language description to a 3D
human motion:

```text
text prompt
    -> spaCy preprocessing and word/POS features
    -> text encoder
    -> motion-length estimator
    -> text-to-motion generator
    -> normalized 263-D motion representation
    -> inverse normalization
    -> 22-joint XYZ motion
    -> MP4 animation
```

This study adds three small experiments:

1. **Diversity**: generate multiple motions from the same text.
2. **Direction**: compare forward, backward, left, and right prompts.
3. **Length**: compare generated durations for short and compositional prompts.

It also contains a small rule-based **Action Primitive JSON** interface. This
interface is a project extension, not a component of the original paper.

## Repository Structure

```text
gen_motion_script.py          Official inference entry point
data/                         Dataset and raw-text loading
networks/                     Model modules and trainers
scripts/                      Motion representation recovery
utils/                        Word vectors, plotting, and utilities
options/                      Command-line options
project/                      Custom experiments and analysis scripts
project/prompts_*.txt         Experiment prompts
project/results/              CSV summaries and Action Primitive JSON
project/figures/              Experiment figures
docs/code_walkthrough.md      Inference pipeline explanation
docs/analysis_code_line_by_line.md
                              Detailed analysis-code walkthrough
docs/reproduction_summary.md  Results and interpretation
```

Model checkpoints and generated videos are intentionally excluded from Git.
They are downloaded or generated locally.

## Environment

The experiments were run on Windows with:

```text
Conda environment: motion-study
Python:            3.8
PyTorch:           2.0.1
GPU:               NVIDIA CUDA GPU
spaCy:             3.4.4
matplotlib:        3.3.1
```

Create and activate the environment:

```bat
conda create -n motion-study python=3.8 -y
conda activate motion-study
```

Install the main packages. The exact PyTorch command depends on the local
CUDA setup; the study used the CUDA 11.8 wheel when the network allowed it:

```bat
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
conda install -c conda-forge spacy=3.4.4 -y
conda install numpy=1.23.5 scipy tqdm matplotlib=3.3.1 -y
```

Install the spaCy English model:

```bat
python -m spacy download en_core_web_sm
```

If the network cannot access the spaCy model registry, download the compatible
wheel on a networked machine and install it locally.

## Pretrained Models

Download the HumanML3D pretrained checkpoints from the upstream project and
place them under:

```text
checkpoints/t2m/Comp_v6_KLD01/
checkpoints/t2m/Decomp_SP001_SM001_H512/
checkpoints/t2m/length_est_bigru/
checkpoints/t2m/text_mot_match/
```

The normalization files required by the analysis scripts are:

```text
checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy
checkpoints/t2m/Comp_v6_KLD01/meta/std.npy
```

Do not commit checkpoint files to GitHub.

## Minimal Inference

From the repository root:

```bat
conda activate motion-study
python gen_motion_script.py ^
  --name Comp_v6_KLD01 ^
  --text_file input_single.txt ^
  --repeat_times 1 ^
  --ext customized ^
  --gpu_id 0
```

Generated files are written under:

```text
eval_results/t2m/Comp_v6_KLD01/customized/
```

The `.npy` motion representation is converted back to joint coordinates and
rendered as an MP4 by the official inference script.

## Run the Experiments

Generate several samples for the direction experiment:

```bat
python gen_motion_script.py ^
  --name Comp_v6_KLD01 ^
  --text_file project/prompts_direction.txt ^
  --repeat_times 5 ^
  --ext exp_direction_v1 ^
  --gpu_id 0
```

Analyze the generated motions:

```bat
python -m project.analyze_direction
```

Run the length experiment:

```bat
python gen_motion_script.py ^
  --name Comp_v6_KLD01 ^
  --text_file project/prompts_length.txt ^
  --repeat_times 5 ^
  --ext exp_length_v1 ^
  --gpu_id 0

python -m project.analyze_length
```

Run the diversity experiment:

```bat
python gen_motion_script.py ^
  --name Comp_v6_KLD01 ^
  --text_file project/prompts_diversity.txt ^
  --repeat_times 5 ^
  --ext exp_diversity_v1 ^
  --gpu_id 0

python -m project.analyze_diversity
```

The analysis scripts first undo feature normalization:

```text
original_motion = normalized_motion * std + mean
```

They then call the official `recover_from_ric()` function to obtain
`[T, 22, 3]` joint coordinates before computing exploratory statistics.

## Action Primitive Interface

Run:

```bat
python -m project.action_primitives
```

The result is saved to:

```text
project/results/action_primitives.json
```

This parser is intentionally small and rule-based. It does not train a model,
recognize arbitrary actions, or control a robot.

## Results

The current results and interpretation are documented in:

- [Reproduction summary](docs/reproduction_summary.md)
- [Code walkthrough](docs/code_walkthrough.md)
- [Analysis code line-by-line guide](docs/analysis_code_line_by_line.md)

The CSV files and PNG figures are included under `project/results/` and
`project/figures/`. The custom metrics are exploratory and should not be
reported as the official CVPR 2022 FID, R-Precision, Matching Score,
Diversity, or Multimodality benchmark.

## Reproducibility Scope

Included in this study:

- pretrained-model text-to-motion inference;
- custom prompt generation;
- multiple-sample diversity analysis;
- direction-conditioned trajectory analysis;
- generated-duration analysis;
- a rule-based action primitive export;
- code-reading and data-shape documentation.

Not included as a full reproduction:

- retraining all official models;
- reproducing the complete HumanML3D benchmark table;
- official FID, R-Precision, Matching Score, and Multimodality evaluation;
- training a new embodied-agent policy.

## Attribution

The model architecture, pretrained weights, data representation, and core
inference implementation come from the upstream CVPR 2022 project. This
repository adds the learning notes, experiments, analysis scripts, and
interpretation described above.

## License

See the upstream [LICENSE](LICENSE) and the upstream repository for the
original project terms.
