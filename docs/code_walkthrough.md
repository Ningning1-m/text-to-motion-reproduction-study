# Text-to-Motion Code Walkthrough

This document explains the code that is actually used by the project.
The original upstream files are kept unchanged.

## 1. Inference entry: `gen_motion_script.py`

The command we run is:

```bat
python gen_motion_script.py --name Comp_v6_KLD01 --text_file input.txt --repeat_times 5 --ext exp_direction_v1 --gpu_id 0
```

### Imports

```python
import os
```

Loads Python's operating-system utilities. This file uses `os.makedirs` to
create result directories.

```python
from os.path import join as pjoin
```

Imports `os.path.join` and renames it to `pjoin`. It joins path components
using the correct separator for Windows or Linux.

```python
import utils.paramUtil as paramUtil
```

Loads the HumanML3D skeleton definition, including the kinematic chains used
to draw connected bones.

```python
from options.evaluate_options import TestOptions
```

Loads the command-line parser. It turns arguments such as `--gpu_id 0` and
`--text_file input.txt` into the `opt` configuration object.

```python
from torch.utils.data import DataLoader
```

Imports PyTorch's batch iterator. The inference script uses it to iterate over
the text descriptions one at a time.

```python
from utils.plot_script import *
from networks.modules import *
from scripts.motion_process import *
from utils.utils import *
```

These wildcard imports expose project helpers directly. They are convenient
in the original research code, but explicit imports would be easier to audit.

### `plot_t2m`

```python
def plot_t2m(data, save_dir, captions):
```

Defines the post-processing function. `data` contains generated normalized
motion features; `save_dir` is the output prefix; `captions` are the texts.

```python
data = dataset.inv_transform(data)
```

Undo z-score normalization:

```text
original_feature = normalized_feature * std + mean
```

This step must happen before `recover_from_ric`.

```python
for i, (caption, joint_data) in enumerate(zip(captions, data)):
```

Pairs each caption with its generated motion. `enumerate` adds the sample
index.

```python
joint = recover_from_ric(
    torch.from_numpy(joint_data).float(),
    opt.joints_num
).numpy()
```

Converts the NumPy array to a float PyTorch tensor, recovers 22 joints with
three coordinates each, and converts the result back to NumPy.

```python
save_path = '%s_%02d' % (save_dir, i)
```

Builds a zero-padded output name such as `gen_motion_00`.

```python
joint = motion_temporal_filter(joint, sigma=1)
```

Applies temporal smoothing to reduce frame-to-frame jitter before rendering.

```python
np.save(save_path + '_a.npy', joint)
```

Stores recovered joint coordinates in NumPy format.

```python
plot_3d_motion(
    save_path + '_a.mp4',
    paramUtil.t2m_kinematic_chain,
    joint,
    title=caption,
    fps=20
)
```

Draws each frame as a 3D skeleton and encodes the animation as an MP4.

### `build_models`

```python
def build_models(opt):
```

Constructs the neural-network modules. Construction defines the architecture;
it does not yet provide learned behavior.

```python
if opt.text_enc_mod == 'bigru':
```

Selects the bidirectional GRU text encoder configured by the checkpoint.

```python
text_encoder = TextEncoderBiGRU(...)
```

Creates the network that converts word embeddings and POS features into
contextual text features.

```python
text_size = opt.dim_text_hidden * 2
```

A bidirectional GRU concatenates forward and backward hidden states, hence the
factor of two.

```python
seq_prior = TextDecoder(...)
seq_decoder = TextVAEDecoder(...)
att_layer = AttLayer(...)
movement_enc = MovementConvEncoder(...)
movement_dec = MovementConvDecoder(...)
```

These modules respectively model latent motion variables, decode motion
snippets, attend to text, and convert between motion features and movement
latents.

```python
return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec
```

Returns the modules to `CompTrainerV6`.

### Main execution

```python
if __name__ == '__main__':
```

Runs the following block only when this file is executed directly, not when it
is imported.

```python
parser = TestOptions()
opt = parser.parse()
```

Creates the parser and reads command-line arguments.

```python
opt.device = torch.device(
    "cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id)
)
```

Selects CPU for `-1`, otherwise selects a CUDA device such as `cuda:0`.

```python
opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
opt.model_dir = pjoin(opt.save_root, 'model')
opt.meta_dir = pjoin(opt.save_root, 'meta')
```

Builds the checkpoint paths. For our experiment these resolve to
`checkpoints/t2m/Comp_v6_KLD01`.

```python
opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
```

Builds a separate output directory for each experiment name.

```python
os.makedirs(opt.joint_dir, exist_ok=True)
os.makedirs(opt.animation_dir, exist_ok=True)
```

Creates output folders without failing if they already exist.

```python
mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
std = np.load(pjoin(opt.meta_dir, 'std.npy'))
```

Loads the 263-dimensional feature normalization statistics.

```python
w_vectorizer = WordVectorizer('./glove', 'our_vab')
```

Loads the vocabulary, word vectors, and word-to-index mapping.

```python
text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec = build_models(opt)
```

Creates the model architecture.

```python
trainer = CompTrainerV6(
    opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc
)
```

Wraps the modules in the project training/inference controller.

```python
dataset = RawTextDataset(opt, mean, std, opt.text_file, w_vectorizer)
```

Reads every non-empty line in `input.txt`, tokenizes it with spaCy, and
converts tokens into word and POS features.

```python
epoch, it, sub_ep, schedule_len = trainer.load(
    pjoin(opt.model_dir, opt.which_epoch + '.tar')
)
```

Loads learned parameters from `latest.tar`. This is where the randomly
constructed architecture receives the author's trained behavior.

```python
trainer.eval_mode()
trainer.to(opt.device)
```

Switches off training behavior such as dropout and moves model parameters to
the selected CPU/GPU device.

```python
estimator = MotionLenEstimatorBiGRU(...)
```

Creates the separate text-to-length network.

```python
checkpoints = torch.load(...)
estimator.load_state_dict(checkpoints['estimator'])
```

Loads the learned length-estimator parameters.

```python
data_loader = DataLoader(dataset, batch_size=1, ...)
```

Creates an iterator that returns one prompt at a time.

```python
with torch.no_grad():
```

Disables gradient storage because this is inference, not training.

```python
for i, data in enumerate(data_loader):
```

Processes each input prompt.

```python
word_emb, pos_ohot, caption, cap_lens = data
```

Unpacks the dataset output: word embeddings, POS one-hot vectors, original
caption, and caption length.

```python
word_emb = word_emb.detach().to(opt.device).float()
pos_ohot = pos_ohot.detach().to(opt.device).float()
```

Detaches tensors from any graph, moves them to the device, and ensures
floating-point computation.

```python
pred_dis = estimator(word_emb, pos_ohot, cap_lens)
pred_dis = nn.Softmax(-1)(pred_dis).squeeze()
```

Predicts a probability distribution over motion-length classes.

```python
for t in range(opt.repeat_times):
```

Repeats generation for the same prompt. Repeated samples can differ because
the model samples length and latent motion variables.

```python
length = torch.multinomial(pred_dis, 1)
m_lens = length * opt.unit_length
```

Samples one length class and converts it to a frame length.

```python
pred_motions, _, att_wgts = trainer.generate(
    word_emb, pos_ohot, cap_lens, m_lens,
    m_lens[0] // opt.unit_length, dim_pose
)
```

Generates normalized 263-dimensional motion features conditioned on the text
and selected length.

```python
sub_dict['motion'] = pred_motions.cpu().numpy()
```

Moves generated motion back to CPU and stores it as a NumPy array.

```python
np.save(
    pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])),
    motion
)
```

Saves the normalized generated representation. The filename records the
sample index and number of frames.

```python
plot_t2m(motion, ..., captions)
```

Denormalizes the representation, recovers joints, smooths them, and renders
the MP4.

## 2. Text processing: `data/dataset.py`

The class used by our inference script is `RawTextDataset`.

```python
self.nlp = spacy.load('en_core_web_sm')
```

Loads the English spaCy model.

```python
with cs.open(text_file) as f:
    for line in f.readlines():
```

Reads the prompt file line by line.

```python
word_list, pos_list = self.process_text(line.strip())
```

Removes surrounding whitespace and obtains normalized words and POS tags.

```python
tokens = [
    '%s/%s' % (word_list[i], pos_list[i])
    for i in range(len(word_list))
]
```

Combines each word with its POS tag, for example `walk/VERB`.

```python
self.data_dict.append({
    'caption': line.strip(),
    'tokens': tokens
})
```

Stores the original sentence and processed tokens.

```python
doc = self.nlp(sentence)
```

Runs spaCy tokenization, lemmatization, and POS tagging.

```python
if not word.isalpha():
    continue
```

Skips punctuation and tokens containing non-letter characters.

```python
if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
    word_list.append(token.lemma_)
else:
    word_list.append(word)
```

Lemmatizes nouns and verbs, except `left`, which is preserved because it is
also a direction word.

```python
tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
```

Adds start-of-sentence and end-of-sentence markers.

```python
tokens = tokens + ['unk/OTHER'] * (...)
```

Pads short sentences to a fixed maximum length.

```python
word_embeddings.append(word_emb[None, :])
pos_one_hots.append(pos_oh[None, :])
```

Builds arrays containing the word vectors and POS vectors for each token.

## 3. Word vectors: `utils/word_vectorizer.py`

```python
vectors = np.load(pjoin(meta_root, '%s_data.npy' % prefix))
```

Loads the matrix of pretrained word vectors.

```python
words = pickle.load(...)
word2idx = pickle.load(...)
```

Loads the vocabulary and maps each word to its row in the vector matrix.

```python
self.word2vec = {w: vectors[word2idx[w]] for w in words}
```

Builds a Python dictionary from word to vector.

```python
def _get_pos_ohot(self, pos):
```

Creates a one-hot POS/category vector.

```python
if word in self.word2vec:
```

Uses a known word vector when available.

```python
else:
    word_vec = self.word2vec['unk']
```

Uses the learned unknown-word vector for out-of-vocabulary words.

## 4. Motion recovery: `scripts/motion_process.py`

```python
def recover_root_rot_pos(data):
```

Recovers the root joint's orientation and global position from the first
motion features.

```python
rot_vel = data[..., 0]
```

Reads root rotation velocity.

```python
r_rot_ang[..., 1:] = rot_vel[..., :-1]
r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
```

Integrates angular velocity over time to obtain root rotation angle.

```python
r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
```

Reads root horizontal velocity features.

```python
r_pos = qrot(qinv(r_rot_quat), r_pos)
r_pos = torch.cumsum(r_pos, dim=-2)
```

Rotates local translation into the world frame and accumulates it over time.

```python
def recover_from_ric(data, joints_num):
```

Recovers 22 joint positions from the Relative Joint Coordinates representation.

```python
positions = data[..., 4:(joints_num - 1) * 3 + 4]
```

Extracts relative positions for non-root joints from the 263-dimensional
representation.

```python
positions = positions.view(
    positions.shape[:-1] + (-1, 3)
)
```

Reshapes the flat coordinate block into `(time, joints, xyz)`.

```python
positions = qrot(
    qinv(r_rot_quat[..., None, :]).expand(...),
    positions
)
```

Rotates local joint coordinates into the global orientation.

```python
positions[..., 0] += r_pos[..., 0:1]
positions[..., 2] += r_pos[..., 2:3]
```

Adds the root's global X/Z translation to every joint.

```python
positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
```

Prepends the root joint, producing `(batch, time, 22, 3)`.

## 5. Our analysis scripts

`project/analyze_direction.py` and `project/analyze_length.py` are copies of
the corrected analysis pipeline. Their only experiment-specific changes are:

```python
EXPERIMENT_NAME = "exp_direction_v1"
PROMPTS_FILE = ROOT / "project" / "prompts_direction.txt"
```

or:

```python
EXPERIMENT_NAME = "exp_length_v1"
PROMPTS_FILE = ROOT / "project" / "prompts_length.txt"
```

They then:

1. Load normalized `[1, T, 263]` files.
2. Apply `data * std + mean`.
3. Call `recover_from_ric`.
4. Compute duration, root path, root displacement, speed, and acceleration.
5. Resample motions to a common length for pairwise comparison.
6. Root-center poses and align trajectories.
7. Write CSV summaries and PNG figures.

The diversity numbers are exploratory project metrics, not official CVPR 2022
benchmark metrics.
