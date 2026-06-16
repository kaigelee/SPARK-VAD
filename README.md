# Video Anomaly Detection Code Usage

This repository provides a multi-stage pipeline for generating, scoring, filtering, and evaluating normal/abnormal event descriptions for video anomaly detection.

The current workflow contains several manual steps, including manually updating event files, copying generated event lists, and updating keyword prompts between epochs.

## 1. Environment Setup

Create and activate the Conda environment:

```bash
conda create -n SPARK_VAD python=3.10
conda activate SPARK_VAD
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install FlashAttention using the provided wheel file:

```bash
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## 2. Data Preparation

You may refer to `transformer_data.py` to generate the training and testing JSON files.

The generated JSON files are used by the training and testing scripts.

---

## 3. Overall Pipeline

The overall pipeline is:

```text
training.py
    -> generate normal/abnormal event descriptions

train_after.py
    -> score the generated events

sort_and_filter.py
    -> filter and select events for the current epoch

update_instruction.py
    -> generate and merge keywords for the next epoch

Traffic-VAD/test.py
    -> evaluate the selected event set on the test set and report AUC
```

---

## 4. Event Generation: `training.py`

### Purpose

`training.py` generates normal and abnormal event sets based on a prompt that contains keywords.

### Main Variables

```python
# Output path of generated events.
# Currently, the file name should be manually changed for each epoch:
# ep1 -> ep2 -> ep3 -> ...
text_file = './TAD_InterVL3/generated_events_InterVL3_TAD_no_keyframe_ep1_test.txt'

# Path to the MLLM weights
path = '/root/autodl-tmp/InternVL3-8B'

# Dataset name
data_set = "TAD"

# Training dataset.
# vis_root: root path of video frames
# ann_root: annotation file
# num_sampled_frame: number of sampled frames for each video
train_dataset = Video_Instruct_Dataset(
    vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
    ann_root=f'/root/autodl-tmp/{data_set}/train_new_.json',
    num_sampled_frame=16,
    vlm_model=vlm_model
)

# Validation dataset.
# This dataset is currently not used in the main workflow.
val_dataset = Video_Instruct_Dataset(
    vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
    ann_root=f'/root/autodl-tmp/{data_set}/val_new_.json',
    TEST_FLAG=True,
    num_sampled_frame=16,
    vlm_model=vlm_model
)

# Prompt file
file = open('VERA_optimizer_instruct_2.txt', 'r')
optimizer_instruct = file.read()
```

The trainer is defined as follows:

```python
trainer = pl.Trainer(
    logger=tb_logger,
    val_check_interval=280,
    log_every_n_steps=5,
    max_epochs=1,
    enable_checkpointing=False,
    devices=1,
    accelerator="gpu",
    # strategy="ddp"
)
```

Here, `val_check_interval=280` is used because the validation dataset is currently not used, and `280` corresponds to the number of training videos.

### Path Modification

Before running the script, globally replace the following path with your local repository path:

```text
/root/autodl-tmp/VAD
```

For example:

```text
/path/to/VAD
```

### Run

```bash
python training.py
```

After running this script, the generated normal and abnormal event descriptions will be saved to the file specified by `text_file`.

---

## 5. Prompt File: `VERA_optimizer_instruct_2.txt`

This file defines the instruction used by `training.py` to generate event descriptions.

```text
You are tasked with generating events in the given video segments based on a series of consecutive image frames and corresponding keywords.
The given video is [$label] events. Please generate [$num] distinct [$label] events based on the input video frames.

The content inside the following brackets is empty for ep1.
For ep_n where n > 1, it should be filled with the keywords generated from the previous epoch ep_n-1.

{
normal_keywords = ["navigating", "road", "passing", "vehicles", "storefronts", "sidewalk", "pedestrians", "intersection", "retail area", "fire truck", "driving", "cars", "green area"]
abnormal_keywords = ["swerves", "swerving", "narrowly misses", "narrowly missing", "pedestrian", "debris", "lost control", "sudden", "vehicle", "highway", "scatters debris", "miss collision site", "avoid", "scatter debris", "miss pedestrian", "miss car", "miss parking area", "miss highway", "miss intersection", "miss pedestrian cross", "miss another vehicle"]
}

Output events in the following format, make sure to generate events that are clear and contextually relevant to the video content:
[$desc_list]
Please strictly follow the format, DO NOT OUTPUT ANYTHING ELSE EXCEPT EVENTS DESCRIPTION.
```

For the first epoch, the keyword section is empty.

For later epochs, copy the keywords generated by `update_instruction.py` into this prompt file.

---

## 6. Event Scoring: `train_after.py`

### Purpose

`train_after.py` scores the event set generated in one epoch.

### Main Variables

```python
# Dataset name
dataset = "TAD"

# Path to pre-generated video features.
# Each .npy file corresponds to the visual feature of a 1-second video segment.
# The features are extracted using ImageBind-Huge.
feature_dir = "video_feature"

# After running training.py, the generated events are saved in a file such as:
# generated_events_InterVL3_TAD_no_keyframe_ep1_test.txt
# Copy the generated normal and abnormal event sets into the following variables.
normal_traffic_conditions = ...
abnormal_traffic_conditions = ...

# Video frame path and annotation file path
vis_root = f'/root/autodl-tmp/{dataset}/frames_train'
ann_root = f'/root/autodl-tmp/{dataset}/train_new_.json'
```

### Path Modification

Before running the script, globally replace the following path with your local repository path:

```text
/root/autodl-tmp/VAD
```

For example:

```text
/path/to/VAD
```

### Run

```bash
python train_after.py
```

The script will print the score of each normal and abnormal event in dictionary format:

```python
print(dict_tab_norm)
print(dict_tab_abnorm)
```

The printed results are in the following format:

```python
{
    "event_1": score_1,
    "event_2": score_2,
    ...
}
```

---

## 7. Event Filtering: `sort_and_filter.py`

### Purpose

`sort_and_filter.py` filters the scored event set.

The script selects:

```text
all normal events
+
all abnormal events whose value is larger than 0
```

The reason is that the scoring of normal segments is reliable, so excluding abnormal events based on normal segments is reasonable. However, the scoring of abnormal segments is not always robust, because it is difficult to precisely extract the frames corresponding to the abnormal part in abnormal videos. Therefore, normal events should not be excluded based on abnormal videos.

### Main Variables

After running `train_after.py`, copy the printed `dict_tab_norm` and `dict_tab_abnorm` into the following variables:

```python
normal_tab = ...
abnormal_tab = ...
```

### Run

```bash
python sort_and_filter.py
```

The printed output is the final event set of the current epoch.

This filtered event set can be directly used for testing on the test set.

---

## 8. Keyword Update: `update_instruction.py`

### Purpose

`update_instruction.py` generates keywords from the selected high-quality events and merges them with the keywords from the previous epoch.

The generated keywords are used to update the keyword section in `VERA_optimizer_instruct_2.txt` for the next epoch.

### Main Variables

After running `sort_and_filter.py`, manually select the top-10 highest-scoring normal events and the top-10 highest-scoring abnormal events from the current epoch.

Then concatenate them with the previous epoch’s `normal_events_good` and `abnormal_events_good`.

For example:

```text
epoch 1: 10 normal events + 10 abnormal events
epoch 2: 20 normal events + 20 abnormal events
...
```

Set the following variables:

```python
normal_events_good = ...
abnormal_events_good = ...

# Path to the model
path = '/root/autodl-tmp/InternVL3-8B'
```

### Run

```bash
python update_instruction.py
```

After running this script, copy the generated keywords into `VERA_optimizer_instruct_2.txt` for the next epoch.

---

## 9. Testing: `Traffic-VAD/test.py`

### Purpose

`Traffic-VAD/test.py` evaluates the selected event set on the test set and reports AUC.

### Main Variables

```python
# Dataset name
dataset = "TAD"

# Feature folder
feature_dir = "video_feature"

# Test video frame path and annotation files
root_path = f"/root/autodl-tmp/{dataset}/frames_test"
annotationfile_path = f"/root/autodl-tmp/{dataset}/test.txt"
ann_root = f'/root/autodl-tmp/{dataset}/test.json'

# Event set to be tested
normal_traffic_conditions = ...
abnormal_traffic_conditions = ...
```

The variables `normal_traffic_conditions` and `abnormal_traffic_conditions` should be filled with the selected event set from `sort_and_filter.py`.

### Run

```bash
cd Traffic-VAD
python test.py
```

The script will output the AUC on the test set.

---

## 10. Recommended Usage Workflow

A typical epoch-level workflow is:

```bash
python training.py
```

Then manually copy the generated normal and abnormal events into `train_after.py`:

```bash
python train_after.py
```

Then copy the printed event-score dictionaries into `sort_and_filter.py`:

```bash
python sort_and_filter.py
```

Then manually select the top-10 normal and abnormal events, update `update_instruction.py`, and run:

```bash
python update_instruction.py
```

Finally, copy the generated keywords into `VERA_optimizer_instruct_2.txt` and start the next epoch.

After obtaining the final event set, evaluate it on the test set:

```bash
cd Traffic-VAD
python test.py
```

---

## 11. Notes

- The current implementation contains several manual steps.
- The output event file name in `training.py` should be manually updated for each epoch, such as `ep1`, `ep2`, and `ep3`.
- The generated event sets should be manually copied into `train_after.py`.
- The printed scoring dictionaries from `train_after.py` should be manually copied into `sort_and_filter.py`.
- The selected high-quality events should be manually copied into `update_instruction.py`.
- The generated keywords from `update_instruction.py` should be manually copied into `VERA_optimizer_instruct_2.txt` for the next epoch.
- Before running the code, replace `/root/autodl-tmp/VAD` with your own local path.


## TODO List

- [ ] Replace hard-coded paths with configurable arguments or a unified configuration file.
- [ ] Add command-line arguments for dataset name, model path, feature directory, annotation files, and output paths.
- [ ] Automate epoch-level file naming and output management, such as `ep1`, `ep2`, and `ep3`.
- [ ] Reduce manual copy-and-paste operations by automatically passing generated events, scoring results, selected events, and updated keywords between scripts.
- [ ] Save intermediate outputs, including generated events, event scores, filtered event sets, and keywords, in structured files such as `.json`.
- [ ] Integrate `training.py`, `train_after.py`, `sort_and_filter.py`, and `update_instruction.py` into a unified epoch-level pipeline.
