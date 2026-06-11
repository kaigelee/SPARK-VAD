import os
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from torchvision import transforms
import pdb
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import pytorch_lightning as pl
import itertools

from datasets.video_instruction_dataset_VERA import Video_Instruct_Dataset
import math
import torch
from transformers import AutoTokenizer, AutoModel

import argparse
import json
import sys
from pathlib import Path
import os
import pathlib
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from scipy.special import softmax
# import faiss
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("ImageBind")

import faiss
from ImageBind.imagebind import data
from ImageBind.imagebind.models.imagebind_model import ModalityType
from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos
from src.utils.sample_utils import uniform_temporal_subsample
from src.utils.torch_utils import initialize_vlm_model_and_device


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    print(world_size, 'xxx')
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class VideoAnomalyDetectionModel(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizer_instruct, normal_traffic_conditions,
                 abnormal_traffic_conditions, generation_config, vlm_model=None, epochs=5, train_vis_root=""):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer_instruct = optimizer_instruct
        self.normal_traffic_conditions = normal_traffic_conditions
        self.abnormal_traffic_conditions = abnormal_traffic_conditions


        self.generation_config = generation_config
        self.epochs = epochs
        self.validation_step_outputs = []
        self.validation_step_count = []
        self.automatic_optimization = False
        self.vlm_model = vlm_model
        self.train_vis_root = train_vis_root
        # self.feature_dir = "video_feature_UCF-crime"
        self.feature_dir = "video_feature"

    def _load_and_transform_data(self, batch_frame_paths):
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(
                batch_frame_paths, "cuda:0"
            ),
        }
        return inputs

    def _calculate_search_vectors(self, inputs):
        with torch.no_grad():
            # self.vlm_model.to('cpu')
            embeddings = self.vlm_model(inputs)
            search_vectors = embeddings[ModalityType.VISION].cpu().numpy()
            del embeddings  # 显式释放 GPU 显存
            # torch.cuda.empty_cache()  # 强制清理缓
        return search_vectors

    def _prepare_frame_data(self, video, video_len, batch_center_frame_idxs, frames_per_clip):
        batch_clip_frame_paths = [
            [
                video + "/{}.jpg".format(frame_idx)
                for frame_idx in range(clip_center_frame,
                                       min(clip_center_frame + frames_per_clip, video_len),
                                       )
            ]
            for clip_center_frame in batch_center_frame_idxs
        ]

        return batch_clip_frame_paths

    def forward(self, batch_idx, search_vectors):

        self.model.eval()
        # Get batch and frame size
        feat_list = []
        for normal in self.normal_traffic_conditions:
            final_normal = 'Normal Scene: ' + normal
            feat_list.append(final_normal)

        for abnormal in self.abnormal_traffic_conditions:
            final_abnormal = 'Anomalous Scene: ' + abnormal
            feat_list.append(final_abnormal)

        #####
        input_text = {
            ModalityType.TEXT: data.load_and_transform_text(feat_list, "cuda:0")}
        with torch.no_grad():
            text_embeddings = self.vlm_model(input_text)
            text_index_vectors = text_embeddings[ModalityType.TEXT].cpu().numpy()
            del text_embeddings  # 显式释放 GPU 显存
            # torch.cuda.empty_cache()  # 强制清理缓
            faiss.normalize_L2(text_index_vectors)

        #Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection
        text_index_vectors[len(self.normal_traffic_conditions):] = text_index_vectors[
                                                                   len(self.normal_traffic_conditions):] * 0.95

        simi_matrix = search_vectors @ text_index_vectors.T
        simi_score = [round(float(x), 4) for x in simi_matrix.reshape(-1)]
        del text_index_vectors

        #####

        # if batch_idx % 10 == 0:
        #     f=open('./log_generating_5q.txt','a')
        #     f.write('1--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        #     f.write(question+'\n')
        #     f.write('2--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        #
        #     f.write('\n')
        #     f.close()
        return simi_score

    def select_flexible_interval_items(self, lst, max_interval=15, min_interval=1, count=16):

        for interval in range(max_interval, min_interval - 1, -1):
            max_start = len(lst) - (count - 1) * interval
            if max_start > 0:
                start_index = random.randint(0, max_start - 1)
                return [lst[start_index + i * interval][0] if len(lst[0]) == 1 else lst[start_index + i * interval] for
                        i in range(count)]

    def training_step(self, batch, batch_idx):

        # response = self.model.chat(tokenizer, None, '<image><image><image><image>'+self.optimizer_instruct, generation_config)

        self.model.eval()
        # self.model.to('cuda:0')
        # self.vlm_model.to('cpu')

        frame_path_list, labels, video_name = batch
        print("\n正常事件数量：",len(self.normal_traffic_conditions))
        print("异常事件数量：",len(self.abnormal_traffic_conditions))

        batch_end_frame = len(frame_path_list)
        batch_center_frame_idxs = range(
            0, batch_end_frame, 30
        )
        batch_clip_frame_paths = self._prepare_frame_data(
            self.train_vis_root + "/" + video_name[0], len(frame_path_list), batch_center_frame_idxs, 30
        )

        if not os.path.exists(f'/root/autodl-tmp/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy'):
            inputs = self._load_and_transform_data(batch_clip_frame_paths)
            search_vectors_all = self._calculate_search_vectors(inputs)
            faiss.normalize_L2(search_vectors_all)
            np.save(f'/root/autodl-tmp/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy',
                    search_vectors_all)

        else:
            search_vectors_all = np.load(f'/root/autodl-tmp/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy')


        select_frames = self.select_flexible_interval_items(frame_path_list)

        pixel_values = torch.cat(
            [load_image(image_file, max_num=1).to(torch.bfloat16).cuda() for image_file in select_frames], dim=0)



        gt_labels = 'anomalous' if labels[0] == 1 else 'normal'

        optimizer_instruct_batch = self.optimizer_instruct.replace(
            '[$label]',
            str(gt_labels)
        )

        num = 2
        desc_list = ''.join([f'{i+1}. Generated event {i+1}\n'for i in range(num)])
        optimizer_instruct_batch = optimizer_instruct_batch.replace(
            '[$num]',
            str(num)
        )
        optimizer_instruct_batch = optimizer_instruct_batch.replace(
            '[$desc_list]',
            str(desc_list)
        )


        video_prefix = ''.join(['<image>' for i in range(len(select_frames))])

        # input_string = '[' + ('[' + video_prefix + '] ') * 1 + ']\n'
        input_string = video_prefix
        optimizer_instruct_batch = input_string + optimizer_instruct_batch

        # Insert the current prompt question into the optimizer instructions

        # Generate model response for training
        # response = self.model.chat(tokenizer, pixel_values, optimizer_instruct_batch, generation_config)
        # torch.cuda.empty_cache()
        count = 5
        descriptions = None
        while count:
            response = self.model.chat(tokenizer, pixel_values, optimizer_instruct_batch, generation_config)
            print("-------------------------")
            print(response)
            print("-------------------------")
            # print(response)
            try:
                if "Generated event 1" in response:
                    des = re.findall(r"Generated event \d+:([\s\S]*?)(?=\n\d\.|$)", response)
                    descriptions = []
                    for d in des:
                        descriptions.append(d.strip())
                else:
                    descriptions = [line.split('. ', 1)[1].split(': ')[-1].strip() for line in response.split('\n') if line.strip() and line.split('. ', 1)[1].split(':')[-1].strip()!=""]
                print(descriptions)
                break
            except:
                count -=1
        if descriptions is not None:
            if labels[0] == 1:
                self.abnormal_traffic_conditions += descriptions
            else:
                self.normal_traffic_conditions += descriptions

            new_descriptions = []
            for desc in descriptions:
                if labels[0] == 1:
                    desc = 'Anomalous Scene: ' + desc
                else:
                    desc = 'Normal Scene: ' + desc
                new_descriptions.append(desc)




        return torch.tensor(1.0, requires_grad=True).to(pixel_values.device)

    def validation_step(self, batch, batch_idx):
        # frame_path_list, labels, video_name = batch

        # # 计算等间隔的索引（包含首尾）
        # # indices = np.linspace(0, len(frame_path_list) - 1, 8, dtype=int)
        # # select_frames = [frame_path_list[i][0] for i in indices]
        # # # inputs_normal = self._load_and_transform_data([select_frames])
        # # inputs = {
        # #     ModalityType.VISION: data.load_and_transform_video_data(
        # #         [select_frames], "cuda:0"
        # #     ),
        # # }
        # # with torch.no_grad():
        # #     # self.model.to('cpu')
        # #     # self.vlm_model.to('cuda:0')
        # #     embeddings = self.vlm_model(inputs)
        # #
        # #     search_vectors = embeddings[ModalityType.VISION].cpu().numpy()
        # #     del embeddings  # 显式释放 GPU 显存
        # #     torch.cuda.empty_cache()  # 强制清理缓
        # #     faiss.normalize_L2(search_vectors)
        # batch_end_frame = len(frame_path_list)
        # batch_center_frame_idxs = range(
        #     0, batch_end_frame, 30
        # )
        # batch_clip_frame_paths = self._prepare_frame_data(
        #     self.train_vis_root + "/" + video_name[0], len(frame_path_list), batch_center_frame_idxs, 30
        # )
        #
        # if not os.path.exists(f'/root/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy'):
        #     inputs = self._load_and_transform_data(batch_clip_frame_paths)
        #     search_vectors_all = self._calculate_search_vectors(inputs)
        #     faiss.normalize_L2(search_vectors_all)
        #     np.save(f'/root/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy',
        #             search_vectors_all)
        #
        # else:
        #     search_vectors_all = np.load(f'/root/VAD/{self.feature_dir}/30/' + video_name[0] + '.npy')
        #
        # # feat_list = []
        # # for abnormal in self.abnormal_traffic_conditions:
        # #     final_abnormal = 'Anomalous Scene: ' + abnormal
        # #     feat_list.append(final_abnormal)
        # #
        # # #####
        # # input_text = {
        # #     ModalityType.TEXT: data.load_and_transform_text(feat_list, "cuda:0")}
        # # with torch.no_grad():
        # #     # self.vlm_model.to('cpu')
        # #     text_embeddings = self.vlm_model(input_text)
        # #     # self.vlm_model.to('cuda:0')
        # #     text_index_vectors = text_embeddings[ModalityType.TEXT].cpu().numpy()
        # #     del text_embeddings  # 显式释放 GPU 显存
        # #     # torch.cuda.empty_cache()  # 强制清理缓
        # #     faiss.normalize_L2(text_index_vectors)
        # #     abnormal_feat_avg = text_index_vectors.mean(axis=0, keepdims=True)
        # #
        # # similarities = (search_vectors_all @ abnormal_feat_avg.T).reshape(-1)
        # # # shape: (4,)
        # #
        # # # 获取相似度最高的两个片段索引
        # # # print(batch_end_frame, len(similarities), len(batch_clip_frame_paths))
        # # # assert len(similarities) == len(batch_clip_frame_paths), "长度不一致"
        # #
        # # top3_indices = np.argsort(-similarities)[:min(3, len(batch_clip_frame_paths))]
        # # top3_indices_sorted = sorted(top3_indices)
        # #
        # # select_clip = []
        # # for i in top3_indices_sorted:
        # #     select_clip.extend(batch_clip_frame_paths[i])
        # #
        # # select_frames = self.select_flexible_interval_items(select_clip)
        #
        # # 前一半是异常，后一半是正常
        # num_abn = self.len_abnormal_before
        # num_norm = self.len_norm_before
        # text_index_abn = text_index_vectors_before[:num_abn]
        # text_index_norm = text_index_vectors_before[num_abn:]
        #
        # # 与每个文本特征计算相似度
        # # search_vectors_all: (num_segments, dim)
        # sim_abn = search_vectors_all @ text_index_abn.T  # shape: (N, num_abn)
        # sim_norm = search_vectors_all @ text_index_norm.T  # shape: (N, num_norm)
        #
        # # 拼接后softmax归一化
        # all_sim = np.concatenate([sim_abn, sim_norm], axis=1)
        # weights = np.exp(all_sim) / np.sum(np.exp(all_sim), axis=1, keepdims=True)
        #
        # # 构造分数向量：异常=1，正常=0
        # score_vec = np.concatenate([np.ones(num_abn), np.zeros(num_norm)])
        #
        # # softmax加权平均分数
        # segment_scores = (weights * score_vec).sum(axis=1)
        #
        # # 取top3片段
        # top3_indices = np.argsort(-segment_scores)[:min(3, len(batch_clip_frame_paths))]
        # top3_indices_sorted = sorted(top3_indices)
        #
        # # 提取片段帧路径
        # select_clip = []
        # for i in top3_indices_sorted:
        #     select_clip.extend(batch_clip_frame_paths[i])
        #
        # # 从片段中选择图片
        # select_frames = self.select_flexible_interval_items(select_clip)
        #
        # inputs_normal = self._load_and_transform_data([select_frames])
        # search_vectors = self._calculate_search_vectors(inputs_normal)
        # faiss.normalize_L2(search_vectors)
        #
        # simi_score = self.forward(batch_idx, search_vectors)
        #
        # ####
        #
        # # 构建文本异常标签 row：前 normal 是 0，后 abnormal 是 1
        # text_anomaly_score = [0] * len(self.normal_traffic_conditions) + [1] * len(self.abnormal_traffic_conditions)
        #
        # # 找出 top-10 相似度的下标（从大到小排序）
        # top_indices = sorted(range(len(simi_score)), key=lambda i: -simi_score[i])[:min(10, len(simi_score))]
        #
        # top_sims = [simi_score[i] for i in top_indices]
        #
        # top_labels = [text_anomaly_score[i] for i in top_indices]
        #
        # top_weights = softmax(top_sims)
        #
        # # 计算加权异常分数
        # weighted_score = sum(w * label for w, label in zip(top_weights, top_labels))
        # del search_vectors
        #
        # predict_labels = torch.tensor([1 if weighted_score > 0.5 else 0]).to("cuda:0")
        #
        # ############
        #
        # # Convert the last token in response to a label (1, 0, or 2 for unknown)
        # # predict_labels = torch.tensor([1 if response[-1] == '1' else 0 if response[-1] == '0' else 2 for response in responses]).to(pixel_values.device)
        #
        # # Calculate accuracy by comparing predicted labels to actual labels
        # correct_predictions = (predict_labels == labels).sum()
        # accuracy = correct_predictions / len(labels)
        #
        # # Log validation accuracy
        # self.log('val_acc', accuracy.item(), prog_bar=True, on_step=True, sync_dist=True, batch_size=len(labels))
        #
        # self.validation_step_outputs.append(correct_predictions)
        # self.validation_step_count.append(len(labels))
        #
        # epoch_average = torch.stack(self.validation_step_outputs).sum() / sum(self.validation_step_count)
        # print(epoch_average.item())
        # torch.cuda.empty_cache()
        # return {'val_acc': accuracy}
        return {'val_acc': 1.0}

    def on_validation_epoch_end(self):
        # Calculate average accuracy across all batches in the validation set
        # epoch_average = torch.stack(self.validation_step_outputs).sum() / sum(self.validation_step_count)
        # self.log("val_avg_acc", epoch_average.item(), sync_dist=True)
        # self.validation_step_outputs.clear()  # free memory
        # self.validation_step_count.clear()  # free memory

        f = open(text_file, 'a')
        # f.write(f'accuracy: {epoch_average.item()}' + '\n')
        f.write('New Event:' + '\n')
        f.write(str(self.normal_traffic_conditions))
        f.write('\n')
        f.write(str(self.abnormal_traffic_conditions))
        f.write('\n')
        f.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)



text_file = './TAD_InterVL3/generated_events_InterVL3_TAD_no_keyframe_ep1_test.txt'

f = open(text_file, 'w')
f.writelines('Generative Events\n')
f.close()

path = '/root/autodl-tmp/InternVL3-8B'



vlm_model, device = initialize_vlm_model_and_device()

# data_set = "UCF-crime"
data_set = "TAD"


train_dataset = Video_Instruct_Dataset(vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
                                       ann_root=f'/root/autodl-tmp/{data_set}/train_new_.json', num_sampled_frame=16,
                                       vlm_model=vlm_model)
val_dataset = Video_Instruct_Dataset(vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
                                     ann_root=f'/root/autodl-tmp/{data_set}/val_new_.json', TEST_FLAG=True, num_sampled_frame=16,
                                     vlm_model=vlm_model)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)




new_normal_traffic_conditions = ['A normal scene']
new_abnormal_traffic_conditions = ['An abnormal scene']


file = open('VERA_optimizer_instruct_2.txt', 'r')
optimizer_instruct = file.read()
file.close()






model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
    eos_token_id=151645,
    pad_token_id=151645,
)
# generation_config = dict(max_new_tokens=1024, do_sample=True)


# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModel(
    model=model, tokenizer=tokenizer, optimizer_instruct=optimizer_instruct,
    normal_traffic_conditions=new_normal_traffic_conditions, abnormal_traffic_conditions=new_abnormal_traffic_conditions,
    generation_config=generation_config,
    vlm_model=vlm_model, train_vis_root=f'/root/autodl-tmp/{data_set}/frames_train'
)

from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger(save_dir="logs_VERA/")

# Train the model using the Lightning Trainer
trainer = pl.Trainer(
    logger=tb_logger,
    val_check_interval=280,
    log_every_n_steps=5,
    max_epochs=1,
    enable_checkpointing=False,
    devices=1,  # Use all available GPUs
    accelerator="gpu",  # GPU training
    # strategy="ddp"  # Distributed Data Parallel training
)

TRAIN_FLAG = True

if TRAIN_FLAG:
    trainer.fit(lightning_model, train_loader, val_loader)
else:
    trainer.validate(lightning_model, val_loader)
