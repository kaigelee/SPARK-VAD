import json
import random
import os
# input_path = "../shared-nvme/UCF-crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"  # 替换为你的 txt 文件路径
# output_path = "../shared-nvme/UCF-crime/test.json"

# data_list = []
# id = 0
# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         temporal_label = [-1, -1, -1, -1 ,-1, -1]
#         parts = line.strip().split()
#         b = -1
#         e = -1
#         b1 = -1
#         e1 = -1
#         # if len(parts)==4:
#         #     video, length, begin, end = parts
#         # elif len(parts)==6:
#         #     video, length, begin, end, b, e= parts
#         # else:
#         #     video, length, begin, end, b, e, b1, e1 =parts
#         video = parts[0].split('.')[0]
#         video_file_path = f"../shared-nvme/UCF-crime/frames_test/{video}"
#         files = [f for f in os.listdir(video_file_path) if os.path.isfile(os.path.join(video_file_path, f))]
#         length = len(files)
#         begin = parts[2]
#         end = parts[3]
#         b = parts[4]
#         e = parts[5]

#         if begin!=-1:
#             temporal_label[0] = int(begin)
#             temporal_label[1] = int(end)
#             temporal_label[2] = int(b)
#             temporal_label[3] = int(e)
#             temporal_label[4] = int(b1)
#             temporal_label[5] = int(e1)
#         data_list.append({"video": video, "length": int(length), "temporal_label":temporal_label})

#         # if id%1==0:
#         #     data_list.append({"video": video, "length": int(length)-1})
#         id+=1

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data_list, f, indent=2, ensure_ascii=False)

# print(f"成功写入 {output_path}")




data = [f for f in os.listdir("../shared-nvme/UCF-crime/frames_train")]
path = "../shared-nvme/TAD/train.json"
len_list = []
print(data)
for d in data:
    len_list.append(len([f for f in os.listdir(f"../shared-nvme/UCF-crime/frames_train/{d}")]))


print(len_list)
normal_videos = [{"video":v,"length":l-1} for v,l in zip(data,len_list) if 'Normal' in v and l<10000 and l!=0]
abnormal_videos = [{"video":v,"length":l-1} for v,l in zip(data,len_list) if 'Normal' not in v]
print(normal_videos)
print(abnormal_videos)

sampled_normal_200 = random.sample(normal_videos, 100)
sampled_abnormal_200 = random.sample(abnormal_videos, 200)
train_sampled = sampled_normal_200 + sampled_abnormal_200

# 创建已使用集合，便于排除
used_videos_set = set([v['video'] for v in sampled_normal_200 + sampled_abnormal_200])

# 过滤未使用的视频
remaining_normal = [v for v in normal_videos if v['video'] not in used_videos_set]
remaining_abnormal = [v for v in abnormal_videos if v['video'] not in used_videos_set]

# 再次采样：30 Normal + 30 Abnormal，确保不重复
sampled_normal_30 = random.sample(remaining_normal, 80)
sampled_abnormal_30 = random.sample(remaining_abnormal, 80)

# 合并新列表
new_sampled_list = sampled_normal_30 + sampled_abnormal_30

# with open(path, "r") as f:
#     data = json.load(f)
#     normal_videos = [v for v in data if 'Normal' in v['video']]
#     abnormal_videos = [v for v in data if 'Normal' not in v['video']]

#     sampled_normal_50 = random.sample(normal_videos, 140)
#     sampled_abnormal_50 = random.sample(abnormal_videos, 140)

#     train_sampled = sampled_normal_50 + sampled_abnormal_50

#     # 创建已使用集合，便于排除
#     used_videos_set = set([v['video'] for v in sampled_normal_50 + sampled_abnormal_50])

#     # 过滤未使用的视频
#     remaining_normal = [v for v in normal_videos if v['video'] not in used_videos_set]
#     remaining_abnormal = [v for v in abnormal_videos if v['video'] not in used_videos_set]

#     # 再次采样：30 Normal + 30 Abnormal，确保不重复
#     sampled_normal_30 = random.sample(remaining_normal, 50)
#     sampled_abnormal_30 = random.sample(remaining_abnormal, 50)

#     # 合并新列表
#     new_sampled_list = sampled_normal_30 + sampled_abnormal_30

with open("../shared-nvme/UCF-crime/train_new_.json", "w") as f:
    json.dump(train_sampled, f, indent=4)

with open("../shared-nvme/UCF-crime/val_new_.json", "w") as f:
    json.dump(new_sampled_list, f, indent=4)
