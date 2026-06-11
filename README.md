```bash
conda create -n VAD python=3.10

conda activate VAD

pip install -r requirements.txt

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

#可参考transformer_data.py生成训练和测试json文件



1.training.py

作用：基于Prompt（包含关键词）生成正常和异常事件集合

主要变量：

```python
#输出事件的文件路径，目前是每次迭代手动修改名称，ep1->ep2->ep3
text_file = './TAD_InterVL3/generated_events_InterVL3_TAD_no_keyframe_ep1_test.txt'
#MLLM权重路径
path = '/root/autodl-tmp/InternVL3-8B'
#数据集名称
data_set = "TAD"
#创建训练数据集，vis_root为视频帧根路径，ann_root为标注文件，num_sampled_frame为每个视频采样帧数
train_dataset = Video_Instruct_Dataset(vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
                                       ann_root=f'/root/autodl-tmp/{data_set}/train_new_.json', num_sampled_frame=16,
                                       vlm_model=vlm_model)
#这个验证数据集目前没什么用
val_dataset = Video_Instruct_Dataset(vis_root=f'/root/autodl-tmp/{data_set}/frames_train',
                                     ann_root=f'/root/autodl-tmp/{data_set}/val_new_.json', TEST_FLAG=True, num_sampled_frame=16,
                                     vlm_model=vlm_model)

#Prompt文件
file = open('VERA_optimizer_instruct_2.txt', 'r')
optimizer_instruct = file.read()

#val_check_interval为val的间隔，因为val没什么用，所以这里的280是训练视频的数量
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


#路径修改
'/root/autodl-tmp/VAD'全文搜索改为自己的本地路径'/path/to/VAD'
```



2.VERA_optimizer_instruct_2.txt

```
You are tasked with generating events in the given video segments based on a series of consecutive image frames and corresponding keywords.
The given video is [$label] events. Please generate [$num] distinct [$label] events based on the input video frames.

括号里的内容ep1时为空，ep_n(n>1)时为上一个ep即ep_n-1生成的关键词
{
normal_keywords = ["navigating", "road", "passing", "vehicles", "storefronts", "sidewalk", "pedestrians", "intersection", "retail area", "fire truck", "driving", "cars", "green area"]
abnormal_keywords = ["swerves", "swerving", "narrowly misses", "narrowly missing", "pedestrian", "debris", "lost control", "sudden", "vehicle", "highway", "scatters debris", "miss collision site", "avoid", "scatter debris", "miss pedestrian", "miss car", "miss parking area", "miss highway", "miss intersection", "miss pedestrian cross", "miss another vehicle"]
}

Output events in the following format, make sure to generate events that are clear and contextually relevant to the video content:
[$desc_list]
Please strictly follow the format, DO NOT OUTPUT ANYTHING ELSE EXCEPT EVENTS DESCRIPTION.
```



3.train_after.py

作用：给这一个epoch生成的事件集合进行打分

主要变量：

```python
dataset = "TAD"
#预生成的视频特征路径，每一个.npy文件为1s视频对应的视觉特征，基于ImageBind-Huge生成
feature_dir = "video_feature"

#执行完training.py后，例如第一个epoch生成的generated_events_InterVL3_TAD_no_keyframe_ep1_test.txt文件中会保存这一个epoch生成的正常和异常和事件集合，对应复制到下面对应的正常和异常变量
normal_traffic_conditions = ... 
abnormal_traffic_conditions = ...

#视频帧路径和标注文件路径
vis_root=f'/root/autodl-tmp/{dataset}/frames_train'
ann_root=f'/root/autodl-tmp/{dataset}/train_new_.json'

#路径修改
'/root/autodl-tmp/VAD'全文搜索改为自己的本地路径'/path/to/VAD'

#最后会打印normal_traffic_conditions和abnormal_traffic_conditions每个事件对应的分数，字典形式打印{'event_1':score_1,...,'event_n':score_n}
print(dict_tab_norm)
print(dict_tab_abnorm)
```



4.sort_and_filter.py

作用：选择所有normal，和所有value大于0的abnormal，因为正常片段打分总是对的，所以排除异常事件是正确的，异常片段打分不一定对（因为对于异常视频无法精确提取到异常片段对应的帧来分析，所以分数不是鲁棒的），所以还不能排除正常片段

主要变量：

```python
#执行完train_after.py后，打印dict_tab_norm和dict_tab_abnorm，对应复制到下面对应的正常和异常变量
normal_tab = ...
abnormal_tab = ...
```

这个筛选后的打印输出是**这个epoch**最终的事件集合，用来在测试集上测试得到结果



5.update_instruction.py

作用：基于优选事件集合生成关键词，并与前一个epoch关键词进行合并去重，得到下一次生成事件时在VERA_optimizer_instruct_2.txt中Prompt的关键词部分

主要变量：

```python
#执行完sort_and_filter.py后，分别手动提取这个epoch正常和异常得分前10高的事件，和前一个epoch得到的normal_events_good和abnormal_events_good分别拼接在一起，即epoch1（10+10）、epoch2（20+20）...
normal_events_good =
abnormal_events_good =

#模型路径
path = '/root/autodl-tmp/InternVL3-8B'
```



6.Traffic-VAD/test.py

在测试集上测试，输出AUC

```python
#数据集
dataset = "TAD"

#特征文件夹
feature_dir = "video_feature"
#数据文件路径
root_path=f"/root/autodl-tmp/{dataset}/frames_test"
annotationfile_path=f"/root/autodl-tmp/{dataset}/test.txt"
ann_root = f'/root/autodl-tmp/{dataset}/test.json'


#待测试的事件集合
normal_traffic_conditions =...
abnormal_traffic_conditions =...
```



