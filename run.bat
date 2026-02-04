@REM 解压zip3.tar 并根据图片名称  生成metadata.json
python extract_tar_and_build_metadata.py
@REM 帧间去重分组 计算组内每张图片的质量得分 选出最佳图片 生成metadata_grouped.json
python group_consecutive_similar_frames.py --no-vis
@REM 使用微调过的yolov8来检测提取动漫人物的 person head face 图片 结果在metadata_crops.json中
python extract_anime_persons_yolo.py
@REM dinov2 去提取 face图和head图的视觉特征 768 维 CLS token 嵌入 生成metadata_crops_dinov2_features.pt
python extract_dinov2_features.py
@REM tagger 去提取 face图和head图的视觉特征 512 维 logits 生成metadata_crops_tagger_features.pt
python extract_tagger_features.py
@REM 二次分组过滤，针对face 和 head分别做一次，过滤掉高度相似的face和head 得到metadata_crops_grouped.json
python group_face_by_phash.py
@REM 拼接 [dino_umap, tagger_umap] 后聚类 生成metadata_crops_clustered.json 测试两组参数 
@REM 结果在metadata_crops_clustered_head.json和metadata_crops_clustered_face.json中
python cluster_and_copy_crops.py --feature-type head --min-cluster-size 3 --cluster-selection-epsilon 0.50
python cluster_and_copy_crops.py --feature-type face --min-cluster-size 3 --cluster-selection-epsilon 0.40