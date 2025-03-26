# HST-SLR: Hierarchical Sub-action Tree for Continuous Sign Language Recognition
This repo holds codes of the paper: Hierarchical Sub-action Tree for Continuous Sign Language Recognition

## 1. Prerequisites
1. Create a `Conda` environment.
```bash
conda create -n HST python=3.7 -y && conda activate HST
```
2. Install PyTorch with Conda
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
You can download PyTorch for your own CUDA version by yourself, but the version is better >=1.13 to be compatible with ctcdecode or these may exist errors.

3. Install ctcdecode
ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode. (ctcdecode is only supported on the Linux platform.)

4. Install other requirements 
```bash
pip install -r requirements.txt 
```

## 2. Data Preparation
You can choose any one of following datasets to verify the effectiveness of HST-SLR.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
`ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     
```bash
cd ./preprocess
python dataset_preprocess.py --process-image --multiprocessing
```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
`ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     
```bash
cd ./preprocess
python T_process.py
python dataset_preprocess-T.py --process-image --multiprocessing
```

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
`ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     
```bash
cd ./preprocess
python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
```

## 3. Build Hierarchical Sub-action Tree

### 3.1 Generate Sub-action
Run the folling command to generate descriptions for each word. You should modify the api_key in `description_generate.py`(line 5) and set the target dataset in (line 3) `phoenix2014`, `phoenix2014T`, `CSLDaily`. If you want to generate the descriptions yourself, please make sure to remove the file `description_{target}.txt` in the directory first.
```bash
cd ./hst_build/generation
python description_generate.py
```

### 3.2 Build HST
Run the following command to build HST with the descriptions obtained before. You can set the target dataset in `cluster.py`(line 5) `phoenix2014`, `phoenix2014T`, `CSLDaily`. We have provided the generated results.
```bash
cd ./hst_build
python cluster.py
```

### 3.3 Prototype, Update Matrix and Loss Matrix
Run the following command to set the prototype for each tree node, generate the update matrix for updating and find the tree node that contains a certain word. You can set the target dataset in `prototype_set.py`(line 5), in `update_matrix.py`(line 4) and in `search_matrix.py`(line 4) `phoenix2014`, `phoenix2014T`, `CSLDaily`.
```bash
python prototype_set.py
python update_matrix.py
python search_matrix.py
```

We have provided all the generated results mentioned earlier. Please download the zip file through the link [Google Drive](https://drive.google.com/file/d/1z2n-bh2pgR5iCX9tDJpgixMgHGtKDts1/view?usp=drive_link). Then put the file in `./HDT_prototype` and unzip it.

## 4. Testing

### PHOENIX2014 dataset

| Dev WER  | Test WER  | Pretrained model                                             |
| ---------- | ----------- | --- |
| 17.6%      | 18.5%       | [[Google Drive]](https://drive.google.com/file/d/14ZtqXj7GN9qtc38UqyJFIZMPZeZSRXNt/view?usp=drive_link)|

We wrongly delete the original checkpoint and retrain the model with similar accuracy (Dev: 17.9%, Test: 18.2%)

### PHOENIX2014-T dataset

| Dev WER  | Test WER  | Pretrained model                                             |
| ---------- | ----------- | --- |
| 17.4%      | 19.0%       | [[Google Drive]](https://drive.google.com/file/d/1oXnrgd7nGKGLvipW3paU6_gYi7l5BSt1/view?usp=drive_link)|

We wrongly delete the original checkpoint and retrain the model with similar accuracy (Dev: 17.4%, Test: 19.1%)

### CSL-Daily dataset

| Dev WER  | Test WER  | Pretrained model                                            |
| ---------- | ----------- | --- |
| 27.5%      | 27.4%       | [[Google Drive]](https://drive.google.com/file/d/112_GqITfK4I0jtWQloDN7RTNRcgvScOi/view?usp=drive_link)|


​To evaluate the pretrained model, choose the dataset from `phoenix2014/phoenix2014-T/CSL-Daily` in `./configs/baseline.yaml`(line 3) and set the target in `tree_network.py`(line 14) `phoenix2014`, `phoenix2014T`, `CSLDaily`. Then run the command below：   
```bash
python main.py --config ./configs/baseline.yaml --device your_device --work-dir ./work_dir/your_expname/ --load-weights path_to_weight.pt --phase test
```

## 5. Training

To train the SLR model, choose the dataset from `phoenix2014/phoenix2014-T/CSL-Daily` in `./configs/baseline.yaml`(line 3) and set the target in `tree_network.py`(line 14) `phoenix2014`, `phoenix2014T`, `CSLDaily`. Then run the command below:
```bash
python main.py --config ./configs/baseline.yaml --device your_device --work-dir ./work_dir/your_expname/
```

For CSL-Daily dataset, You may choose to reduce the lr by half from 0.0001 to 0.00005 in `./configs/baseline.yaml`(line 24)

## 6. Sign Language Gesture

We also conduct experiments on the Sign Language Gesture dataset. First, download the dataset through the link [Google Drive](https://drive.google.com/file/d/12a0mQ_kH7Pk4B2ntb0qg_qGN9tfbwnnu/view?usp=drive_link) and the pretrained weights through the link [Google Drive](https://drive.google.com/file/d/1u8IdnniordVVdmDYkIV5qBMmjuViiLGx/view?usp=drive_link). Then put the files in `./SLG` and unzip them. Then run the command below:
```bash
python train.py
```
SLG is a quite small and simple image dataset, and you can get an accuracy of nearly 100% in less than 20 minutes. Thus, we do not provide our checkpoints. You can modify the gpu device in `train.py`(line 12).
