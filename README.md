# Recurrence without Recurrence: Stable Video Landmark Detection with DEQs

This repo includes inference code for our Landmark Deep Equilibrium Model network (LDEQ), which was developed during an internship at **Nvidia** by myself, Pavlo Molchanov, Arash Vahdat, Hongxu Yin and Jan Kautz.

The weights we provide here were reproduced on different hardware than the hardware used for the paper experiments, and results may be slightly different. 

## TLDR
This work uses deep equilibrium models to add a form of recurrence at test time, without having access to a recurrent loss at train time. This can be used to improve temporal coherence in video landmark detection when the model is trained on still images.

Please check out this video for a demo:

[![demo](https://img.youtube.com/vi/8Mmpc_-oP6w/0.jpg)](https://www.youtube.com/watch?v=8Mmpc_-oP6w)


## Environment setup

```
conda create -y -n ldeq python=3.8
conda activate ldeq
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python==4.6.0.66 matplotlib==3.5.1 scipy==1.7.3 torchinfo==1.7.0 pandas
```
I haven't tried other versions of these librairies so use them at your own risk.
NB: if `opencv` gives weird errors, `pip uninstall opencv-python` and then `pip install opencv-python-headless==3.5.5.64`


## Test LDEQ on WFLW
1) Download the WFLW dataset that was used for training and testing LDEQ on WFLW. We used the precropped version from the [HIH repo](https://github.com/starhiking/HeatmapInHeatmap) and it can be downloaded there (WFLW.zip).
2) (optional) Check dataset loading is good by running datasets/WFLW/dataset.py, after updating path there
3) Download the LDEQ trained weights [here](https://drive.google.com/drive/folders/1r0NJBXtAW2mIT30bPw83ZgDuPgIUdN9K?usp=share_link). This particular model uses the Anderson acceleration solver.
4) run `python test_LDEQ_WFLW.py --landmark_model_weights /path/to/final.pth.tar --dataset_path /path/to/WFLW --workers 4 --batch_size 32`


## Test LDEQ on WFLW-V

### WFLW-V download

### Option 1 (Easy)
Someone has kindly uploaded the official dataset in his name [here](https://github.com/polo5/LDEQ_RwR/issues/2). I made sure this is the same version I used for the results in the paper

### Option 2 (Hard)
While all videos used in WFLW-V have creative commons licences, nvidia has stricter internal privacy rules.
As such, while I provide an official download link for landmark and bboxes annotations, I can only provide a python script to download and crop the videos yourself from youtube.
This is limiting because some videos may disappear in the future. If you want to preprocess the dataset yourself do the following:

1. Download official bboxes annotations [here](https://drive.google.com/file/d/17r2w3abzUsPlsDfqYOsjGGkTp2nvHxds/view?usp=sharing)
2. Download official landmark annotations [here](https://drive.google.com/file/d/1ITmlgXydTogFa5HkE0NvluKAxA4LCloj/view?usp=sharing)
3. `pip install pytube==12.1.2`
3. Run `python utils/download_WFLW_V.py --output_folder /path/to/WFLW_V --n_processes 16`.

Note that pytube may ask you to sign in to your google account and enter some code when you first run this script, depending on the platform you're using.
If things freeze after entering the code (due to multiprocessing), restart the python script.
You may need to restart this script several times if your connection gets closed by youtube servers.

The folder structure should look like:
```
WFLW_V
      |
       bboxes
            |
            -3rDiJYQ6CQ.npy
            -3uh4B-qDcs.npy
            ...
       landmarks
            |
            -3rDiJYQ6CQ.npy
            -3uh4B-qDcs.npy
            ...
       videos
            |
            -3rDiJYQ6CQ.mp4
            -3uh4B-qDcs.mp4
            ...

```


### WFLW-V inference

1) Download the LDEQ trained weights [here](https://drive.google.com/drive/folders/1Jo4BCSSZTBM4Ms3Q9L7dyN4QktbpEM5e?usp=share_link). This particular model uses the fixed point iteration solver to go faster. These weights are different that the ones above because they were obtained with more augmentations (in particular more crops). This helps performance on WFLW_V but worsens performance on WFLW.
2) Edit/pass in args in `test_LDEQ_WFLW_V.py` and run it. You can uncomment the line `solver.test_all_videos_sequential(RWR=args.rwr, plot=True)` in the `main()` funciton if you want to visualize video predictions + ground truth.

## Training
nvidia won't release it, sorry:(

## Cite

If you find this work useful, please consider citing us:

```
@InProceedings{Micaelli_2023_CVPR,
    author    = {Micaelli, Paul and Vahdat, Arash and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
    title     = {Recurrence Without Recurrence: Stable Video Landmark Detection With Deep Equilibrium Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22814-22825}
}
```
