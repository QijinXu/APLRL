## Adaptive Prior and Long-Range Dependency-Based Learners for Image Inpainting
This is the official PyTorch implementation of APLRL.
## Prerequisites
- Python 3.7
- PyTorch 1.2
- NVIDIA GPU + CUDA cuDNN
### Installation

- Install python requirements:

```
pip install -r requirements.txt
```
### Datasets

**Image Dataset.** We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder), and [Places2](http://places2.csail.mit.edu/) datasets, which are widely adopted in the literature. 

**Mask Dataset.** Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

## Getting Started
Download the pre-trained models using the following links and copy them under `./snapshots` directory.

[Places2](https://drive.google.com/file/d/1uae6Gl6vC-7y6NvGi-eF1lfGR8d-ZJgA/view?usp=drive_link) | [CelebA](https://drive.google.com/file/d/1t0QXRx0PZqYmFlwhefg48q3qiXCet__J/view?usp=drive_link) | [Paris-StreetView](https://drive.google.com/file/d/1MTtB9M7bpjKQ_i8cywfGR6V0WBU_R4WR/view?usp=drive_link)
### Testing

To test the model, you run the following code.

```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```

## Acknowledgements
Some of the code of this repo is borrowed from:

- [CTSDG](https://github.com/xiefan-guo/ctsdg)

## Citation

If any part of our paper and repository is helpful to your work, please generously cite with:

```
@ARTICLE{Cao2025TCSVT,
  author={Cao, Feilong and Xu, Qijin and Ye, Hailiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Adaptive Prior and Long-Range Dependency-Based Learners for Image Inpainting}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2025.3574529}}
```
