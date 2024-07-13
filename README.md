## Adaptive Prior and Long-Range Dependency-Based Learners for Image Inpainting
This is the official PyTorch implementation of APLRL.

### Datasets

**Image Dataset.** We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder), and [Places2](http://places2.csail.mit.edu/) datasets, which are widely adopted in the literature. 

**Mask Dataset.** Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

## Getting Started
Download the pre-trained models using the following links and copy them under `./snapshots` directory.

[Places2](https://drive.google.com/drive/folders/158ch9Psjop0mQEdeIp9DKjrYIGTDsZKN) | [CelebA](https://drive.google.com/drive/folders/13JgMA5sKMYgRwHBp4f7PBc5orNJ_Cv-p) | [Paris-StreetView](https://drive.google.com/drive/folders/1hMGVz6Ck3erpP3BRNzG90HNCJl85kveN)
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
