# Visual-Transformers
Unofficial implimentation of Visual Transformers: Token-based Image Representation and Processing for Computer Vision.

## Usage:
`python main.py task_mode learning_mode data --model --weights`, where:
* `task_mode`: `classification` or 'semantic_segmentation' for corresponding task
* `learning_mode`: `train` to train `--model` from scratch, `test` to validate `--model` with `--weights` on validation data.
* `data`: path to dataset, in case of classification should be path to image net, in case of semantic segmentation to coco.
* `--model`:   
  ○ classification: `ResNet18` as a baseline, `VT_ResNet18` (will be used by default).  
  ○ semantic segmentation: TODO as a baseline, TODO (will be used by default).  
* `--weights` must be provided if `learning_mode` equals to `test`, won't be used in `train` mode.
* `--from_pretrained` uses to continue training from some point, should be `state_dict` that contains `model_state_dict`, `optimizer_state_dict` and `epoch`.

## Results:

## Weights:

  
