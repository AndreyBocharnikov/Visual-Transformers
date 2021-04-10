# Visual-Transformers
Unofficial implimentation of Visual Transformers: Token-based Image Representation and Processing for Computer Vision.

## Usage:
`python main.py task_mode learning_mode data --model --weights`, where:
* `task_mode`: `classification` or `semantic_segmentation` for corresponding task
* `learning_mode`: `train` to train `--model` from scratch, `test` to validate `--model` with `--weights` on validation data.
* `data`: path to dataset, in case of classification should be path to image net, in case of semantic segmentation to coco.
* `--model`:   
  ○ classification: `ResNet18` or `VT_ResNet18` (will be used by default).  
  ○ semantic segmentation: `PanopticFPN` or `VT_FPN` (will be used by default).  
* `--weights` must be provided if `learning_mode` equals to `test`, won't be used in `train` mode.
* `--from_pretrained` uses to continue training from some point, should be `state_dict` that contains `model_state_dict`, `optimizer_state_dict` and `epoch`.

## Results:  
* final metrics and losses after 15 and 5 epochs of classification and semantic segmentation respectively:
<table>
  <tr>  
    <td>    
    
|                      | ResNet18 | VT-ResNet18 |
|----------------------|----------|-------------|
| Training accuracy    | 0.664675 |  0.672889   |
| Validation accuracy  | 0.691541 |  0.696929   |
|                      |          |             |
| Training loss        | 1.312150 |  1.249382   |
| Validation loss      | 1.173559 |  1.114401   |

   
   </td><td>  
      
|                 | Panoptic FPN |  VT-FPN  |
|-----------------|--------------|----------|
| Training mIOU   |   8.0968     | 7.0343   |
| Validation mIOU |   4.3148     | 3.2351   |
|                 |              |          |
| Training loss   |   2.044084   | 2.068598 |
| Validation loss |   2.101253   | 2.120928 |


   </td>  
  </tr>
</table>

* loss and metric curves of classification and semantic segmentation:


cross entropy loss         |  accuracy
:-------------------------:|:-------------------------:
![classification loss](https://user-images.githubusercontent.com/41442977/114195120-daaee980-9958-11eb-97b2-b4b91908d159.png)  |  ![classification metric](https://user-images.githubusercontent.com/41442977/114195759-6cb6f200-9959-11eb-953b-69f66788110e.png)

pixel-wise cross entropy loss   |  mIOU
:------------------------------:|:-------------------------:
![semantic segmentation_loss](https://user-images.githubusercontent.com/41442977/114266799-f077e980-9a00-11eb-9804-1a386e29729c.png)  | ![semantic segmentation mIOU](https://user-images.githubusercontent.com/41442977/114266809-071e4080-9a01-11eb-98e9-553463db2c7c.png)





* Efficiency and parameters

|              | Params (M) | FLOPs | Forward pass (s) |
|--------------|------------|-------|------------------|
| ResNet18     |    11.2    |       |       0.016      |
| VT-ResNet18  |    12.7    |       |       0.02       |
|              |            |       |                  |
| Panoptic FPN |    16.4    |       |       0.08       |
| VT-FPN       |    40.3    |       |       0.062      |

## Weights:
* classification: [ResNet18](https://drive.google.com/file/d/102_XFdm9mnQbZVbw8ChoywvxG3IOhXCh/view?usp=sharing), [VT-ResNet18](https://drive.google.com/file/d/1-7zrZD2TekIIcAa4im0i5fi31ZG90sP9/view?usp=sharing)
* semantic segmentation: [Panoptic FPN](https://drive.google.com/file/d/1hEYHuaWhc-JqpPyjdr86kMorkDGX3gIN/view?usp=sharing), [VT-FPN](https://drive.google.com/file/d/1-GUY6KdQBF5q4VFrdb79XyROqE70lbHv/view?usp=sharing)
