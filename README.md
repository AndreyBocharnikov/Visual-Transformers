# Visual-Transformers
Unofficial implimentation of Visual Transformers: Token-based Image Representation and Processing for Computer Vision.

## Usage:
`python main.py task_mode learning_mode data --model --weights`, where:
* `task_mode`: `classification` or `semantic_segmentation` for corresponding task
* `learning_mode`: `train` to train `--model` from scratch, `test` to validate `--model` with `--weights` on validation data.
* `data`: path to dataset, in case of classification should be path to image net, in case of semantic segmentation to coco.
* `--model`:   
  ○ classification: `ResNet18` as a baseline, `VT_ResNet18` (will be used by default).  
  ○ semantic segmentation: TODO as a baseline, TODO (will be used by default).  
* `--weights` must be provided if `learning_mode` equals to `test`, won't be used in `train` mode.
* `--from_pretrained` uses to continue training from some point, should be `state_dict` that contains `model_state_dict`, `optimizer_state_dict` and `epoch`.

## Results:  
* final metrics and losses after 15 and 5 epochs of classification and semantic segmentation respectively:
<table>
  <tr>  
    <td>    
    
|                      | ResNet18 | VT-ResNet18 |
|----------------------|----------|-------------|
| Training accuracy    |          |             |
| Validation accuracy  |          |             |
|                      |          |             |
| Training loss        |          |             |
| Validation loss      |          |             |

   
   </td><td>  
      
|                 | Panoptic FPN | VT-FPN |
|-----------------|--------------|--------|
| Training mIOU   |              |        |
| Validation mIOU |              |        |
|                 |              |        |
| Training loss   |              |        |
| Validation loss |              |        |


   </td>  
  </tr>
</table>

* loss and metric curves of classification and semantic segmentation:


cross entropy loss         |  accuracy
:-------------------------:|:-------------------------:
![classification loss](https://user-images.githubusercontent.com/41442977/114195120-daaee980-9958-11eb-97b2-b4b91908d159.png)  |  ![classification metric](https://user-images.githubusercontent.com/41442977/114195759-6cb6f200-9959-11eb-953b-69f66788110e.png)

pixel-wise cross entropy loss   |  mIOU
:------------------------------:|:-------------------------:




* Efficiency and parameters

|              | Params | FLOPs | Forward pass (s) |
|--------------|--------|-------|------------------|
| ResNet18     |        |       |       0.016      |
| VT-ResNet18  |        |       |       0.02       |
|              |        |       |                  |
| Panoptic FPN |        |       |                  |
| VT-FPN       |        |       |                  |

## Weights:
* classification: [baseline](https://drive.google.com/file/d/1-7zrZD2TekIIcAa4im0i5fi31ZG90sP9/view?usp=sharing), [VT_ResNet](TODO)
* semantic segmentation: [baseline](TODO), [VT](TODO)
