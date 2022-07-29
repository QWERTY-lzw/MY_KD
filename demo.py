from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/ship/yolov3p_d53_320_273e_ship8.py'
checkpoint_file = 'work_dirs/yolov3p_d53_320_273e_ship8/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


for i in range(1,10):
    # test a single image and show the results
    img = f'data/ship-comp-committee-new/ImageSets/trainimg000{i}.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, [img])
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=f'result{i}.jpg', bbox_color='citys')