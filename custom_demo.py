import torch
import argparse
import os
import json
import cv2
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images that end with .jpg or .png
    # img_folder = args.img_folder
    img_folder = '/home/hongsuk/projects/human_in_world/demo_data/input_images/arthur_tyler_pass_by_nov20/cam01'
    img_paths = [img for end in args.file_type for img in Path(img_folder).glob(end)]
    img_paths.sort()

    det_out_dir = '/home/hongsuk/projects/human_in_world/demo_data/input_masks/arthur_tyler_pass_by_nov20/cam01/json_data'

    # Iterate over all images in folder
    dataset_list = []
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # # Detect humans in image
        # det_out = detector(img_cv2)

        # det_instances = det_out['instances']
        # valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        # boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # read bbox
        frame_idx = int(img_path.stem.split('_')[-1])
        bbox_path = Path(det_out_dir) / f'mask_{frame_idx:05d}.json'
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)
        # if value of "labels" key is empty, continue
        if not bbox_data['labels']:
            continue
        else:
            labels = bbox_data['labels']
            # "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 454, "y1": 399, "x2": 562, "y2": 734, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 45, "y1": 301, "x2": 205, "y2": 812, "logit": 0.0}}}
            boxes = np.array([[labels[str(i)]['x1'], labels[str(i)]['y1'], labels[str(i)]['x2'], labels[str(i)]['y2']] for i in range(1, len(labels)+1)])
            person_ids = np.array([labels[str(i)]['instance_id'] for i in range(1, len(labels)+1)])
            # sanity check; if boxes is empty, continue
            if boxes.sum() == 0:
                continue

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, person_ids, frame_idx)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        dataset_list.append(dataset)

    batch_size = 128
    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)
    dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    vis_result_dict = defaultdict(lambda: defaultdict(list)) # key: frame_idx, value: dictionary with keys: 'all_verts', 'all_cam_t'
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))

            # if batch['frame_idx'][n] == -1:
            #     img_path = img_paths[batch_idx*batch_size+n]
            #     img_fn = os.path.splitext(os.path.basename(img_path))[0]
            frame_idx = int(batch['frame_idx'][n])
            person_id = int(batch['personid'][n])
            img_fn = os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]

            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t_full[n]
            vis_result_dict[frame_idx]['all_verts'].append(verts)
            vis_result_dict[frame_idx]['all_cam_t'].append(cam_t)
            vis_result_dict[frame_idx]['all_person_id'].append(person_id)
            vis_result_dict[frame_idx]['img_size'] = img_size[n].tolist()

 
    import pdb; pdb.set_trace()

    # Render front view
    vis = True
    for frame_idx, result_dict in vis_result_dict.items():
        all_verts = result_dict['all_verts']
        all_cam_t = result_dict['all_cam_t']
        img_size = result_dict['img_size']
        if vis and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size, **misc_args)

            # Overlay image
            img_fn = os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]
            img_cv2 = cv2.imread(str(img_paths[frame_idx]))
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
