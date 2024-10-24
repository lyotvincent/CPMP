from __future__ import print_function

import numpy as np
import pandas as pd

import torch
import os, h5py, yaml, argparse

from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model
from utils.heatmap_utils import initialize_wsi, drawHeatmap, feat_extraction_tiles, load_pretrain_model, infer_single_slide, load_params
from utils.file_utils import save_hdf5
import utils.config_utils as cfg_utils

from wsi_core.batch_process_utils import initialize_df
from wsi_core.wsi_utils import sample_rois


#  Parse arguments
def get_parser():
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--configs', type=str, default=None, help='config file for heatmap parameters')
    parser.add_argument('--opts', help='see wsi_heatmap_params.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.configs is not None, "Please provide heatmap config file for parameters."
    cfg = cfg_utils.load_cfg_from_cfg_file(args.configs)
    if args.opts is not None:
        cfg = cfg_utils.merge_cfg_from_list(cfg, args.opts)

    print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] HEATMAP CONFIG PARAMS:\n{cfg}")
    return cfg



if __name__ == "__main__":
    args = get_parser()

    step_size = tuple((np.array(args.patch_size) * (1-args.overlap)).astype(int))
    print('patch_size: {}, with {:.2f} overlap, step size is {}'.format(args.patch_size, args.overlap, step_size))

    # defined tissue segmentation parameters
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': True, 
                        'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':13.0, 'a_h': 8.0, 'max_n_holes':10} 
    def_vis_params = {'vis_level': -1, 'line_thickness': 50, 'draw_grid': True}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    # pre-set tissue segmentation/patching parameters file
    if args.preset is not None:
        preset_df = pd.read_csv(args.preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    # read the slide list to process
    if args.process_list is not None: # first priority, a csv file with slide names
        df = pd.read_csv(args.process_list)
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=True)

    else:
        if args.data_dir is not None: # second priority, a path of slides
            if isinstance(args.data_dir, list):
                slides = []
                for data_dir in args.data_dir:
                    slides.extend(os.listdir(data_dir))
            else:
                slides = sorted(os.listdir(args.data_dir))
            slides = [slide for slide in slides if args.slide_ext in slide]
            df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=True)

        else:
            if args.spec_slide is not None and isinstance(args.spec_slide, list): # third priority, a list of slides
                slides = args.spec_slide
                
                df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=True)
            else:                
                raise NotImplementedError

    
    if args.data_csv_info is not None:
        data = pd.read_csv(args.data_csv_info)
        slide_ids, label = data["slide_id"], data["MMPrisk"].tolist()
        labels = [args.label_dict[label[idx]] for idx in range(len(label))]

        slide_id_label_dict = dict(zip(slide_ids, labels))
            
        for idx in range(len(df)):
            df.loc[idx, 'label'] = slide_id_label_dict[df.loc[idx, 'slide_id']] 
    else:
        df['label'] = 'Unspecified'

    mask = df['status'] != 'success'
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))


    if torch.cuda.is_available() and args.gpu is not None:
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device('cuda:'+ str(args.gpu))
    else:
        device = torch.device('cpu')


    model =  initiate_model(args, ckpt_path)
    model = model.to(device)


    feature_extractor = load_pretrain_model(args.tile_model, device=device) # 

    label_dict =  args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[idx]: class_labels[idx] for idx in range(len(class_labels))} 


    os.makedirs(args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': args.patch_size, 'step_size': step_size,
                        'custom_downsample':args.custom_downsample, 'level': args.patch_level, 'use_center_shift': args.use_center_shift}


    # process_stack['slide_id'] = process_stack['slide_id'].astype(int).astype(str)
    for idx in range(len(process_stack)):
        slide_name = process_stack.loc[idx, 'slide_id']
        if args.slide_ext not in slide_name:
            slide_name+=args.slide_ext
        print('\nprocessing: {}  ({}/{})'.format(slide_name, idx+1, total))

        try:
            label = process_stack.loc[idx, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label


        r_slide_save_dir = os.path.join(args.raw_save_dir, args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        p_slide_save_dir = os.path.join(args.raw_save_dir, args.save_exp_code, str(grouping))

        if args.use_roi:
            x1, x2 = process_stack.loc[idx, 'x1'], process_stack.loc[idx, 'x2']
            y1, y2 = process_stack.loc[idx, 'y1'], process_stack.loc[idx, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(args.data_dir, str):
            slide_path = os.path.join(args.data_dir, slide_name)
        elif isinstance(args.data_dir, dict):
            data_dir_key = process_stack.loc[idx, args.data_dir_key]
            slide_path = os.path.join(args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[idx], seg_params)
        filter_params = load_params(process_stack.loc[idx], filter_params)
        vis_params = load_params(process_stack.loc[idx], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('>>>>>>>>>>>>>>>>>>>> Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)

        wsi_ref_downsample = wsi_object.level_downsamples[args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(args.patch_size) * np.array(wsi_ref_downsample) * args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.level_count - 1
            vis_params['vis_level'] = best_level

        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        coords_path = os.path.join(r_slide_save_dir, slide_id+'_coords.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')

        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            try:
                wsi_object = feat_extraction_tiles(wsi_object=wsi_object, feature_extractor=feature_extractor, 
                                            batch_size=args.batch_size, feat_save_path=h5_path, device=device, **blocky_wsi_kwargs)
            except:
                process_stack.loc[idx, 'status'] = 'failed_feat_extraction'
                torch.cuda.empty_cache()
                continue
        
        torch.cuda.empty_cache()

        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            coords = torch.tensor(file['coords'][:])
            torch.save(features, features_path)
            torch.save(coords, coords_path)
            file.close()

        # load features 
        features = torch.load(features_path)
        coords = torch.load(coords_path) // (args.patch_size[0]*args.custom_downsample)

        process_stack.loc[idx, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)

        try:
            Y_prob, Y_hat, scores = infer_single_slide(model, features, coords, args.discard_ratio, args.best_thresh, device=device)
        except:
            process_stack.loc[idx, 'status'] = 'failed_inference'
            del features, coords
            torch.cuda.empty_cache()
            continue

        del features, coords
        torch.cuda.empty_cache()
        
        print('Label: {}, Y_hat: {}, Y_prob: {}'.format(label, reverse_label_dict[Y_hat], "{:.4f}".format(Y_prob[0])))
        process_stack.loc[idx, 'Prediction'] = reverse_label_dict[Y_hat]
        process_stack.loc[idx, 'Prob'] = Y_prob[0]

        coords = torch.load(coords_path).numpy()
        asset_dict = {'attention_scores': scores, 'coords': coords}
        block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')


        heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, vis_level=args.vis_level, 
                            cmap=args.cmap, alpha=args.alpha, overlap=args.overlap, blur=args.blur,
                            use_holes=True, binarize=args.binarize, blank_canvas=args.blank_canvas, thresh=args.binary_thresh,
                            patch_size = vis_patch_size, custom_downsample=args.custom_downsample,
                            convert_to_percentiles=True)
        heatmap.save(os.path.join(p_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
        del heatmap

        samples = args.samples # save samples of patches with highest attention scores
        for sample in samples:
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, int(Y_hat))
                sample_save_dir =  os.path.join(args.raw_save_dir, args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for smp_idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), args.patch_level,
                                                    list(np.array(args.patch_size)*args.custom_downsample)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(smp_idx, slide_id, s_coord[0], s_coord[1], s_score)))

        process_stack.loc[idx, 'status'] = 'success'
        if args.save_orig:
            if args.vis_level >= 0:
                vis_level = args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), args.save_ext)

            heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=args.custom_downsample)
            if args.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

        process_stack.loc[idx, 'process'] = 0
        if args.process_list is not None:
            process_stack.to_csv(os.path.join(args.raw_save_dir, '{}.csv'.format(args.process_list.replace('.csv', ''))), index=False)
        else:
            process_stack.to_csv(os.path.join(args.raw_save_dir, '{}.csv'.format(args.save_exp_code)), index=False)
        

    with open(os.path.join(args.raw_save_dir, args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)    