import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import os
import pandas as pd
from utils.utils import *

from wsi_datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

from models.model_CLAM import CLAM_MB, CLAM_SB
from models.model_ABMIL import ABMIL
from models.transformer import Transformer
from models.transformer_utils import rollout


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object



def load_pretrain_model(pretrain_model, device="cuda"):
    """
    load model from retccl filepath or imagenet pretrained resnet50 model
    """
    print('loading model checkpoint...')
    if pretrain_model == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True)

    elif pretrain_model == "retccl":
        from models.resnet_RetCCL import resnet50
        model = resnet50(num_classes=2,mlp=False, two_branch=False, normlinear=True) # num_classes is random, that's fine. because we will: model.fc = nn.Identity()
        model.fc = nn.Identity()

        pretext_model = torch.load('pretrained_model_weights/CCL_best_ckpt.pth')
        model.load_state_dict(pretext_model, strict=True)

    elif pretrain_model == "ctranspath":
        from models.model_swinTrans import ctranspath
        model = ctranspath()

        pretext_model = torch.load('pretrained_model_weights/ctranspath.pth')
        model.load_state_dict(pretext_model['model'], strict=True)

    elif pretrain_model == "phikon":
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False) # will load from huggingface model

    elif pretrain_model == "uni":
        import timm
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)

        pretext_model = torch.load("pretrained_model_weights/UNI/pytorch_model.bin")
        model.load_state_dict(pretext_model, strict=True) # , map_location="cpu"

    elif pretrain_model == "plip":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained("pretrained_model_weights/plip/")

    elif pretrain_model == "conch":
        pass

    else:
        raise NotImplementedError
    
    model = model.to(device)
            
    # print_network(model)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.eval()
    model = [model]

    if pretrain_model in ["plip", "conch"]:
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained("pretrained_model_weights/plip/")

        def_types = ["tumor", "adipose", "stroma", "immune infiltrates", "gland", "necrosis or hemorrhage", "background or black", "non"]
        full_texts = ["a H&E image of {} tissue".format(label) for label in def_types]
        model.extend([processor, full_texts])

    return model



@torch.no_grad()
def feat_extraction_tiles(wsi_object, feature_extractor=None, batch_size=512,  feat_save_path=None, device="cpu", **wsi_kwargs):    

    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('feat_extraction >>>>> total number of patches to process: ', len(roi_dataset))

    mode = "w"
    for idx, (roi, coords, _) in tqdm(enumerate(roi_loader), total=len(roi_loader)):
        roi = roi.to(device)
        
        if feature_extractor[0].__class__.__name__ == 'ViTModel': # Phikon pretrained model
            roi = {"pixel_values": roi}
            features = feature_extractor[0](**roi).last_hidden_state[:, 0, :]  # (1, 768) shape
        else:
            features = feature_extractor[0](roi)

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords.numpy()}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return wsi_object



@torch.no_grad()
def infer_single_slide(model, features, coords, discard_ratio=0.9, best_thresh=0.5, device='cpu'):
    features = features.to(device)
    coords = coords.to(device)
    if isinstance(model, (CLAM_SB, CLAM_MB, ABMIL)):
        _, Y_prob, Y_hat, A, results_dict = model(features, train=False)
        Y_hat = Y_hat.item()

        if isinstance(model, (CLAM_MB,)):
            A = A[Y_hat]

    elif isinstance(model, Transformer):
        Y_prob, attentions_matrix, _ = model(features, coords=coords, register_hook=True)
        Y_hat = int((Y_prob[0] > best_thresh).item())

        del features, coords
        torch.cuda.empty_cache()

        try:
            A = rollout(attentions_matrix, discard_ratio=discard_ratio, head_fusion='max')
        except:
            print("Transfer to CPU for rollout")
            attentions_matrix = [am.to("cpu") for am in attentions_matrix]
            A = rollout(attentions_matrix, discard_ratio=discard_ratio, head_fusion='max')
            A = A.to(device)
        
        A = A / A.max()
        del attentions_matrix
        torch.cuda.empty_cache()
    
    else:
        raise NotImplementedError

    A = A.view(-1, 1).cpu().numpy()

    return Y_prob[0].cpu().numpy(), Y_hat, A