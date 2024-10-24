import os, time, argparse, h5py, datetime
from tqdm import tqdm
import numpy as np

import openslide

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wsi_datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

from utils.file_utils import save_hdf5, stat_feat_patch_num
from utils.utils import print_network, collate_features


"""
class: features extraction from patches
"""
class featsExtraction(object):
    def __init__(self, feat_to_dir, csv_path, h5_dir, slide_dir, pretrain_model=None, 
                 slide_ext="", auto_skip=False, gpu="0"):
        """
        feat_to_dir:    dir for saving extracted feats
        csv_path:       csv file with slide_id column
        h5_dir:         DIR path to folder containing masks/ patches/ stitches/ subfolders
        slide_dir:      DIR for h5 files data
        pretrain_model: [resnet50, retccl, ctranspath, phikon, uni], use pretrained model for feature extraction
        slide_ext:      slide image suffix extension
        auto_skip
        """
        self.support_pretrain_models = ['resnet50', 'retccl', 'ctranspath', 'phikon', 'phikonV2',
                                        'uni', 'plip', 'conch', 
                                        'gigapath', 'H-optimus-0', 'virchow', 'virchow2']

        if torch.cuda.is_available() and gpu is not None:
            # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
            self.device = torch.device('cuda:'+gpu)
        else:
            self.device = torch.device('cpu')

        assert feat_to_dir is not None, f"directory to save extracted feats data {feat_to_dir}"
        self.featdirdicts = self.create_featsubdirs(feat_to_dir)

        if auto_skip:
            dest_files = os.listdir(self.featdirdicts["pt_feats_subdir"]) # only auto_skip is True, dest files is used
        else:
            dest_files = None
        self.dest_files = dest_files

        # assert csv_path is not None or h5_dir is not None, f"Dir containing coordinate h5 files {h5_dir} or csvpath {csv_path} must be given :)"
        # print("[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*]")
        assert csv_path is not None and os.path.isfile(csv_path), f"csvpath {csv_path} must be given or file do not EXIST :)"
        print('initializing dataset')
        self.bags_dataset = Dataset_All_Bags(csv_path)
        
        assert pretrain_model in self.support_pretrain_models, \
            f"{pretrain_model} is not in the pretrained model name lists --> {self.support_pretrain_models}."
        
        self.model = self.load_model(pretrain_model, device=self.device)

        assert h5_dir is not None and os.path.exists(h5_dir), f"path to folder containing masks/ patches/ stitches/ subfolders {h5_dir} is None or do not EXIST =_=!"
        self.h5_dir = h5_dir
        assert slide_dir is not None and os.path.exists(slide_dir), f"path to folder containing raw slide images {slide_dir} is None or do not EXIST =_=!"        
        self.slide_dir = slide_dir

        assert slide_ext in ['.svs', '.mrxs', '.ndpi'], f"{slide_ext} should be in ['.svs', '.mrxs', '.ndpi']"
        self.slide_ext = slide_ext

    def run(self, batch_size=256, custom_downsample=1, gaussian_blur=False, resize_size=None, target_patch_size=-1, 
            float16 = False, zero_shot=False):
        """
        Perform feature extraction
        batch_size: Size of the batch for processing
        custom_downsample: Downsampling factor relative to the size specified in WsiPatching. 
                   For example, if the patch size in WsiPatching is 256 and this is set to 2, 
                   the actual patch extracted will be resized from 256 to 256//2=128 for feature extraction.
        target_patch_size: Target patch size. If different from the size specified in WsiPatching, 
                   the patch will be resized to this target size. If neither target size nor downsample is specified, 
                   the default is to use the patch size created in WsiPatching without resizing.
        """
        
        total = len(self.bags_dataset)

        for bag_candidate_idx in range(total):
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

            slide_id = self.bags_dataset[bag_candidate_idx].split(self.slide_ext)[0]
            print(slide_id)

            if self.dest_files is not None and slide_id+'.pt' in self.dest_files:
                print('skipped {}'.format(slide_id))
                continue

            h5_file_path = os.path.join(self.h5_dir, 'patches', slide_id+'.h5')
            if not os.path.exists(h5_file_path): # fix BUG: in case of some svs files lack of useful foreground
                print(f"{h5_file_path} do not exist. It may lack of foreground tissue regions. So skip.")
                continue

            slide_file_path = os.path.join(self.slide_dir, slide_id+self.slide_ext)
            output_path = os.path.join(self.featdirdicts['h5_feats_subdir'], slide_id+'.h5')
            
            time_start = time.time()
            self.compute_w_loader(h5_file_path, slide_file_path, output_path,
                        model = self.model, batch_size = batch_size, verbose = 1, 
                        gaussian_blur=gaussian_blur, resize_size=resize_size,
                        custom_downsample=custom_downsample, target_patch_size=target_patch_size, device=self.device)
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
            
            file = h5py.File(output_path, "r")

            features = file['features'][:]
            coords = file['coords'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', coords.shape)

            # In the saved h5 data, there are coordinates marked as [-1, -1], indicating that the corresponding patch is damaged. 
            # These features are only used for positioning. Therefore, when saving the pt data without coordinates, these features need to be removed.
            features = features[file['coords'][:, 0] != -1, :] # Only retain features corresponding to patches not marked as -1
            print('After filtering by coords MARKING; features size: ', features.shape)

            assert features.shape[0] == coords.shape[0], f"shape feature {features.shape[0]} != coords {coords.shape[0]}"

            features = torch.from_numpy(features)
            coords = torch.from_numpy(coords)
            if float16:
                features = features.type(torch.float16)
            
            torch.save(features, os.path.join(self.featdirdicts['pt_feats_subdir'], slide_id+'.pt'))
            torch.save(coords, os.path.join(self.featdirdicts['pt_coords_subdir'], slide_id+'.pt'))

            if zero_shot:
                zeroshot_tissue_idx = file['zeroshot_tissue_idx'][:][file['coords'][:, 0] != -1]

                os.makedirs(self.featdirdicts['pt_feats_subdir']+"_zs_tissueidx", exist_ok=True)
                torch.save(zeroshot_tissue_idx, os.path.join(self.featdirdicts['pt_feats_subdir']+"_zs_tissueidx", slide_id+'.pt'))

                reserve_tissue_flag = [each in [0, 3, 4] for each in zeroshot_tissue_idx] # select specific tissue. 0 tumor, 3 immune infiltrates, 4 gland 
                features = features[reserve_tissue_flag, :]
                print('After ONLY selecting specific tiles; features size: ', features.shape)

                os.makedirs(self.featdirdicts['pt_feats_subdir']+"_zs_feat", exist_ok=True)
                torch.save(features, os.path.join(self.featdirdicts['pt_feats_subdir']+"_zs_feat", slide_id+'.pt'))

    def stat_patch_num(self):
        stat_feat_patch_num(feat_dir=self.featdirdicts['feat_to_dir'], to_csv=True)

    @staticmethod
    def create_featsubdirs(feat_to_dir):
        """
        create subdirs by feat_to_dir
        """
        pt_feats_subdir = os.path.join(feat_to_dir, 'pt_files')
        pt_coords_subdir = os.path.join(feat_to_dir, 'pt_files_coords')
        h5_feats_subdir = os.path.join(feat_to_dir, 'h5_files')

        dirsdict = {'feat_to_dir': feat_to_dir,
                    'pt_feats_subdir': pt_feats_subdir, 
                    'pt_coords_subdir': pt_coords_subdir, 
                    'h5_feats_subdir' : h5_feats_subdir} 
        
        for key, val in dirsdict.items():
            print("mkdir {} : {}".format(key, val))
            os.makedirs(val, exist_ok=True)

        return dirsdict

    @staticmethod
    def load_model(pretrain_model, device="cuda"):
        """
        load model from retccl filepath or imagenet pretrained resnet50 model
        """
        print('loading model checkpoint...')
        if pretrain_model == 'resnet50':
            from models.resnet_custom import resnet50_baseline
            model = resnet50_baseline(pretrained=True)
            model.__info__ = "resnet50"

        elif pretrain_model == "retccl":
            from models.resnet_RetCCL import resnet50
            model = resnet50(num_classes=2,mlp=False, two_branch=False, normlinear=True) # num_classes is random, that's fine. because we will: model.fc = nn.Identity()
            model.fc = nn.Identity()

            pretext_model = torch.load('pretrained_model_weights/CCL_best_ckpt.pth')
            model.load_state_dict(pretext_model, strict=True)
            model.__info__ = "retccl"

        elif pretrain_model == "ctranspath":
            from models.model_swinTrans import ctranspath
            model = ctranspath()

            pretext_model = torch.load('pretrained_model_weights/ctranspath.pth')
            model.load_state_dict(pretext_model['model'], strict=True)
            model.__info__ = "ctranspath"

        elif pretrain_model == "phikon":
            from transformers import ViTModel
            model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False) # will load from huggingface model
            model.__info__ = "phikon"

        elif pretrain_model == "phikonV2":
            from transformers import AutoModel
            model = AutoModel.from_pretrained("owkin/phikon-v2")
            model.__info__ = "phikonV2"

        elif pretrain_model == "uni":
            import timm
            model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)

            pretext_model = torch.load("pretrained_model_weights/UNI/pytorch_model.bin")
            model.load_state_dict(pretext_model, strict=True) # , map_location="cpu"
            model.__info__ = "uni"

        elif pretrain_model == "plip":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained("pretrained_model_weights/plip/")
            model.__info__ = "plip"

        elif pretrain_model == "gigapath": #
            import timm
            model = timm.create_model("vit_giant_patch14_dinov2", img_size=224, patch_size=16, init_values=1e-5, num_classes=0)
            pretext_model = torch.load("pretrained_model_weights/gigapath/pytorch_model.bin")
            model.load_state_dict(pretext_model, strict=True) # , map_location="cpu"
            model.__info__ = "gigapath"

        elif pretrain_model == "H-optimus-0": # H-optimus-0
            import timm
            checkpoint_path = "pretrained_model_weights/H-optimus-0/pytorch_model.bin"
            # model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, 
            #                           init_values=1e-5, dynamic_img_size=False)
            model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, checkpoint_path=checkpoint_path, 
                                      img_size=224, global_pool="token",
                                      init_values=1e-5, num_classes=0, dynamic_img_size=True)
            model.__info__ = "H-optimus-0"
            
        elif pretrain_model == "virchow":
            # issue: https://huggingface.co/paige-ai/Virchow/discussions/4#66be06d055a1210c535870d4
            import timm
            checkpoint_path = "pretrained_model_weights/Virchow/pytorch_model.bin"
            model = timm.create_model("vit_huge_patch14_224", pretrained=False, checkpoint_path=checkpoint_path, 
                                      mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                                      img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375, 
                                      global_pool="",dynamic_img_size=True)
            model.__info__ = "Virchow"

        elif pretrain_model == "virchow2":
            import timm
            checkpoint_path = "pretrained_model_weights/Virchow2/pytorch_model.bin"
            model = timm.create_model("vit_huge_patch14_224", pretrained=False, checkpoint_path=checkpoint_path, 
                                      mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                                      img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375, reg_tokens=4, # reg_tokens=4 is for Virchow2
                                      global_pool="",dynamic_img_size=True)
            model.__info__ = "Virchow2"

        elif pretrain_model == "conch":
            pass

        else:
            raise NotImplementedError
        
        model = model.to(device)
                
        # print_network(model)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        model = [model]

        if pretrain_model in ["plip", ]:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained("pretrained_model_weights/plip/")

            def_types = ["tumor", "adipose", "stroma", "immune infiltrates", "gland", "necrosis or hemorrhage", "background or black", "non"]
            full_texts = ["a H&E image of {} tissue".format(label) for label in def_types]
            model.extend([processor, full_texts])

        elif pretrain_model in ["conch", ]:
            pass

        elif pretrain_model in ["gigapath", ]:
            pass

        return model

    @staticmethod
    def compute_w_loader(file_path, slidewsi_path, output_path, model,
        batch_size = 8, verbose = 0, gaussian_blur=False, resize_size=None,
        custom_downsample=1, target_patch_size=-1, device="cuda"):
        """
        args:
            file_path: directory of bag (.h5 file)
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            gaussian_blur: use  gaussian_blur or not, default False
            custom_downsample: custom defined downscale factor of image patches
            target_patch_size: custom defined, rescaled image size before embedding
        """
        wsi = openslide.open_slide(slidewsi_path)

        dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, 
                                     gaussian_blur=gaussian_blur, resize_size = resize_size,
                                     custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        # x, y = dataset[0]
        kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

        if verbose > 0:
            print('processing {}: total of {} batches'.format(file_path,len(loader)))

        mode = 'w'
        for batch, coords in tqdm(loader, total=len(loader)):
            with torch.no_grad():	
                batch = batch.to(device, non_blocking=True)
                
                if model[0].__class__.__name__ == 'CLIPModel': # PLIP pretrained model
                    processor = model[1]
                    full_texts = model[2]
                    inputs = processor(text=full_texts, images=None, return_tensors="pt", padding=True).to(device) # only process text
                    inputs.update({"pixel_values": batch}) # udpate batch to `pixel_values`

                    outputs = model[0](**inputs)

                    no_norm_features = model[0].visual_projection(outputs.vision_model_output[1]).detach() # get the no normalized features
                    features = no_norm_features # we only get the image embedding, so we do not need to normalize it
                    # norm_features = outputs.image_embeds.detach()  # (1, 512) shape, **has been normalized**，与下述等价

                    # image_embeddings = plip_model.get_image_features(inputs['pixel_values']).cpu().detach().numpy()
                    # image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

                    probs = outputs.logits_per_image.softmax(dim=1)   # this is the image-text similarity score and softmax probs
                    tissue_type_idx = probs.argmax(dim=1).cpu().numpy() 
                    # pred_cls_type = [full_texts[each] for each in tissue_type_idx]

                elif model[0].__class__.__name__ == 'ViTModel': # Phikon pretrained model
                    batch = {"pixel_values": batch}
                    features = model[0](**batch).last_hidden_state[:, 0, :]  # (1, 768) shape
                    tissue_type_idx = np.array([0]*len(batch)) # no zero shot tissue type 

                elif model[0].__info__ == "phikonV2" and model[0].__class__.__name__ == 'Dinov2Model': # PhikonV2 pretrained model
                    batch = {"pixel_values": batch}
                    features = model[0](**batch).last_hidden_state[:, 0, :]  # (1, 1024) shape
                    tissue_type_idx = np.array([0]*len(batch)) # no zero shot tissue type 

                elif model[0].__info__ == "H-optimus-0" and model[0].__class__.__name__ == 'VisionTransformer': #  h-optimus-0 pretrained model
                    features = model[0](batch) # (n, 1536) shape
                    assert features.shape[-1] == 1536
                    tissue_type_idx = np.array([0]*len(batch)) # no zero shot tissue type 

                else:
                    features = model[0](batch)
                    tissue_type_idx = np.array([0]*len(batch)) # no zero shot tissue type 
                
                if model[0].__info__ == "Virchow": # we cat class_token and patch_tokens for Virchow pretrained model 
                    class_token = features[:, 0]    # size: 1 x 1280
                    patch_tokens = features[:, 1:]  # size: 1 x 256 x 1280
                    # concatenate class token and average pool of patch tokens
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

                elif model[0].__info__ == "Virchow2": # we cat class_token and patch_tokens for Virchow pretrained model 
                    class_token = features[:, 0]    # size: 1 x 1280
                    patch_tokens = features[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
                    # concatenate class token and average pool of patch tokens
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
                
                else:
                    pass

                features = features.cpu().numpy()

                asset_dict = {'features': features, 'coords': coords, 'zeroshot_tissue_idx': tissue_type_idx}
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'
        

def set_args():
    parser = argparse.ArgumentParser(description='Features Extraction')
    parser.add_argument('--h5_dir', type=str, default=None, help='(better absolute) DIR for h5 files data')
    parser.add_argument('--slide_dir', type=str, default=None, help='(better absolute) DIR for raw image slides')

    parser.add_argument('--csv_path', type=str, default=None, help='csv file with slide_id column')
    parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model name for feature extraction')
    
    parser.add_argument('--feat_to_dir', type=str, default=None, help='dir for saving extracted feats')

    parser.add_argument('--slide_ext', type=str, default= '.svs', help='slide image suffix extension')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--custom_downsample', type=int, default=1, help='')
    parser.add_argument('--target_patch_size', type=int, default=-1, help='')
    # parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--resize_size', type=int, default=None)

    parser.add_argument('--gaussian_blur', default=False, action='store_true')
    parser.add_argument('--auto_skip', default=False, action='store_true')
    parser.add_argument('--float16', default=False, action='store_true')
    parser.add_argument('--zeroshot', default=False, action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    print(datetime.datetime.now())
    args = set_args()

    featsextract = featsExtraction(args.feat_to_dir,
                                   args.csv_path, args.h5_dir, args.slide_dir, 
                                   args.pretrain_model, 
                                   args.slide_ext, args.auto_skip, gpu="0")

    featsextract.run(args.batch_size, args.custom_downsample, args.gaussian_blur, args.resize_size,
                     args.target_patch_size, args.float16, zero_shot=args.zeroshot)

    featsextract.stat_patch_num()
    print(datetime.datetime.now())
