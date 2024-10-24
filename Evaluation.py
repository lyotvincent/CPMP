import torch
import os, argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, roc_curve, precision_recall_curve

from utils.train_utils import _init_model
from utils.general_utils import set_seed_torch

from models.transformer_utils import rollout


# load data from csv path
def load_data(csv_path, label_mapping=None, filter_dict = None):
    data = pd.read_csv(csv_path)

    filter_mask = np.full(len(data), True, bool)
    # assert 'label' not in filter_dict.keys()
    for key, val in filter_dict.items():
        mask = data[key].isin(val)
        filter_mask = np.logical_and(filter_mask, mask)
    data = data[filter_mask]

    slide_ids, label = data["slide_id"], data["MMPrisk"]
    label = label.tolist()
    labels = [label_mapping[label[idx]] for idx in range(len(label))]

    data_dict = dict(zip(slide_ids, labels))
    return data_dict


def load_model(params_dict, trained_pt_loc=None, fold=0, device=None):
    model = _init_model(params_dict["model_type"], params_dict["model_size"], params_dict["encoding_dim"], 
                        params_dict["drop_out"], params_dict["n_classes"], 
                        device=device, agent_num = params_dict['agent_token_num'])

    if trained_pt_loc is None:
        raise FileNotFoundError
    else:
        model.load_state_dict(torch.load(os.path.join(trained_pt_loc, "s_{}_checkpoint.pt".format(fold))))
        print("\nLoading model state dict from {}.\n ".format(os.path.join(trained_pt_loc, "s_{}_checkpoint.pt".format(fold))))
    
    return model.eval()


def read_params_file(filename):
    with open(filename, 'r') as file:
        js = file.read()
    dic = json.loads(js)   
    print(dic)
    return dic


@torch.no_grad()
def load_inference(slide_id_list, slide_label_list, feats_path, tidx, kidx, model, patch_num=None, **params_dict):
    patient_results = []
    all_probs = []
    all_preds = []
    all_labels_used = []
    all_cases_embedding = []

    params_dict["num_perslide"] = patch_num

    with tqdm(total= len(slide_id_list)) as _tqdm: # 使用需要的参数对tqdm进行初始化
        for sidx, slide_id in enumerate(slide_id_list):
            slide_id = str(slide_id)+".svs"

            _tqdm.set_postfix(slide_id="{}".format(slide_id))

            if slide_id.split('.')[-1] in ["svs", "mrxs"]:
                slide_id = '.'.join(slide_id.split('.')[:-1])
            
            full_path = os.path.join(feats_path, 'pt_files', '{}.pt'.format(slide_id))
            
            if not os.path.exists(full_path): # This slide will not be used if there is no tissue area
                continue

            features = torch.load(full_path) # load feats data
            coords = torch.load(os.path.join(feats_path, 'pt_files_coords', '{}.pt'.format(slide_id)))//512


            if params_dict["zeroshot_idx_dir"] is not None:
                zs_tissue_idx = torch.load(os.path.join(params_dict["zeroshot_idx_dir"], '{}.pt'.format(slide_id)))
                reserve_tissue_flag = [each in [0, 1, 2, 3, 4, 5] for each in zs_tissue_idx] # select specific tissue. 0 tumor, 3 immune infiltrates, 4 gland 
                features = features[reserve_tissue_flag, :]
                coords = coords[reserve_tissue_flag, :]


            Y_prob = 0.0
            for _ in range(1):
                if params_dict["num_perslide"] is not None and params_dict["num_perslide"] < len(features):
                    rnd_idx = np.random.choice(len(features), params_dict["num_perslide"])
                    features = features[rnd_idx, :]
                    coords = coords[rnd_idx, :]

                features = features.to(device).type(torch.float32)
                coords = coords.to(device).type(torch.float32)
                
                if model.__class__.__name__ in ["ABMIL", "CLAM_SB", "TransMIL"] or model.mlp_head[1].out_features == 2:
                    _, Y_prob_tmp, _, _, res_dict = model(features, coords=coords)
                    Y_prob_tmp = Y_prob_tmp[:, 1]
                else:
                    Y_prob_tmp, _, res_dict = model(features, coords=coords, register_hook=False)
                # inst_attn_scores = rollout(attentions_matrix, discard_ratio=0.9, head_fusion='max')
                torch.cuda.empty_cache()

                    # Y_prob_tmp = torch.sigmoid(model.mlp_head(model.norm(model.projection(sub_features.to(device).type(torch.float32).unsqueeze(0)).squeeze(0))))
                    # res_dict = {'embedding': model.projection(sub_features.to(device).type(torch.float32).unsqueeze(0))}
                Y_prob += Y_prob_tmp

            embedding = res_dict['embedding'].cpu().numpy().squeeze()

            probs = Y_prob.cpu().numpy().squeeze(0)
            all_probs.append(probs)
            all_labels_used.append(slide_label_list[sidx])
            all_cases_embedding.append(embedding)

            patient_results.append({'slide_id': slide_id, 'prob': probs, 'label': slide_label_list[sidx]})
            
            _tqdm.update(1)
    
    metric_res_dict = {"time": tidx, "fold": kidx}

    auc = roc_auc_score(all_labels_used, np.array(all_probs))
    metric_res_dict.update({"auc": auc})

    fpr, tpr, thresholds = roc_curve(all_labels_used, np.array(all_probs))
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    metric_res_dict.update({"best thresh": optimal_threshold})

    all_preds = (np.array(all_probs) > optimal_threshold).astype(int)

    eval_res_dict = classification_report(all_labels_used, np.array(all_preds), 
                                        target_names=['neg', 'pos'], output_dict=True)

    for key, vals in eval_res_dict.items():
        if type(vals) is dict:
            for sub_key, sub_vals in vals.items():
                print(f"{sub_key}: {sub_vals}")
                metric_res_dict[key+"_"+sub_key] = sub_vals

        else:
            print(f"{key}: {vals}")
            metric_res_dict[key] = vals
    
    balanced_acc = balanced_accuracy_score(all_labels_used, np.array(all_preds)) # balanced accuracy is defined as the average of recall obtained on each class.
    metric_res_dict.update({"balanced acc": balanced_acc})
    
    allfold_summary.append(metric_res_dict)

    eval_res = pd.concat((pd.DataFrame(patient_results), pd.DataFrame(all_preds, columns=["pred"]), 
                            pd.DataFrame(np.array(all_cases_embedding), columns=[f"sim{i}" for i in  list(range(len(all_cases_embedding[0])))])),
                            axis=1)
    
    return allfold_summary, eval_res


def set_args():
    parser = argparse.ArgumentParser(description='independent evaluation on internal or external test.')
    parser.add_argument('-m', '--task_model_dir', type=str, 
                        default=None, help='Spec exp task path for loading model.')
    parser.add_argument('-n', '--patch_num', type=int, 
                        default=None, help='patch num in one slide for evaluation.')     
    
    parser.add_argument('-fp', '--indepedent_feats_path', type=str, default=None, 
                        help='feats path for evaluating on indepedent dataset, if Set.')
    parser.add_argument('-cp', '--indepedent_csv_path', type=str, default=None,
                        help='csv path for indepedent dataset, if Set.')
    parser.add_argument('-sp', '--indepedent_save_path', type=str, default=None,
                        help='saving path for indepedent dataset evaluation results, if Set.')
        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    task_model_dir = args.task_model_dir
    print(">>>>>>>>>>>>>>>>>>>{}".format(task_model_dir))
    
    ## read saved params
    params_dict = read_params_file(os.path.join(task_model_dir, "params_setting.txt"))

    if args.indepedent_feats_path is None: # means we evaluate on test set of the same dataset corresponding to task_name
        feats_path = params_dict['data_root_dir']
        save_to_path = os.path.join(task_model_dir, "independent_eval_results")

        slide_id_label_dict = load_data(params_dict['csv_info_path'], label_mapping={"Low": 1, "High": 0}, 
                                        filter_dict={"MMPrisk": ["Low", "High"]})
        
        csv_path = params_dict['split_dir']

    else:
        assert os.path.exists(args.indepedent_feats_path) and os.path.exists(args.indepedent_csv_path), \
            "Please provide correct path for indepedent_set_feats_path and indepedent_set_csv_path"

        feats_path = args.indepedent_feats_path
        csv_path = args.indepedent_csv_path
        save_to_path = args.indepedent_save_path
        slide_id_label_dict = load_data(csv_path, label_mapping={"Low": 1, "High": 0}, 
                                                    filter_dict={"MMPrisk": ["Low", "High"]})
        

    alltimes_summary = []
    for tidx in range(params_dict["times"]):
        device = set_seed_torch(params_dict["gpu"], params_dict["seed"])

        allfold_summary = []
        for kidx in range(params_dict["k"]): # each fold in each time
            
            if 'agent_token_num' not in params_dict:
                params_dict['agent_token_num'] = None
            model = load_model(params_dict, trained_pt_loc= os.path.join(task_model_dir, "time"+str(tidx)), fold=kidx, device=device)

            if csv_path.split('_')[-1] == "KFoldsCV":
                data = pd.read_csv(os.path.join(csv_path, f"splits_time{tidx}_fold{kidx}.csv"))

                slide_id_list = data['test'].dropna().reset_index(drop=True).astype(int)
                slide_label_list = [slide_id_label_dict[slide] for slide in slide_id_list]
            else:
                slide_id_list = list(slide_id_label_dict.keys())
                slide_label_list = list(slide_id_label_dict.values())

            allfold_summary, eval_res = load_inference(slide_id_list, slide_label_list, feats_path, tidx, kidx, 
                                                       model, patch_num=args.patch_num, **params_dict)
            os.makedirs(os.path.join(save_to_path, str(args.patch_num), "time"+str(tidx)), exist_ok=True)
            eval_res.to_csv(os.path.join(save_to_path, str(args.patch_num), "time"+str(tidx), f"indepent_eval_{kidx}_res.csv"))
        
        allfold_summary = pd.DataFrame(allfold_summary)
        alltimes_summary.append(allfold_summary)   

    alltimes_summary = pd.concat(alltimes_summary)
    alltimes_summary.loc['mean'] = alltimes_summary.apply(lambda x: x.mean())
    alltimes_summary.loc['std'] = alltimes_summary.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内
    
    alltimes_summary.to_csv(os.path.join(save_to_path, str(args.patch_num), f"summary_metrics_alltimes_kfolds.csv"))    
    print(f'>>>>>>>>>>>>>>>> {args.indepedent_feats_path} DATASET evaluated on TASK\n {task_model_dir}:\n {alltimes_summary}')