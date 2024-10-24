task=MMPriskRegression
datasource=TJMUCH_MMP                                          # used to determine yaml files
results_dir=results/developed_models

exp_code=debug  
model_type=CPMP                                                # (default: CPMP), ['TransMIL', 'CLAM', 'ABMIL', 'Transformer']

data_root_dir=results/TJMUCH_MMP_2Feats_20x_uni                # data directory
encoding_dim=1024                                              # used for CPMP, Transformer, and TransMIL
model_size=uni1024                                             # only used for ABMIL and CLAM


## >>>>>>>>>>>>>>>> determine the logpath <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
logpath=${results_dir}/${datasource}/${task}/${exp_code}
if [ ! -d $logpath ]
        then mkdir -p $logpath
fi

## >>>>>>>>>>>>>>>> train the model and save logs <<<<<<<<<<<<<<<<<<<<<<<<<
CUDA_VISIBLE_DEVICES=1 python ModelTraining.py \
    --config cfgs/${datasource}.yaml \
    --opts \
    exp_code $exp_code model_type $model_type \
    data_root_dir $data_root_dir \
    encoding_dim $encoding_dim \
    model_size $model_size 2>&1 | tee $logpath/train_logs.log