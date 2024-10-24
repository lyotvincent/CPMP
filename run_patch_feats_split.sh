# TJMUCH wsi dataset
labelname="MMPrisk"
csvinfopath="results/MPcohort_info.csv"

datapath="/home/cyyan/Projects/TJMUCHMMP-cohort/"
patchpath="results/TJMUCH_MMP_1WsiPatching_20x"
featspath="results/TJMUCH_MMP_2Feats_20x"
splitspath="results/TJMUCH_MMP_3CaseSplits"

tocsvpath=$patchpath"/process_list_autogen.csv"

tile_size=512
overlap_size=512
downsample_factor=2 # tile_size: ds_factor should be {256: 1, 512: 2, 1024: 4, 2048: 8}

batchsize=512


if false; then
echo "WsiPatching..."
python WsiTiling.py \
	-s  $datapath \
	-d  $patchpath \
    -ps $tile_size \
    -ss $overlap_size \
	--patch \
	--bgtissue \
	--stitch
fi

############## feature extraction by RetCCL model
if false; then
pretrain_model="retccl"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --custom_downsample $downsample_factor \
	--float16
fi
#--gaussian_blur \


############## feature extraction by ctranspath model
if false; then
pretrain_model="ctranspath"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi

############## feature extraction by PhiKon model
if false; then
pretrain_model="phikon"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
    --auto_skip \
	--float16
fi


############## feature extraction by PhiKonV2 model
if false; then
pretrain_model="phikonV2"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
    --auto_skip \
	--float16
fi


############## feature extraction by uni model
if false; then
pretrain_model="uni"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi


############## feature extraction by plip model
if false; then
pretrain_model="plip"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16 \
    --zeroshot
fi

############## feature extraction by gigapath model
if false; then
pretrain_model="gigapath"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi


############## feature extraction by virchow model
if false; then
pretrain_model="virchow"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi


############## feature extraction by virchow2 model
if false; then
pretrain_model="virchow2"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi


############## feature extraction by H-optimus-0 model
if false; then
pretrain_model="H-optimus-0"
CUDA_VISIBLE_DEVICES=0 python FeatsExtracting.py \
	--feat_to_dir $featspath"_"$pretrain_model \
	--csv_path $tocsvpath \
	--h5_dir $patchpath \
	--pretrain_model $pretrain_model \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size $batchsize \
    --resize_size 224 \
    --custom_downsample $downsample_factor \
	--float16
fi


if false; then
echo "Cross Validation splitting ..."
echo "N times K folds cross validation mode split."
python CaseSplitting.py \
    --task_name "MMPrisk" \
	--csv_info_path $csvinfopath \
	--split_to_dir $splitspath \
    --times 5 \
	--kfold 5 \
	--val_frac 0 \
	--test_frac 0.2 \
	--label_column_name $labelname \
	--label_list "Low" "High"\
    --slide_featspath $featspath\
	--seed 2020
fi

if false; then
echo "N times train-val-test mode split."
python CaseSplitting.py \
    --task_name "MMPrisk" \
	--csv_info_path $csvinfopath \
	--split_to_dir $splitspath \
    --times 5 \
	--kfold 0 \
	--val_frac 0.15 \
	--test_frac 0.15 \
	--label_column_name $labelname \
	--label_list "Low" "High"\
    --slide_featspath $featspath\
	--seed 2020
fi