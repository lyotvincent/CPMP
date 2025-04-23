### Demo data 

For inference and visualization, you can download this demo slide ([https://portal.gdc.cancer.gov/files/a866f667-ca52-46f6-9bfd-7f89515aad2e](https://portal.gdc.cancer.gov/files/a866f667-ca52-46f6-9bfd-7f89515aad2e), from TCGA-BRCA) into this folder.

Or, you can move other slide data into this directory.

```
CPMP/
├── cfgs/                                       # config. files dir
├── demo-data/
│   ├── README.md                               # readme doc
│   ├── TCGA-Z7-A8R6-01Z-00-DX1.xxxxxx.svs      # demo svs slide data
│   └── ...                                     # other svs slide data
└── xxx/                                        # other dirs

```

Then you can run `CUDA_VISIBLE_DEVICES=0 python SlideInference.py --configs cfgs/wsi_heatmap_params.yaml` for inference.    

The results can be found at `results/demo_TCGABRCA_4Heatmaps_20x_t0f0`


---------

**NOTE**: Pre-trained **UNI** foundation model can be found at [github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)    

You need to `mkdir pretrained_model_weights/` and put these pretrained models into `pretrained_model_weights/`.    

```
CPMP/
├── cfgs/           # config. files dir
├── demo-data/      # demo data dir
├── pretrained_model_weights/
│   ├── UNI/        # pretrained UNI foundation model folder
│   └── xxx/        # other model dirs
└── xxx/            # other dirs
```