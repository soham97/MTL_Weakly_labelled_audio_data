## Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection (PyTorch implementation)
Graduate project report (full-paper): https://arxiv.org/pdf/2008.07085.pdf

A follow up short paper (INTERSPEECH 21) of this work: <b>"Improving weakly supervised sound event detection with self-supervised auxiliary tasks"</b> here: https://arxiv.org/pdf/2106.06858.pdf

## Dataset
The dataset is made by mixing [DCASE 2019 Task 1](http://dcase.community/challenge2019/task-acoustic-scene-classification) of Acoustic scene classificaion and [DCASE 2018 Task 2](http://dcase.community/challenge2018/task-general-purpose-audio-tagging) of General purpose Audio tagging. The sound events are mixed with background sounds to form 10 seconds audio clips. Each audio clip consists of 3 sound events. Only the weakly labelled data is used for training.

## Run the code
**0. Prepare data**. Create a data folder with following structure (Only the logmel files are necessary):
<pre>
.
└── logmel (~75 GB)
     ├── logmel_snr_0.h5
     ├── logmel_snr_10.h5
     └── logmel_snr_20.h5   
└── mixed_audio (~25 GB)
     ├── snr_0 (8000 audios)
     │     └── ...
     ├── snr_10(8000 audios)
     │     └── ...
     └── snr_20 (8000 audios)
           └── ...
</pre>

**1. Install dependent packages**. The packages used are:
- Pytorch
- argparse
- logging
- h5py
- tqdm
- sklearn
- matplotlib

**2. Run**. <br>
- For training the network, execute main.py <br>
Example usage:
```bash
python3 main.py -data_dir data -exp_name test -batch_size 24 -epochs 80 -lr 1e-3 \
-num_workers 64 -data_parallel 1 -model_type MTL_SEDNetwork -val_fold 4 -snr 20 -alpha 0.001
```
- For visualising the results, execute visualise.py (support only for MTL_SEDNetwork) <br>
Example usage:
```bash
python3 main.py -data_dir data -exp_name dual_attn_vis -batch_size 24 \
        -num_workers 64 -data_parallel 1 -model_type MTL_SEDNetwork -snr 20 \
        -pretrained_model_path test/best.pth -val_fold 4
```
## Results
Check the paper out for full result comparison and ablation study
```
-------------------------------------------------------------------------------------------------------
|     Networks    |       SNR 20 dB           |       SNR 10 dB           |        SNR 0 dB            |  
|                 | micro-P   macro-P    AUC  | micro-P   macro-P    AUC  | micro-P   macro-P    AUC   |
-------------------------------------------------------------------------------------------------------
|      2APAE      | 0.7829    0.7645   0.9390 |  0.7603    0.7486  0.9343 | 0.6986    0.6892    0.9177 |
-------------------------------------------------------------------------------------------------------
```

## Citation
@misc{deshmukh2021improving,
      title={Improving weakly supervised sound event detection with self-supervised auxiliary tasks}, 
      author={Soham Deshmukh and Bhiksha Raj and Rita Singh},
      year={2021},
      eprint={2106.06858},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@misc{deshmukh2020multitask,
    title={Multi-Task Learning for Interpretable Weakly Labelled Sound Event Detection},
    author={Soham Deshmukh and Bhiksha Raj and Rita Singh},
    year={2020},
    eprint={2008.07085},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}

## External Links
The base code and dataset mixing is from sed_time_freq_segmentation by Qiuqiang Kong https://github.com/qiuqiangkong/sed_time_freq_segmentation
