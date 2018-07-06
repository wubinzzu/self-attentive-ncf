# Self-Attentive Neural Collaborative Filtering

This is the official implementation of our paper "Self-Attentive Neural Collaborative Filtering" (https://arxiv.org/abs/1806.06446). This repository contains the scripts, models and datasets to reproduce the results in our paper.

# Dependencies

1. Tensorflow 1.7
2. Python 2.7
3. TQDM
4. Numpy, Scikit-Learn etc.

# Usage

In `./prep/` we provide the script to generate the datasets preprocessing yourself. We also loaded the env file onto dropbox for your convenience. You can download it via the following command, which will place the required environment in the `./datasets` folder.

```
bash ./setup/setup_yelp18.sh
```

After which, you may train SA-NCF model with 20 layers via the following command.

```
python train_cr.py --dataset yelp18 --rnn_type RANK_SANCF --opt Adam --lr 1e-3 --l2_reg 1e-8 --batch-size 512 --emb_size 64 --num_neg 2 --num_dense 20
```

To compare with an ablative MLP model with 20 layers, you can run:

```
python train_cr.py --dataset yelp18 --rnn_type RANK_MLP --opt Adam --lr 1e-3 --l2_reg 1e-8 --batch-size 512 --emb_size 64 --num_neg 2 --num_dense 20
```

You should be able to reproduce **similar** results to the paper (though not exactly due to some extent of non-deterministicism).


### Some notes on model calling

This repository works on a language based control of the model architecture. So by appending or pre-pending things to `--rnn_type` we can toggle different models. In general, the arg `rnn_type` controls the model. For example, to test a tower MLP you can use the command `RANK_MLP_TOWER`. By default, all models optimize the BPR loss, you can switch to the hinge loss by adding a `HINGE` at rnn_type.

# Reference and Citation

If you find our codes useful and/or use it in your research, please consider citing our paper:

```
@article{DBLP:journals/corr/abs-1806-06446,
  author    = {Yi Tay and
               Shuai Zhang and
               Luu Anh Tuan and
               Siu Cheung Hui},
  title     = {Self-Attentive Neural Collaborative Filtering},
  journal   = {CoRR},
  volume    = {abs/1806.06446},
  year      = {2018},
  url       = {http://arxiv.org/abs/1806.06446},
  archivePrefix = {arXiv},
  eprint    = {1806.06446},
  timestamp = {Tue, 03 Jul 2018 18:25:22 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1806-06446},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Sample Output

Sample output of running RANK_SANCF. (This was an independent run from the experiments conducted in the paper, you should get similar, though not identical result).

Note: We observe that sometimes, when training SA-NCF, the prediction goes haywire/overfits after x epochs. However, best epoch reports the exact epoch when the best dev score is reported and so it is the one that is reported in the paper.

```
===============================
[yelp18] [Epoch 1] [RANK_SANCF] loss=0.201309546828
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 2] [RANK_SANCF] loss=0.0773584544659
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 3] [RANK_SANCF] loss=0.0563074275851
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 4] [RANK_SANCF] loss=0.0401255264878
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 5] [RANK_SANCF] loss=0.027956046164
GPU=1 | d=64 | num_layers=20
[Dev] nDCG=0.599250506618
[Dev] HR10=0.828712736513
[Dev] ACC=0.238758564363
[Test] nDCG=0.594043919378
[Test] HR10=0.823948455012
[Test] ACC=0.231362584509
====================================
Best epoch=5
[best] nDCG=0.594043919378
[best] HR10=0.823948455012
[best] ACC=0.231362584509
Maxed epoch=5
[max] nDCG=0.594043919378
[max] HR10=0.823948455012
[max] ACC=0.231362584509
===============================
[yelp18] [Epoch 6] [RANK_SANCF] loss=0.018275661394
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 7] [RANK_SANCF] loss=0.0138740120456
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 8] [RANK_SANCF] loss=0.0116575174034
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 9] [RANK_SANCF] loss=0.0101664736867
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 10] [RANK_SANCF] loss=0.00873020105064
GPU=1 | d=64 | num_layers=20
[Dev] nDCG=0.47857012475
[Dev] HR10=0.730704659921
[Dev] ACC=0.167929579382
[Test] nDCG=0.474648417237
[Test] HR10=0.730387041154
[Test] ACC=0.160533599528
====================================
Best epoch=5
[best] nDCG=0.594043919378
[best] HR10=0.823948455012
[best] ACC=0.231362584509
Maxed epoch=5
[max] nDCG=0.594043919378
[max] HR10=0.823948455012
[max] ACC=0.231362584509
===============================
[yelp18] [Epoch 11] [RANK_SANCF] loss=0.00755625218153
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 12] [RANK_SANCF] loss=0.00536831235513
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 13] [RANK_SANCF] loss=0.00523949740455
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 14] [RANK_SANCF] loss=0.00465025380254
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 15] [RANK_SANCF] loss=0.00428779236972
GPU=1 | d=64 | num_layers=20
[Dev] nDCG=0.601561306421
[Dev] HR10=0.831208312537
[Dev] ACC=0.22736966287
[Test] nDCG=0.597203603098
[Test] HR10=0.830164708018
[Test] ACC=0.224692590408
====================================
Best epoch=15
[best] nDCG=0.597203603098
[best] HR10=0.830164708018
[best] ACC=0.224692590408
Maxed epoch=15
[max] nDCG=0.597203603098
[max] HR10=0.830164708018
[max] ACC=0.224692590408
===============================
[yelp18] [Epoch 16] [RANK_SANCF] loss=0.0037534229923
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 17] [RANK_SANCF] loss=0.00312445801683
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 18] [RANK_SANCF] loss=0.00290405750275
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 19] [RANK_SANCF] loss=0.00232721492648
GPU=1 | d=64 | num_layers=20
===============================
[yelp18] [Epoch 20] [RANK_SANCF] loss=0.00247630174272
GPU=1 | d=64 | num_layers=20
[Dev] nDCG=0.790238058772
[Dev] HR10=0.966287036617
[Dev] ACC=0.35273832751
[Test] nDCG=0.786269133615
[Test] HR10=0.965288806207
[Test] ACC=0.3469758156
====================================
Best epoch=20
[best] nDCG=0.786269133615
[best] HR10=0.965288806207
[best] ACC=0.3469758156
Maxed epoch=20
[max] nDCG=0.786269133615
[max] HR10=0.965288806207
[max] ACC=0.3469758156
```
