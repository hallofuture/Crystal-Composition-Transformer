# Crystal-Composition-Transformer
This repository contains the code and datasets for the paper:

[**Crystal Transformer: Self-learning neural language model for Generative and Tinkering Design of Materials**](https://arxiv.org/pdf/2204.11953.pdf)  
*Lai Wei, Qinyang Li, Yuqi Song, Edirisuriya M. D. Siriwardane, Stanislav Stefanov, and Jianjun Hu*

by <a href="http://mleg.cse.sc.edu" target="_blank">Machine Learning and Evolution Laboratory</a>, University of South Carolina.

### Python Dependencies
Install `Pytorch` from [Pytorch web](https://pytorch.org/get-started/previous-versions/) given your python & cuda version

The code is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. It has been tested in `PyTorch 1.6.0`, `PyTorch Lightning 1.0.7`

The version of Pymagen package: `pip install pymatgen==2021.2.16`

### Datasets for training Crystal Composition Transformer

|       | ICSD-mix | OQMD-mix | MP-mix | ICSD-pure | OQMD-pure | MP-pure |
|-------|----------|----------|--------|-----------|-----------|---------|
| Total | 52317    | 363182   | 89121  | 39431     | 216540    | 63703   |
| Train | 50755    | 345022   | 84664  | 37459     | 205713    | 60517   |
| Valid | 1336     | 9080     | 9080   | 986       | 5413      | 1593    |
| Test  | 1336     | 9080     | 9080   | 986       | 5413      | 1593    |


### Acknowledgements

We use the blank language model from [https://github.com/Varal7/blank_language_model/edit/release/README.md](https://github.com/Varal7/blank_language_model)

### How to train the model with Crystal Composition Transformer dataset

#### Download Data
Download datasets from the above link, then unzip it under `BLMM_dataset` folder.
After the above, the directory should be:
```
GMTransformer
   ├── GMTransformer_dataset
       ├── SMILE_data
           ├── SMILES_atom_train.txt
           ├── SMILES_atom_valid.txt
       ├── SELFIES_data
           ├── SELFIES_atom_train.txt
           ├── SELFIES_atom_valid.txt
   └── README.md
```

#### Training
An example is to train a GMTransformer model on the SMILES_atom dataset. 
```
python train.py --train GMTransformer_dataset/SMILE_data/SMILES_atom_train.txt --valid GMTransformer_dataset/SMILE_data/SMILES_atom_valid.txt --root_dir checkpoints/SMILES/atom/ \
--vocab_size 100 --max_len 200 --model_type blm --share_emb_prj_weight
```
The training for other models is similar to SMILES_atom dataset.

#### How to generate new molecules using the trained models
For all of the following, replace `epoch\=???.ckpt` with the checkpoint saved in training.

Generate molecules using the trained SMILES_atom model.
```
python test.py --checkpoint checkpoints/SMILES/atom/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--sample 1000 --decode sample --output sample.txt
```

### Citation

If you use our work, please cite:

```bibtex
@article{wei2022crystal,
  title={Crystal Transformer: Self-learning neural language model for Generative and Tinkering Design of Materials},
  author={Wei, Lai and Li, Qinyang and Song, Yuqi and Stefanov, Stanislav and Siriwardane, Edirisuriya and Chen, Fanglin and Hu, Jianjun},
  journal={arXiv preprint arXiv:2204.11953},
  year={2022}
}
```
