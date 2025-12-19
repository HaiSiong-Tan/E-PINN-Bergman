## E-PINN-Bergman

An application of Evidential Physics-Informed Neural Networks (E-PINN) to Bergman model of glucose-insulin dynamics

#### 📘 Contents

<small>
This repository includes:

- **bergman_epinn.ipynb** - a notebook that guides the application of E-PINN to glucose-insulin dataset of
  [Kartono et al.](https://iopscience.iop.org/article/10.1088/1757-899X/532/1/012016/meta) based on Bergman model of glucose-insulin    dynamics. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HaiSiong-Tan/E-PINN-Bergman/blob/main/EPINN_bergman/bergman_epinn.ipynb)


- **models** - some pretrained models and related objects for reference.
- **src** - Python source code containing utility functions.
- **data** - dataset in the form of two csv files.
- **doc_epinn_bergman.pdf** - a short write-up providing some explanatory details for various equations and algorithms, to be read alongside our paper https://arxiv.org/abs/2509.14568


#### 🚀 Getting Started

 - **Opening in Google Colab** -
simply upload the notebook and the folders to Colab and run them as-is.
 - **Running Locally** -
if you prefer running locally, install the required packages:

```bash
pip install pandas numpy scipy matplotlib numba torch corner
```
#### 💡 What is E-PINN?

<small>
E-PINN is a novel class of uncertainty-aware physics-informed neural networks. It leverages the marginal distribution loss function of evidential deep learning for estimating uncertainty of outputs, and infers unknown parameters of the PDE via a learned posterior distribution.

For full details, see the accompanying paper:
https://arxiv.org/abs/2509.14568
</small>


#### 👥 Authors

Hai Siong Tan, Kuancheng Wang and Rafe McBeth

#### 📝 License

This project is released under the Apache License 2.0.



#### Dataset Citation

```bibtex

@article{Kartono_2019,
	author = {Kartono, A and Nurullaeli and Syafutra, H and Wahyudi, S T and Sumaryada, T},
	journal = {IOP Conference Series: Materials Science and Engineering},
	month = {may},
	number = {1},
	pages = {012016},
	title = {A mathematical model of the intravenous glucose tolerance test illustrating an n-order decay rate of plasma insulin in healthy and type 2 diabetic subjects},
	volume = {532},
	year = {2019}}

```
