# ME341_Project_DPC

# DPC Policy Training and Implementation Project

## 📌 Project Overview
This project is designed for training and implementing a **Deep Predictive Control (DPC) Policy model**. Specifically, the project provides code for:

- Defining and training the neural network model (**DPC Policy**).
- Generating and managing datasets for training.
- Applying the trained model to real predictive control tasks.

The project leverages a pre-trained prediction model (**TiDE**) provided directly from Reference [2].

---

## 📁 File Structure and Descriptions

### Main Files

| File Name | Description |
|-----------|-------------|
| `DPC_Policy.py` | Defines the neural network architecture for the **DPC Policy**. |
| `DPC_Policy_Train.ipynb` | Jupyter notebook that includes data preparation, training procedures for the DPC Policy model, and saves trained model parameters. |
| `DPC_Main.ipynb` | Jupyter notebook to load the trained DPC Policy model and apply it to practical predictive control tasks. |

### Reference and Additional Files

| File Name | Description |
|-----------|-------------|
| `TiDE.py` | Prediction model (**TiDE**) taken directly from Reference [2]. (Not authored in this project.) |
| `nominal_params_w10_mid_noise_stable_final.pkl/.pth` | Pre-trained parameters for the TiDE prediction model provided from Reference [2]. |
| `policy_parameters_basic_case3.pkl/.pth` | Trained DPC Policy model parameters saved after training. |
| `x_and_u_and_c_case3.pickle` | Dataset used for training and evaluation. |

---

## 🛠️ Installation and Execution

### Package Installation
The project requires the following Python packages. Install them before running the notebooks.

```bash
pip install numpy pandas matplotlib plotly tqdm torch scikit-learn
