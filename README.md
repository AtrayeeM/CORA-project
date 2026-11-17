# CORA Citation Network: Node Classification using GNNs

An end-to-end machine learning pipeline that classifies academic papers into 7 distinct categories based on their citation links and text features. This project compares a baseline tabular model (optimized Random Forest) against a deep Graph Neural Network (GNN) built with PyTorch Geometric, served via a FastAPI backend and an interactive Streamlit frontend.

## 🚀 Features
* **Transductive Learning Pipeline:** Implements boolean masking (`train_mask`, `val_mask`, `test_mask`) for localized tensor representations of unstructured graph data.
* **Graph Neural Network (GNN):** Leverages PyTorch Geometric to capture relational citation context, outperforming traditional text-only classifiers.
* **Baseline Benchmarking:** Includes a Random Forest classifier tuned via GridSearchCV for robust baseline comparison.
* **Decoupled Architecture:** Asynchronous FastAPI backend handling model inference, communicating with a lightweight Streamlit UI.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Machine Learning:** Scikit-learn (Random Forest, GridSearchCV)
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit
* **Data Processing:** Pandas, NumPy, NetworkX

## 📊 Model Evaluation
| Model | Context Used | Architecture | Accuracy |
| :--- | :--- | :--- | :--- |
| **Random Forest** | Node Features Only | Ensemble Decision Trees (Tuned) | 58% |
| **Graph Neural Network** | Node Features + Graph Structure | Multi-layer GCN | 77% |

*Note: The GNN demonstrates superior performance by utilizing the homophily of the citation network (papers tend to cite other papers in the same category).*

## 💻 Local Setup & Installation

**1. Clone the repository**
///bash
git clone [https://github.com/AtrayeeM/CORA-project.git](https://github.com/AtrayeeM/CORA-project.git)
cd CORA-project
///
**2. Create a virtual environment and install dependencies**
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

**Running the Application**
This application requires two terminal windows to run the decoupled frontend and backend simultaneously.

**Terminal 1: Start the FastAPI Backend**
# Ensure your virtual environment is activated
uvicorn main:app --reload --port 8000

**Terminal 2: Start the Streamlit Frontend**
# Open a new terminal, navigate to the project folder, and activate the venv
streamlit run app.py
