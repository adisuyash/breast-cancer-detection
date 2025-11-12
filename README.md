[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adisuyash/breast-cancer-detection/blob/main/Breast_Cancer_Detection.ipynb)

# 🩺 Breast Cancer Detection in Healthcare
Our project aims to predict breast cancer using machine learning techniques. It classifies tumours as `Benign` or `Malignant`, supporting early diagnosis and data-driven healthcare decisions.

The model is developed using the **Breast Cancer Wisconsin (Diagnostic) Dataset** from `sklearn.datasets`, which includes medical attributes describing **cell nuclei characteristics**.  

## Tech Stack
### Machine Learning & AI  
- TensorFlow / Keras: Neural network model for tumor classification  
- Scikit-learn: Dataset loading, preprocessing, and evaluation  

### Data Analysis & Visualization  
- NumPy / Pandas: Data handling and processing  
- Matplotlib: Model performance and data visualization  

### Development  
- Python 3.8+  
- Jupyter Notebook / Google Colab  
- GitHub: Version control

## Dataset
The **Breast Cancer Wisconsin (Diagnostic)** dataset can be imported directly from the `sklearn.datasets`.  

```python
import sklearn.datasets
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```

## Related Snapshots
### Model Training
<img width="1209" height="458" alt="Model Training" src="https://github.com/user-attachments/assets/42b439c5-a1f5-4335-a73f-ee971332cd37" />

### Model Accuracy and Loss
<table>
  <tr>
    <td>
      <img width="719" height="574" alt="Model Accuracy" 
           src="https://github.com/user-attachments/assets/3d0b9254-3831-4baf-846f-fd882a86ac90" />
    </td>
    <td>
      <img width="723" height="576" alt="Model Loss" 
           src="https://github.com/user-attachments/assets/06fd85bd-aef4-40ed-96d1-7257f606f2b2" />
    </td>
  </tr>
  <tr>
    <th>Model Accuracy</th>
    <th>Model Loss</th>
  </tr>
</table>

### Tumour Prediction
<img width="404" height="95" alt="Tumour Prediction" src="https://github.com/user-attachments/assets/5801ebdb-465b-42a1-b868-8edb93a10634" />

## How to Run
1. Clone this repository:
```bash
git clone https://github.com/adisuyash/breast-cancer-detection.git
```

2. Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook Breast_Cancer_Detection.ipynb
```

3. Run all cells to train and test the model.

## Contributors
- Aditya Gupta (202210101150125)
- Akshat Yadav (202210101150121)

> Developed under the IBM Project Initiative for Internal Assessment at SRMU.
