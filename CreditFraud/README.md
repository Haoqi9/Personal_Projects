<h2 align="center">DESCRIPTION</h2>

- A model based on LightGBM Classifier to **detect credit fraud** has been developed.

  - `credit_default_notebook.html`, covers from the target definition to data preprocessing.
  - `2_Modeling.ipynb`, contains the modeling and evaluation phase.
  - Finally, the **trained model has been deployed in a web application** where real-time predictions can be made.

<h2 align="center">ACCESS TO WORK</h2>

> [!NOTE]
> If for any reason GitHub cannot display the Jupyter notebook, try the alternative link: **[Alternative Link]**.
- Access the first notebook (**1_Preprocessing.ipynb**):
  - `English version`: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing_en.ipynb) | [[Alternative Link]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing_en.ipynb)
  - `Spanish version`: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing_es.ipynb) | [[Alternative Link]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing_es.ipynb)
    
- Access the second notebook (**2_Modeling.ipynb**):
  - `English version`: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling_en.ipynb) | [[Alternative Link]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling_en.ipynb)
  - `Spanish version`: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling_es.ipynb) | [[Alternative Link]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling_es.ipynb)
     
- Access the streamlit web application (**Credit Risk Analyzer**):
  - `English version`: [[Link]](https://creditriskwebappst-lmmesu5xdk4m45fu9icbmf.streamlit.app/)
    
<h2 align="center">SUMMARY</h2>

## Business Problem
- A financial institution wants to improve its risk assessment process and reduce losses associated with payment default. To address this challenge, it decides to develop **a credit default detection algorithm that can accurately identify customers who are most likely to default on their financial obligations**.

## Initial Features (24)
- id_cliente, empleo, antigüedad_empleo, ingresos, ingresos_verificados,
- rating, dti, vivienda, num_hipotecas, num_lineas_credito,
- porc_tarjetas_75p, porc_uso_revolving, num_cancelaciones_12meses, num_derogatorios, num_meses_desde_ult_retraso,
- id_prestamo, descripcion, finalidad, principal, tipo_interes,
- num_cuotas, imp_cuota, imp_amortizado, imp_recuperado.

## Final Features (14)
- antigüedad_empleo, dti, finalidad, ingresos, ingresos_verificados,
- num_derogatorios, num_hipotecas, num_lineas_credito, porc_tarjetas_75p, porc_uso_revolving,
-  principal, rating, tipo_interes, vivienda.

## Model Selection 
|Model|fit time (s)|accuracy|roc_auc|f1|precision|recall|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GaussianNB|1.00|0.867|0.661|0.039|0.243|0.021|
|LogisticRegression|8.62|0.872|0.521|0.000|0.025|0.000|
|**LGBMClassifier**|**6.05**|**0.872**|**0.723**|**0.007**|**0.518**|**0.004**|
|RandomForestClassifier|86.61|0.872|0.684|0.021|0.505|0.011|

## Distribution of probability estimates
### With Class Imbalance
![pd1](./Images/pd1.png)
### Balanced Class Weights
![pd2](./Images/pd2.png)
### Observations
- Regarding the distribution of probability estimates of the model with default parameters and the other with balanced class weights, it is observed that **the distributions of the two classes are highly overlapping in both models, although in the model with balanced weights these distributions are more separated from each other**. We already knew that the model would have great difficulty in recognizing the 0s due to the small number of samples for that minority class and the available variables. Hence, the distribution for the 0s is more spread out.

- In this particular case, **we are interested in prioritizing the identification of customers who are likely to default** rather than the confidence, which is also important, that they have already defaulted. However, it is not advisable to decrease the probability threshold to increase sensitivity (recall for the positive class) since the increase in false positives would drastically decrease precision. **We will leave the threshold at default value, 0.5**.

## Results
### With Class Imbalance
![resultado1](./Images/result1.png)
### Balanced Class Weights
![resultado2](./Images/result2.png)
### Observations
- As we had seen in the previous results, **the resulting model leaves much to be desired in terms of its ability to correctly detect defaults (positive class) with a recall close to 0%**, although **this metric can be raised to 70% in the model by balancing the classes by changing the class weights during training**. This improvement has been achieved at the expense of greatly sacrificing precision from 56% to 22% and the ability to detect customers who have not defaulted, from a recall for the negative class close to 100% to 64%. All this indicates that it is not possible to discriminate well between the two classes with the features present; there is a significant overlap in the probability distributions of the two classes.

