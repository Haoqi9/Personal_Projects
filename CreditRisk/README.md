# DESCRIPTION
- Se ha desarrollado un modelo basado en **LightGBM Classifier para detectar impagos de créditos** utilizando datos de una institución financiera estadounidense. Se aplicaron pesos de clase balanceados para paliar el desbalanceo de clases.
  - El primer notebook, `1_Preprocessing.ipynb`, abarca desde la definición de la target hasta el preprocesado de los datos.
  - El segundo notebook, `2_Modeling.ipynb`, contiene la fase de modelización y evaluación.
  - Por último, **el modelo entrenado se ha puesto en producción en una aplicación web** donde se pueden realizar predicciones a tiempo real.
    
- Acceder al primer notebook: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing.ipynb)
  - Si por alguna razón GitHub no puede mostrar el cuaderno de Jupyter, intenta con este otro enlace: [[Link Alternativa]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/1_Preprocessing.ipynb)
- Acceder al segundo notebook: [[Link]](https://github.com/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling.ipynb)
  - Alternativa: [[Link Alternativa]](https://nbviewer.org/github/Haoqi9/Personal_Projects/blob/master/CreditRisk/03_Notebooks/2_Modeling.ipynb)    
- Acceder a la aplicación web: [[Link]](https://creditriskwebappst-lmmesu5xdk4m45fu9icbmf.streamlit.app/)

# RESULTS
## Sin balancear
![resultado1](./Images/result1.png)
## Balanced Class Weights
![resultado2](./Images/result2.png)
