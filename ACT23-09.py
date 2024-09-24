#!/usr/bin/env python
# coding: utf-8

# # Regresión Lineal
# 
# En esta actividad realizaras lo siguiente:
# 
# **Creación de variables**
# 
# * importar datos (variable exógena, X a la respuesta Y)
# * variables de estacionales
# * tendencia
# * variables de atípicos
# * variables rezagos
# 
# **Aplicación de regresión lineal múltiple**
# 
# **Evaluación de resultados y errores**
# 
# **Selección de variables**
# 
# 
# > The forecast variable y is sometimes also called the regressand, dependent or explained variable. The predictor variables x are sometimes also called the regressors, independent or explanatory variables.
# 
# 
# ## Variables X
# 
# ### Importar datos
# 
# Importa la serie de tiempo que escogiste anteriormente para usar como variable que explica a tu variable objetivo.
# 
# En el ejemplo de las remeses
# - mi variable objetivo son las remeses recibidas cada mes
# - mi variable exógena es el tipo de cambio

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy as sp
import yfinance as yf


# In[4]:


data = pd.read_excel("inflacion.xlsx")
data.index = pd.to_datetime(data['Fecha'])
del data['Fecha']


# In[5]:


stock = 'MXN=X'
ticker = yf.Ticker(stock)
usdmxn = ticker.history(start= '1993-01-01', end= '2023-12-31', interval='1mo')['Close']

usdmxn.head()


# - Ajusta la serie de tiempo para que las fechas sean las mismas.
# - Ajusta el formato de la fecha.

# In[6]:


data = data["2003":]


# In[7]:


usdmxn = usdmxn["2004":]


# In[8]:


usdmxn.index = usdmxn.index.strftime('%Y-%m')
usdmxn.index = pd.to_datetime(usdmxn.index)
usdmxn.head()


# ### Variables estacionales
# 
# En una regresión lineal cómun $$ y =  \beta_1x +  \beta_0 $$
# 
# perdemos información relacionada a la estacionalidad, en este caso los meses. Para incluir los meses Enero, Febrero, Marzo, etc. agregamos variable binarias, de tal forma que la ecuación sea $$ y =  \beta_0 +  \beta_{1}x_{enero},  \beta_{2}x_{febrero} +  \beta_{3}x_{marzo}  \dots $$
# 
# 

# In[9]:


data = pd.DataFrame(data.values, columns=['Y'], index=data.index)
data.head()


# In[10]:


data = data.join(usdmxn)


# In[11]:


getattr(data.index, 'month_name')


# In[12]:


data['mes'] = data.index.month_name()


# In[13]:


data


# In[14]:


data = pd.get_dummies(data, columns=['mes'], prefix="", prefix_sep="", drop_first=True, dtype=float)


# In[15]:


data


# ### Atípicos
# 
# De la sesión anterior, utiliza los valores atípicos que encontraste en tu varible. Deben ser una tupla fecha - 1

# In[16]:


fecha_2008 = pd.Series(data = [1], index=pd.to_datetime(["2008-11-01"]), name='2008_outlier')
fecha_2020 = pd.Series(data = [1], index=pd.to_datetime(["2020-03-01"]), name='2020_outlier')


# In[17]:


fecha_2008


# In[18]:


data = data.join(fecha_2008).fillna(0)


# In[19]:


data["2008":"2008"]


# In[20]:


data = data.join(fecha_2020).fillna(0)


# In[21]:


data["2020":"2020"]


# ### Lags
# 
# En clases pasadas analizamos la autocorrelación, la cuál nos dice que valores pasados son representativos para el valor actual. Considerando eso y la estacionalidad mensual, puedes crear una 12 variable de rezagos.
# 
# $$ y =  \beta_0 +  \beta_{1}x_{mespasado},  \beta_{2}x_{2mesesatras} +  \beta_{3}x_{3mesesatras}  \dots $$

# In[22]:


data['lag1'] = data['Y'].shift(1)
data['lag2'] = data['Y'].shift(2)
data['lag3'] = data['Y'].shift(3)
data['lag4'] = data['Y'].shift(4)
data['lag5'] = data['Y'].shift(5)
data['lag6'] = data['Y'].shift(6)
data['lag7'] = data['Y'].shift(7)
data['lag8'] = data['Y'].shift(8)
data['lag9'] = data['Y'].shift(9)
data['lag10'] = data['Y'].shift(10)
data['lag11'] = data['Y'].shift(11)
data['lag12'] = data['Y'].shift(12)


# In[23]:


data.head(13)


# In[24]:


data = data["2004":]


# ## Aplica el modelo de regresión

# In[25]:


import statsmodels.api as sm


# In[26]:


print(sm.OLS(data['Y'], data.drop(columns=['Y'])).fit().summary())


# > he Durbin-Watson statistic will always have a value ranging between 0 and 4. A value of 2.0 indicates there is no autocorrelation detected in the sample. Values from 0 to less than 2 point to positive autocorrelation, and values from 2 to 4 mean negative autocorrelation.
# 
# > In statistics, the Jarque–Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).score(data.drop(columns=['Y']), data['Y'])


# In[29]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).coef_


# In[30]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).intercept_


# In[31]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).feature_names_in_


# In[32]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).n_features_in_


# In[33]:


errors = LinearRegression().fit(data.drop(columns=['Y']), data['Y']).predict(data.drop(columns=['Y'])) - data['Y']


# In[34]:


errors.head()


# In[36]:


fig, ax = plt.subplots(1, 1, figsize = (20, 8))
errors.plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('Residuales del modelo ajustado Inflacion 2004 - 2023', fontsize=22)
ax.set_ylabel('Miles de millones de dólares', fontsize=20)
ax.set_xlabel('Mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# In[37]:


errors.mean()


# In[38]:


# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(20, 8),
                        tight_layout = True)

axs.hist(errors, bins = 20)

# Specify graph features:
axs.set_title('histograma residuales', fontsize=22)
axs.set_ylabel('conteo', fontsize=20)
axs.set_xlabel('residuales', fontsize=20)

# Show plot
plt.show()


# In[39]:


import math

ticker_data = errors
ticker_data_acf = [ticker_data.autocorr(i) for i in range(1,25)]

test_df = pd.DataFrame([ticker_data_acf]).T
test_df.columns = ['Autocorr']
test_df.index += 1
test_df.plot(kind='bar', width = 0.05, figsize = (20, 4))

# Statisfical significance.
n = len(errors)
plt.axhline(y = 2/math.sqrt(n), color = 'r', linestyle = 'dashed')
plt.axhline(y = -2/math.sqrt(n), color = 'r', linestyle = 'dashed')

# Adding plot title.
plt.title("Residuals from the Naive method")

# Providing x-axis name.
plt.xlabel("lag[1]")

# Providing y-axis name.
plt.ylabel("ACF")


# In[40]:


import scipy as sp


# In[46]:


sp.stats.boxcox(data['Y'])[0].shape


# In[47]:


data_copy = data.copy()
data_copy['Y']= sp.stats.boxcox(data_copy['Y'])[0]


# In[48]:


data_copy.head()


# In[49]:


LinearRegression().fit(data_copy.drop(columns=['Y']), data_copy['Y']).score(data_copy.drop(columns=['Y']), data_copy['Y'])


# In[52]:


predict = LinearRegression ().fit (data_copy.drop (columns=['Y']),data_copy ['Y']).predict (data_copy.drop(columns=['Y']))


# In[53]:


errors = predict - data_copy['Y']


# In[54]:


fig, ax = plt.subplots(1,1,figsize = (20,8))
errors.plot(ax=ax, linewidth =2)


# ## Conclusión
# #### Tomando en cuenta las diferentes herramientas utilizadas podemos apreciar que el primer modelo creado es más efectivo para nuestra variable exogena. Los factores incluyen la cola hacia la derecha en la tabla de distribución, normalidad y el cálculo del error.

# In[ ]:




