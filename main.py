import streamlit as st
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import numpy as np

# START = "2022-07-16"
# today = date.today()
#emojis = https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title = "Aplikasi Prediksi Harga Beras",
                   page_icon = "📈")

st.title("Aplikasi Prediksi Harga Beras")
def add_space(n_space):
    for i in range(n_space):
        st.text(" ")

add_space(3)
rice_type = ("Beras Premium", "Beras Medium")
col1, col2, col3 = st.columns(3)
default_date = date(2024, 5, 31)

with col1 : 
    selected_rice = st.selectbox("Jenis Beras", rice_type)
with col2 :
    start_date = st.date_input("Tanggal dari :", value = default_date.replace(day = 1))
with col3:
    end_date = st.date_input("Tanggal ke :", value = default_date)

beras_premium = pd.read_excel('datasets/beras_premium.xlsx')
beras_medium = pd.read_excel('datasets/beras_medium.xlsx')
beras_premium['tanggal'] = pd.to_datetime(beras_premium['tanggal'], dayfirst=True)
beras_medium['tanggal'] = pd.to_datetime(beras_medium['tanggal'], dayfirst=True)
beras_premium.set_index('tanggal', inplace=True)
beras_medium.set_index('tanggal', inplace=True)
beras_premium = beras_premium.asfreq('D')
beras_medium = beras_medium.asfreq('D')

def load_datesets(ticker, date1, date2):
    # data = None
    # rice_type = None
    if ticker == 'Beras Premium':
        data = beras_premium.loc[date1 : date2].copy() 
        final_data = beras_premium
        rice_type = 'premium'
    elif ticker == 'Beras Medium':
        data = beras_medium.loc[date1 : date2].copy()
        final_data = beras_medium
        rice_type = 'medium'
    return data, final_data, ticker, rice_type


data, final_data, name, rice_type = load_datesets(selected_rice, start_date, end_date)
data.index = data.index.strftime('%Y-%m-%d')
data = data.reset_index()
data = data.rename(columns={'tanggal':'Tanggal', data.columns[1] : name})

load_data_state = st.text("")
if(data is not None):
    load_data_state.text("Data berhasil dimuat!")
else:
    load_data_state.text("Data gagal dimuat")
add_space(1)

st.subheader('Harga Beras Harian')
add_space(2)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    rows = 8
    st.dataframe(data, height = rows * 35 + 3, width = 400)


fig = go.Figure()
fig.add_trace(go.Scatter(x = data['Tanggal'], 
                         y = data.iloc[:,1], 
                         name = data.columns[1],
                         line=dict(color='#2B60DE')))
fig.layout.update(title_text = f"Grafik Harga {name}", showlegend = True,
                  legend=dict(orientation="h",
                              yanchor="bottom",  
                              y = 1,
                              xanchor="center",
                              x = 0.1,
                              font = dict(size = 14)))
st.plotly_chart(fig)


##Forecasting
add_space(1)
st.subheader('Prediksi Harga 14 Hari Kedepan')
add_space(2)

from tensorflow.keras.models import load_model
import joblib

window_size = 5
steps = 14

def load_model_final():
    lstm_model = []
    sc_path = f"models/{rice_type}/model_final_forecasting/scaller_final_hybrid.pkl"
    sarima_path = f"models/{rice_type}/model_final_forecasting/sarima_model_final_hybrid.pkl"
    sc = joblib.load(sc_path)
    sarima_model = joblib.load(sarima_path)
    for i in range(steps):
        model_path = f"models/{rice_type}/model_final_forecasting/lstm_model_final_{i}_hybrid.keras"
        model = load_model(model_path)
        lstm_model.append(model)
    
    return sc, sarima_model, lstm_model

sc, sarima_model, lstm_model = load_model_final()

def make_data_direct(data, window_size, n_steps):
  x1, y1 = [], []
  for i in range(len(data) - window_size - n_steps + 1):
    x1.append(data[i:(i + window_size)])
    y1.append(data[(i + window_size):(i + window_size + n_steps)])

  return np.array(x1), np.array(y1)

import time

def direct_lstm_pred(models, x, scaler, n_steps):
  # predict
  direct_pred = np.zeros((x.shape[0], n_steps))
  pred_status = st.text("Memproses...")
  bar = st.progress(0)
  progress_status = st.empty()
  bar_step = math.floor(100/steps)
  j_temp = 1
  all_progress = 0
  for i, regs in enumerate(models):
    n_progress = bar_step
    if i == (steps-1):
       n_progress += 2
    direct_pred[:, i] = regs.predict(x).flatten()
    all_progress += n_progress
    for j in range(j_temp, all_progress + 1):
      time.sleep(0.01)
      bar.progress(j)
      progress_status.write(str(j) + " %")
    j_temp = all_progress
  pred_status.text("Selesai!")
  direct_pred = scaler.inverse_transform(direct_pred)

  return direct_pred

def residualForLstm(actual, pred, scaller):
  temp = actual.copy()
  temp['pred'] = pred
  temp['residual'] = temp.iloc[:, 0] - temp.iloc[:, 1]
  residual = pd.DataFrame(temp['residual'][1:], columns=['residual'])
  resid_scaled = scaller.fit_transform(residual)

  return resid_scaled

import math

def hybrid_model_predict(X, scaller):
  sarima_pred= sarima_model.predict(start = X.index[0], end = X.index[-1])
  resid_scaled= residualForLstm(X, sarima_pred, scaller)
  x_am, y_am = make_data_direct(sarima_pred[1:], window_size, steps)
  x, y = make_data_direct(resid_scaled, window_size, steps)
  direct_pred = direct_lstm_pred(lstm_model, x, scaller, steps)

  resid_pred = pd.DataFrame(direct_pred)
  sarima_pred2 = pd.DataFrame(y_am)
  final_pred = pd.DataFrame()
  
  for i in range(steps):
    final_pred[f'step {i+1}'] = sarima_pred2[i] + resid_pred[i]
  return final_pred

pred = hybrid_model_predict(final_data, sc)
final_pred = pred[pred.shape[0]-1:]
final_pred = final_pred.transpose() 

from datetime import timedelta
last_date = final_data.index.max()
new_date = pd.date_range(start = last_date + timedelta(days = 1), periods = steps)
df_pred = pd.DataFrame({f'{final_data.columns[0]}': final_pred.iloc[:,0].values}, index = new_date)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    rows = 15
    df_pred2 = df_pred.copy()
    df_pred2.index = df_pred2.index.strftime('%Y-%m-%d')
    df_pred2 = df_pred2.reset_index()
    df_pred2 = df_pred2.rename(columns={'index' : 'Tanggal', df_pred2.columns[1] : 'Prediction'})
    st.dataframe(df_pred2, height = rows * 35 + 3, width = 400)

add_space(2)
fig = go.Figure()
subset_data = final_data[-30:]
fig.add_trace(go.Scatter(x = subset_data.index, 
                         y = subset_data.iloc[:,0], 
                         name ='Actual',
                         line=dict(color='#2B60DE')))
fig.add_trace(go.Scatter(x = df_pred.index, 
                         y = df_pred.iloc[:,0], 
                         name ='Prediction',
                         line=dict(color='#50C878')))
fig.layout.update(title_text = f"Grafik Prediksi Harga {name}", showlegend = True,
                  legend=dict(orientation="h",
                              yanchor="bottom",  
                              y = 1,
                              xanchor="center",
                              x = 0.1,
                              font = dict(size = 14)))
st.plotly_chart(fig)



