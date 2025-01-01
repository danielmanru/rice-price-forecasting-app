import streamlit as st
from datetime import date, timedelta, datetime
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
import math
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

language = [
   {
      'nav': ['Versi Data', 'Bahasa'],
      'sdbar_title': 'Pengaturan',
      'list_data' : ['Data Penelitian', 'Data Terbaru'],
      'lang' : ['Indonesia','English'],
      'title' : 'Aplikasi Prediksi Harga Beras',
      'sl_box' : ['Jenis Beras', 'Tanggal dari:', 'Tanggal ke:'],
      'rice_type': ['Beras Premium', 'Beras Medium'],
      'load_state' : ['Data berhasil dimuat!','Data gagal dimuat!'],
      'header' : ['Harga Beras Harian','Prediksi Harga 14 Hari Kedepan', 'Error Prediksi'],
      'table': ['Tanggal','Prediksi','Aktual'],
      'actual_chart' : ['Grafik Harga Beras Premium','Grafik Harga Beras Medium'],
      'pred_status': ['Memproses...', 'Selesai!'],
      'pred_chart' : ['Grafik Prediksi Harga Beras Premium','Grafik Prediksi Harga Beras Medium', 'Aktual', 'Prediksi']
   },
   {
      'nav': ['Data Version', 'Language'],
      'sdbar_title': 'Settings',
      'list_data' : ['Research Data', 'Latest data'],
      'lang' : ['Indonesia', 'English'],
      'title' : 'Rice Price Prediction App',
      'sl_box' : ['Rice Type','Date from:', 'Date to:'],
      'rice_type': ['Premium Rice', 'Medium Rice'],
      'load_state' : ['Data loaded successfully!', 'Data failed to load!'],
      'header' : ['Daily Rice Price', 'Price Predictions for the Next 14 Days', 'Prediction Error'],
      'table': ['Date','Prediction','Actual'],
      'actual_chart' : ['Premium Rice Price Chart','Medium Rice Price Chart'],
      'pred_status': ['Processing...', 'Done!'],
      'pred_chart' : ['Premium Rice Price Prediction Chart','Medium Rice Price Prediction Chart', 'Actual', 'Prediction']
   }
]
#emojis = https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title = "Aplikasi Prediksi Harga Beras",
                   page_icon = "📈")

lang = None
with st.sidebar:
  if "lang_choose" not in st.session_state:
    st.session_state.lang_choose = "Indonesia"
  
  for i in range(len(language[0]['lang'])):
    if st.session_state.lang_choose == language[i]['lang'][i]:
      st.session_state.label = f"**{language[i]['nav'][1]}**"
      st.session_state.index = i
      st.session_state.title = language[i]['sdbar_title']
      lang = i
    continue
  st.title(st.session_state.title)

  if "label" not in st.session_state:
    st.session_state.label = f"**{language[0]['nav'][1]}**"
  if "index" not in st.session_state:
    st.session_state.index = 0

  st.radio(
    st.session_state.label, 
    language[0]['lang'], 
    key="lang_choose", 
    index=st.session_state.index, 
  )

  st.radio(
    f"**{language[lang]['nav'][0]}**",
    language[lang]['list_data'], 
    key="data_choose", 
    index=0, 
  )

st.title(f"{language[lang]['title']}")
def add_space(n_space):
  for i in range(n_space):
    st.text(" ") 
add_space(2)

df_beras = pd.read_excel("datasets/export-eceran.xlsx", sheet_name = 1)
df_beras = df_beras.rename(columns={' Komoditas (Rp) ': 'tanggal', 'Beras Premium' : 'beras_premium', 'Beras Medium': 'beras_medium'})

def data_prep(data):
  data['tanggal'] = pd.to_datetime(data['tanggal'], dayfirst=True)
  # beras_medium['tanggal'] = pd.to_datetime(beras_medium['tanggal'], dayfirst=True)
  data[data.columns[1]] = data[data.columns[1]].replace('-', np.nan)
  # beras_medium[['beras_medium']] = beras_medium[['beras_medium']].replace('-', np.nan)
  data.set_index('tanggal', inplace=True)
  # beras_medium.set_index('tanggal', inplace=True)
  data = data.asfreq('D')
  # beras_medium = beras_medium.asfreq('D') 
  data[data.columns[0]] = data[data.columns[0]].interpolate(method='linear')
  # beras_medium['beras_medium'] = beras_medium['beras_medium'].interpolate(method='linear')
  data = data.astype({data.columns[0]: int})
  # beras_medium = beras_medium.astype({'beras_medium': int})
  return data

data_idx = language[lang]['list_data'].index(st.session_state.data_choose)
if(data_idx == 0):
   beras_premium = data_prep(df_beras[['tanggal', 'beras_premium']].iloc[:-14].copy())
   beras_premium1 = data_prep(df_beras[['tanggal', 'beras_premium']].copy())
   beras_medium = data_prep(df_beras[['tanggal', 'beras_medium']].iloc[:-14].copy())
   beras_medium1 = data_prep(df_beras[['tanggal', 'beras_medium']].copy())
elif(data_idx == 1):
  beras_premium = data_prep(df_beras[['tanggal', 'beras_premium']].copy())
  beras_medium = data_prep(df_beras[['tanggal', 'beras_medium']].copy())

first_date = beras_premium.iloc[0].name
last_date = beras_premium.iloc[-1].name
rice_type = language[lang]['rice_type']
col1, col2, col3 = st.columns(3)
default_date = last_date
with col1 : 
    selected_rice = st.selectbox(language[lang]['sl_box'][0], rice_type)
with col2 :
    start_date = st.date_input(language[lang]['sl_box'][1], value = default_date.replace(day = 1),
                               min_value = first_date,
                               max_value=last_date)
with col3:
    end_date = st.date_input(language[lang]['sl_box'][2], value = default_date,
                             min_value = first_date,
                             max_value = last_date)

def load_datesets(ticker, date1, date2):
  rice_index = language[lang]['rice_type'].index(ticker)
  if rice_index == 0:
    final_data = beras_premium.copy()
    data = beras_premium.loc[date1 : date2].copy() 
    final_data['beras_premium'], lmd = stats.boxcox(beras_premium['beras_premium'])
    rice_type = 'premium'
  elif rice_index == 1:
    final_data = beras_medium.copy()
    data = beras_medium.loc[date1 : date2].copy()
    final_data['beras_medium'], lmd = stats.boxcox(beras_medium['beras_medium'])
    rice_type = 'medium'
    
  return data, final_data, ticker, rice_type, lmd, rice_index

data, final_data, name, rice_type, lmd, rice_index = load_datesets(selected_rice, start_date, end_date)

data.index = data.index.strftime('%Y-%m-%d')
data = data.reset_index()
data = data.rename(columns={'tanggal':language[lang]['table'][0], data.columns[1] : name})
load_data_state = st.text("")

if(data is not None):
    load_data_state.text(language[lang]['load_state'][0])
else:
    load_data_state.text(language[lang]['load_state'][1])
add_space(1)

st.subheader(language[lang]['header'][0])
add_space(2)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    rows = 8
    st.dataframe(data, height = rows * 35 + 3, width = 400)

fig = go.Figure()
fig.add_trace(go.Scatter(x = data.iloc[:,0],
                         y = data.iloc[:,1], 
                         name = data.columns[1],
                         line=dict(color='#2B60DE')))
fig.layout.update(title_text = language[lang]['actual_chart'][rice_index],
                  xaxis=dict(tickformat='%Y-%m-%d'), 
                  showlegend = True,
                  legend=dict(orientation="h",
                              yanchor="bottom",  
                              y = 1,
                              xanchor="center",
                              x = 0.1,
                              font = dict(size = 14)))
st.plotly_chart(fig)

##Forecasting
add_space(1)
st.subheader(language[lang]['header'][1])
add_space(1)
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

def make_data_direct(data, window_size, n_steps):
  x1, y1 = [], []
  for i in range(len(data) - window_size - n_steps + 1):
    x1.append(data[i:(i + window_size)])
    y1.append(data[(i + window_size):(i + window_size + n_steps)])

  return np.array(x1), np.array(y1)

def direct_lstm_pred(models, x, scaler, n_steps):
  # predict
  direct_pred = np.zeros((x.shape[0], n_steps))
  pred_status = st.text(language[lang]['pred_status'][0])
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
      bar.progress(j)
      progress_status.write(str(j) + " %")
    j_temp = all_progress
  pred_status.text(language[lang]['pred_status'][1])
  direct_pred = scaler.inverse_transform(direct_pred)

  return direct_pred

def residualForLstm(actual, pred, scaller):
  temp = actual.copy()
  temp['pred'] = pred
  temp['residual'] = temp.iloc[:, 0] - temp.iloc[:, 1]
  residual = pd.DataFrame(temp['residual'][1:], columns=['residual'])
  resid_scaled = scaller.fit_transform(residual)

  return resid_scaled

def make_data_input_only(data, window_size):
  x_inp = []
  for i in range(len(data) - window_size + 1):
    x_inp.append(data[i:(i + window_size)])

  return np.array(x_inp)

def hybrid_model_predict_final(sm_model, lm_model, X, scaller):
  sm_pred= sm_model.predict(start = X.index[0], end = X.index[-1])
  sm_forecast= sm_model.forecast(steps=14)
  sm_forecast = inv_boxcox(sm_forecast, lmd)
  data_temp = inv_boxcox(X, lmd)
  data_temp['pred'] = inv_boxcox(sm_pred, lmd)
  data_temp['residual'] = data_temp.iloc[:, 0] - data_temp.iloc[:, 1]
  residual = pd.DataFrame(data_temp['residual'][1:], columns=['residual'])
  resid_sc = scaller.fit_transform(residual)
  x1 = make_data_input_only(resid_sc, window_size)
  direct_pred = direct_lstm_pred(lm_model, x1, scaller, steps)

  direct_pred1 = direct_pred[direct_pred.shape[0]-1:]
  direct_pred1 = direct_pred1.transpose()
  all_pred = sm_forecast.to_frame()
  all_pred['resid_pred'] = direct_pred1
  all_pred['result'] =  all_pred.iloc[:,0] + all_pred.iloc[:,1]

  return all_pred['result'].to_frame()

sc, sarima_model, lstm_model = load_model_final()
pred = hybrid_model_predict_final(sarima_model, lstm_model, final_data.iloc[:-14], sc)
final_pred = pred
# final_pred = pred[pred.shape[0]-1:]
# final_pred = final_pred.transpose() 
last_date = final_data.index.max()
new_date = pd.date_range(start = last_date + timedelta(days = 1), periods = steps)
df_pred = pd.DataFrame({f'{final_data.columns[0]}': final_pred.iloc[:,0].values}, index = new_date)
if (data_idx == 0):
  if (rice_type == 'premium'):
    df_pred[language[lang]['table'][2]] = beras_premium1.iloc[-14:]    
  elif (rice_type == 'medium'):
    df_pred[language[lang]['table'][2]] = beras_medium1.iloc[-14:]    
else:
   df_pred[language[lang]['table'][2]] = np.nan
add_space(1)
col1, col2, col3 = st.columns([1,2,1])
with col2:
  rows = 15
  df_pred2 = df_pred.copy()
  df_pred2.index = df_pred2.index.strftime('%Y-%m-%d')
  df_pred2 = df_pred2.reset_index()
  df_pred2 = df_pred2.rename(columns={'index' : language[lang]['table'][0], df_pred2.columns[1] : language[lang]['table'][1]})
  st.dataframe(df_pred2, height = rows * 35 + 3, width = 400)
add_space(1)
fig = go.Figure()
subset_data = final_data.copy()
subset_data[subset_data.columns[0]] = inv_boxcox(subset_data[subset_data.columns[0]], lmd)
if (data_idx == 0 ):
  if (rice_type == 'premium'):
    subset_data = beras_premium1
  elif (rice_type == 'medium'):
    subset_data = beras_medium1
subset_data = subset_data[-30:]
fig.add_trace(go.Scatter(x = subset_data.index, 
                         y = subset_data.iloc[:,0], 
                         name = language[lang]['pred_chart'][2],
                         line=dict(color='#2B60DE')))
fig.add_trace(go.Scatter(x = df_pred.index, 
                         y = df_pred.iloc[:,0], 
                         name = language[lang]['pred_chart'][3],
                         line=dict(color='#50C878')))
fig.layout.update(title_text = language[lang]['pred_chart'][rice_index], 
                  xaxis=dict(
                     tickformat='%Y-%m-%d',
                  ), 
                  showlegend = True,
                  legend=dict(orientation="h",
                              yanchor="bottom",  
                              y = 1,
                              xanchor="center",
                              x = 0.1,
                              font = dict(size = 14)))
st.plotly_chart(fig)

def pred_error(actual, pred):
  mae = mean_absolute_error(actual, pred)
  mape = mean_absolute_percentage_error(actual, pred) * 100
  return mae, mape

if (data_idx == 0):
  mae, mape = pred_error(df_pred2.iloc[:,2], df_pred2.iloc[:,1])
  mae = mae.round(3)
  mape = mape.round(3)
else :
  mae, mape = '-', '-'
  
st.subheader(language[lang]['header'][2])
st.write(f"**Mean Absolute Error** : {mae}")
st.write(f"**Mean Absolute Percentage Error** : {mape}%")

st.write('testing streamlit')