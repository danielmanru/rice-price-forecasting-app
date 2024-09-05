
# st.markdown(
#     """
#     <style>
#     .center-table{
#         display: block;
#         margin-left: auto;
#         margin-right: auto;
#         width: fit-content;
#     }
#     </style>
#     """, unsafe_allow_html=True)

import streamlit as st
from datetime import date
from plotly import graph_objs as go
import pandas as pd

# START = "2022-07-16"
today = date.today()

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
    data = None
    if ticker == 'Beras Premium':
        final_data = data = beras_premium.loc[date1 : date2].copy() 
    elif ticker == 'Beras Medium':
        final_data = data = beras_medium.loc[date1 : date2].copy()
    return data, final_data, ticker


data, final_data, name = load_datesets(selected_rice, start_date, end_date)
data.index = data.index.strftime('%Y-%m-%d')
data = data.reset_index()
data = data.rename(columns={'tanggal':'Tanggal', data.columns[1] : name})

load_data_state = st.text("")
if(data is not None):
    load_data_state.text("Data berhasil dimuat!")
else:
    load_data_state.text("Data gagal dimuat")
add_space(1)

st.subheader('Raw Data')
add_space(2)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    rows = 8
    st.dataframe(data, height = rows * 35 + 3, width = 400)

def plot_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index, y = data[data.columns[0]], name = data.columns[0]))
    fig.layout.update(title_text = "Grafik Harga Beras Harian", showlegend = True,
                      legend=dict(
                          orientation="h",  
                          yanchor="bottom",  
                          y = 1,
                          xanchor="center",
                          x = 0.1,
                          font = dict(size = 14)
    ))
    st.plotly_chart(fig)

plot_data(final_data)
        
