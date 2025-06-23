import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Obesity Classification App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        color: black;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Obesity Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis kesehatan dan prediksi klasifikasi obesitas menggunakan Machine Learning</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 Informasi Aplikasi")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Tujuan Aplikasi</h4>
        <p>Aplikasi ini menggunakan algoritma Machine Learning untuk memprediksi klasifikasi obesitas berdasarkan faktor-faktor kesehatan dan gaya hidup.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Klasifikasi Obesitas</h4>
        <ul style="margin: 0;">
            <li><b>Insufficient Weight:</b> Berat badan kurang</li>
            <li><b>Normal Weight:</b> Berat badan normal</li>
            <li><b>Overweight Level I:</b> Kelebihan berat badan tingkat 1</li>
            <li><b>Overweight Level II:</b> Kelebihan berat badan tingkat 2</li>
            <li><b>Obesity Type I:</b> Obesitas tipe 1</li>
            <li><b>Obesity Type II:</b> Obesitas tipe 2</li>
            <li><b>Obesity Type III:</b> Obesitas tipe 3</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Zaneta Fransiske - 2702312146**")
    st.markdown("---")
    st.markdown("**LC09 - UAS Model Deployment**")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Data Kesehatan")
    
    with st.form(key='predict_form'):
        st.subheader("👤 Informasi Pribadi")
        col_personal1, col_personal2 = st.columns(2)
        
        with col_personal1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], help="Pilih jenis kelamin Anda")
            age = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=25, help="Masukkan usia dalam tahun")
        
        with col_personal2:
            height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01, help="Masukkan tinggi badan dalam meter")
            weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.1, help="Masukkan berat badan dalam kilogram")
        
        st.markdown("---")
        
        st.subheader("🍽️ Riwayat Keluarga & Kebiasaan Makan")
        col_food1, col_food2 = st.columns(2)
        
        with col_food1:
            family_history = st.selectbox("Riwayat keluarga dengan kelebihan berat badan", ["yes", "no"], help="Apakah ada riwayat keluarga dengan kelebihan berat badan?")
            favc = st.selectbox("Konsumsi makanan tinggi kalori", ["yes", "no"], help="Apakah sering mengonsumsi makanan tinggi kalori?")
            fcvc = st.slider("Frekuensi konsumsi sayuran", 1.0, 3.0, 2.0, step=0.1, help="1: Jarang, 2: Kadang-kadang, 3: Sering")
        
        with col_food2:
            ncp = st.slider("Jumlah makanan utama per hari", 1.0, 4.0, 3.0, step=0.1, help="Berapa kali makan utama dalam sehari?")
            caec = st.selectbox("Konsumsi makanan di antara waktu makan", ["no", "Sometimes", "Frequently", "Always"], help="Seberapa sering ngemil di antara waktu makan?")
            ch2o = st.slider("Konsumsi air harian (liter)", 1.0, 3.0, 2.0, step=0.1, help="Berapa liter air yang diminum per hari?")
        
        st.markdown("---")
        
        st.subheader("🏃‍♂️ Gaya Hidup")
        col_lifestyle1, col_lifestyle2 = st.columns(2)
        
        with col_lifestyle1:
            smoke = st.selectbox("Apakah Anda merokok?", ["no", "yes"], help="Status merokok Anda")
            scc = st.selectbox("Monitoring konsumsi kalori", ["yes", "no"], help="Apakah Anda memantau asupan kalori?")
            faf = st.slider("Frekuensi aktivitas fisik", 0.0, 3.0, 1.0, step=0.1, help="0: Tidak pernah, 1: Jarang, 2: Kadang-kadang, 3: Sering")
        
        with col_lifestyle2:
            tue = st.slider("Waktu penggunaan teknologi (jam/hari)", 0.0, 3.0, 1.0, step=0.1, help="Berapa jam menggunakan perangkat teknologi per hari?")
            calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"], help="Seberapa sering mengonsumsi alkohol?")
            mtrans = st.selectbox("Transportasi yang digunakan", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"], help="Jenis transportasi utama yang digunakan")
        
        st.markdown("---")
        
        submit_button = st.form_submit_button(label='🔍 Prediksi Klasifikasi Obesitas')

with col2:
    st.header("📊 Hasil Analisis")
    
    if height > 0 and weight > 0:
        bmi = weight / (height ** 2)
        
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "#3498db"
        elif 18.5 <= bmi < 25:
            bmi_status = "Normal"
            bmi_color = "#2ecc71"
        elif 25 <= bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "#f39c12"
        else:
            bmi_status = "Obese"
            bmi_color = "#e74c3c"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Body Mass Index (BMI)</h3>
            <h2 style="color: white;">{bmi:.1f}</h2>
            <p style="color: #f0f0f0;">Status: {bmi_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bmi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BMI Gauge"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': bmi_color},
                'steps': [
                    {'range': [0, 18.5], 'color': "lightblue"},
                    {'range': [18.5, 25], 'color': "lightgreen"},
                    {'range': [25, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

if submit_button:
    with st.spinner('🔄 Sedang memproses prediksi...'):
        data = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                st.success("✅ Prediksi berhasil!")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                    <h2 style="color: white; margin-bottom: 1rem;">🎯 Hasil Prediksi</h2>
                    <h1 style="color: #fff; font-size: 2.5rem; margin: 0;">{prediction}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                if "Normal" in prediction:
                    st.info("🎉 Selamat! Berat badan Anda dalam kategori normal. Pertahankan gaya hidup sehat!")
                elif "Insufficient" in prediction:
                    st.warning("⚠️ Berat badan Anda kurang. Konsultasikan dengan ahli gizi untuk program penambahan berat badan yang sehat.")
                else:
                    st.error("🚨 Perhatian! Berat badan Anda dalam kategori kelebihan/obesitas. Disarankan untuk konsultasi dengan dokter atau ahli gizi.")
                
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction': prediction,
                    'bmi': bmi
                })
                
            else:
                st.error(f"❌ Error dari server: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("❌ Timeout: Server tidak merespons dalam waktu yang ditentukan.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Error: Tidak dapat terhubung ke server. Pastikan server API berjalan di http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Gagal menghubungi API: {str(e)}")

st.markdown("---")
col_footer1, col_footer2 = st.columns(2)

with col_footer1:
    st.markdown("**🔒 Privasi**")
    st.markdown("Data Anda aman dan tidak disimpan secara permanen")

with col_footer2:
    st.markdown("**📈 Akurasi**")
    st.markdown("Model dilatih dengan akurasi 96%")

if 'prediction_history' in st.session_state and st.session_state.prediction_history:
    with st.sidebar:
        st.markdown("---")
        st.header("📈 Riwayat Prediksi")
        for i, record in enumerate(reversed(st.session_state.prediction_history[-5:])): 
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; color:black;">
                <small>{record['timestamp']}</small><br>
                <b>{record['prediction']}</b><br>
                <small>BMI: {record['bmi']:.1f}</small>
            </div>
            """, unsafe_allow_html=True)