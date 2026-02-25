import streamlit as st
import pandas as pd
import joblib

# 1. Model bileşenlerini ve eğitimde kullanılan sütun listesinin yüklenmesi
model = joblib.load('isparta_ev_fiyat_modeli.pkl')
scaler = joblib.load('isparta_scaler.pkl')
sutunlar = joblib.load('model_sutunlari.pkl')

# 2. Sayfa konfigürasyonu ve başlıklar
st.title("🏡 Isparta Kiralık Ev Fiyat Tahmincisi")
st.markdown("Isparta'daki kiralık ev verileri üzerinden Ridge Regression modeli kullanılarak geliştirilmiş fiyat tahmin uygulaması.")
st.divider()

# 3. Kullanıcı girdileri (Arayüz bileşenleri)
mahalleler = [
    'Fatih', 'Çünür', 'Modern Evler', 'Bahçelievler', 'Pirimehmet', 
    'Hızırbey', 'Davraz', 'Muzaffer Türkeş', 'Yedişehitler', 'Zafer', 'İstiklal', 'Diğer'
]

col1, col2 = st.columns(2)

with col1:
    secilen_mahalle = st.selectbox("Mahalle Seçimi", mahalleler)
    metrekare = st.number_input("Metrekare (Net)", min_value=15, max_value=350, value=80)

with col2:
    oda_sayisi = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5, 6, 7], index=2)

# 4. Tahmin algoritmasının çalıştırılması
if st.button("Fiyatı Hesapla", type="primary"):
    
    # Modelin beklediği formatta (tüm sütunlar 0 olacak şekilde) bir sözlük yapısı kurulması
    input_data = {col: 0 for col in sutunlar}
    
    # Sayısal değişkenlerin atanması
    input_data['Metrekare'] = metrekare
    input_data['Toplam_Oda'] = oda_sayisi
    
    # Kategorik değişkenin (Mahalle) One-Hot Encoding formatına uygun hale getirilmesi
    mahalle_sutun_adi = f"Mahalle_{secilen_mahalle}"
    if mahalle_sutun_adi in sutunlar:
        input_data[mahalle_sutun_adi] = 1 
    
    # Veriyi DataFrame yapısına çevirip ölçeklendirme (Scaling) işleminin yapılması
    df_kullanici = pd.DataFrame([input_data])
    df_kullanici_scaled = scaler.transform(df_kullanici)
    
    # Tahmin hesaplama
    tahmin_edilen_fiyat = model.predict(df_kullanici_scaled)[0]
    
    # Sonuç ekranı
    st.success(f"Tahmini Aylık Kira Bedeli: **{int(tahmin_edilen_fiyat):,} TL**".replace(',', '.'))

