import streamlit as st
import pandas as pd
import joblib

# 1. Kaydettiğimiz Model, Scaler ve Sütun bilgilerini içeri aktaralım
model = joblib.load('isparta_ev_fiyat_modeli.pkl')
scaler = joblib.load('isparta_scaler.pkl')
sutunlar = joblib.load('model_sutunlari.pkl')

# 2. Web sayfasının başlığı ve açıklaması
st.title("🏡 Isparta Kiralık Ev Fiyat Tahmincisi")
st.markdown("Bu uygulama, makine öğrenmesi (Ridge Regression) kullanarak Isparta'daki evlerin tahmini kira bedellerini hesaplar.")
st.divider() # Araya şık bir çizgi çeker

# 3. Kullanıcıdan verileri alacağımız form alanları (Arayüz)
# Mahalle listemizi verideki en yoğun 11 mahalle ve 'Diğer' olarak tanımlıyoruz
mahalleler = [
    'Fatih', 'Çünür', 'Modern Evler', 'Bahçelievler', 'Pirimehmet', 
    'Hızırbey', 'Davraz', 'Muzaffer Türkeş', 'Yedişehitler', 'Zafer', 'İstiklal', 'Diğer'
]

col1, col2 = st.columns(2) # Ekranı iki sütuna bölelim daha şık dursun

with col1:
    secilen_mahalle = st.selectbox("Mahalle Seçiniz", mahalleler)
    metrekare = st.number_input("Evin Büyüklüğü (Metrekare)", min_value=15, max_value=350, value=80)

with col2:
    oda_sayisi = st.selectbox("Toplam Oda Sayısı", [1, 2, 3, 4, 5, 6, 7], index=2) # Default 3 (Yani 2+1 ev gibi)

# 4. Tahmin Butonu ve Arka Plan İşlemleri
if st.button("Fiyatı Tahmin Et", type="primary"):
    
    # Tüm model sütunlarının başlangıçta 0 olduğu bir sözlük (dictionary) yaratalım
    input_data = {col: 0 for col in sutunlar}
    
    # Kullanıcının girdiği sayısal değerleri sözlüğe ekleyelim
    input_data['Metrekare'] = metrekare
    input_data['Toplam_Oda'] = oda_sayisi
    
    # Kullanıcının seçtiği mahalleyi One-Hot Encoding formatına (1'e) çevirelim
    mahalle_sutun_adi = f"Mahalle_{secilen_mahalle}"
    if mahalle_sutun_adi in sutunlar:
        input_data[mahalle_sutun_adi] = 1 
        # Not: drop_first=True yüzünden düşen mahalle seçilirse, hiçbir sütun 1 olmaz (hepsi 0 kalır), 
        # bu matematiksel olarak tamamen doğru bir harekettir!
        
    # Sözlüğü tek satırlık bir Pandas DataFrame'ine dönüştürelim
    df_kullanici = pd.DataFrame([input_data])
    
    # 5. Modeli uygulamadan önce veriyi ölçeklendirme (Scaling)
    df_kullanici_scaled = scaler.transform(df_kullanici)
    
    # 6. Büyük An: Tahmin!
    tahmin_edilen_fiyat = model.predict(df_kullanici_scaled)[0]
    
    # Sonucu ekrana şık bir mesajla yazdıralım
    st.success(f"✨ Bu evin tahmini aylık kira bedeli: **{int(tahmin_edilen_fiyat):,} TL**".replace(',', '.'))