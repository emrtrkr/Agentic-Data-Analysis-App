import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# Import our custom modules
from config_models import (
    get_available_models, get_model_info, 
    GROQ_API_KEY, OLLAMA_AVAILABLE, groq_client,
    is_data_analysis_query, get_translation_status
)
from data_analysis_engine import (
    code_generation_agent, execution_agent, reasoning_agent, 
    data_insight_agent, safe_json_serialize, analyze_chart_data
)
from data_preprocessing import preprocess_dataframe
from manager_report import (
    display_report_button, track_dataset_upload, 
    track_analysis_query, track_preprocessing
)

def apply_custom_css():
    """Apply professional dark theme styling"""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: #0f1419;
        color: #ffffff;
    }
    
    .main > div {
        background: rgba(15, 20, 25, 0.95);
        border: 1px solid #2d3748;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        padding: 1.5rem;
        border-radius: 12px;
        color: #ffffff;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.1);
    }
    
    .metric-card h4 {
        color: #00ffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #ffffff;
        margin: 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #00ffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    .analysis-result {
        background: rgba(0, 20, 40, 0.8);
        border: 1px solid #00ffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ffff 0%, #0080ff 100%);
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.4);
    }
    
    .stSidebar {
        background: rgba(15, 20, 25, 0.95);
        border-right: 1px solid #2d3748;
    }
    
    .stSidebar > div {
        background: transparent;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stMarkdown, .stText {
        color: #ffffff;
    }
    
    .stDataFrame {
        background: rgba(15, 20, 25, 0.8);
        border: 1px solid #2d3748;
        border-radius: 8px;
    }
    
    .stChatMessage {
        background: rgba(15, 20, 25, 0.8);
        border: 1px solid #2d3748;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .stChatInput > div > div > input {
        background: rgba(15, 20, 25, 0.9);
        border: 1px solid #00ffff;
        color: #ffffff;
        border-radius: 8px;
    }
    
    .stFileUploader > div {
        background: rgba(15, 20, 25, 0.8);
        border: 2px dashed #00ffff;
        border-radius: 12px;
        color: #ffffff;
    }
    
    .stSelectbox > div > div {
        background-color: #000000;
    }
    
    details {
        background: rgba(15, 20, 25, 0.6);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    details summary {
        color: #00ffff;
        font-weight: 600;
        cursor: pointer;
    }
    
    pre, code {
        background: #1a202c !important;
        color: #00ffff !important;
        border: 1px solid #2d3748;
        border-radius: 4px;
    }
    
    .warning-box {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid #ff6b6b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)

def clear_chat():
    """Clear chat history and reset session state"""
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'plots' in st.session_state:
        del st.session_state.plots
    if 'plot_data_history' in st.session_state:
        del st.session_state.plot_data_history
    plt.close('all')  # Close all matplotlib figures
    st.rerun()

def validate_query(query: str) -> bool:
    """Validate if query is appropriate for data analysis"""
    if not query or len(query.strip()) < 3:
        return False
    
    return is_data_analysis_query(query)

def display_model_sidebar():
    """Display enhanced model selection sidebar"""
    st.markdown("### 🤖 AI Model Seçimi")
    
    # Model selection with enhanced UI
    available_models = get_available_models()
    
    # Set default model if not set
    if 'selected_model' not in st.session_state and available_models:
        # Prefer Ollama if available, otherwise Groq
        ollama_models = [m for m in available_models if "Ollama" in m and not any(keyword in m for keyword in ["Error", "Failed", "Not Installed"])]
        if ollama_models:
            st.session_state.selected_model = ollama_models[0]
        else:
            st.session_state.selected_model = available_models[0]
    
    # Handle model switch requests
    if 'model_switch_target' in st.session_state:
        target_model = st.session_state['model_switch_target']
        del st.session_state['model_switch_target']
        st.session_state.selected_model = target_model
        st.rerun()
    
    selected_model = st.selectbox(
        "💻 Aktif Model:",
        available_models,
        key='selected_model',
        help="AI modelini değiştirmek için seçin"
    )
    
    # Model info display
    model_info = get_model_info(selected_model)
    
    if model_info["type"] != "none":
        st.markdown(f"""
        <div style='background: rgba(0, 255, 255, 0.1); border: 1px solid #00ffff; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;'>
            <h4 style='color: #00ffff; margin: 0 0 0.5rem 0;'>📊 Model Bilgisi</h4>
            <p style='margin: 0.2rem 0; color: #ffffff;'><strong>Hız:</strong> {model_info["speed"]}</p>
            <p style='margin: 0.2rem 0; color: #ffffff;'><strong>Maliyet:</strong> {model_info["cost"]}</p>
            <p style='margin: 0.2rem 0; color: #ffffff;'><strong>RAM:</strong> {model_info["ram"]}</p>
            <p style='margin: 0.2rem 0; color: #ffffff;'><strong>İnternet:</strong> {model_info["internet"]}</p>
        </div>
        """, unsafe_allow_html=True)

def display_home_page():
    """Display comprehensive home page with data analysis examples"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #00ffff; margin-bottom: 1rem; text-shadow: 0 0 10px #00ffff;'>TuckerMind</h1>
        <p style='font-size: 1.3rem; color: #ffffff; margin: 1rem 0; font-weight: 300;'>
            Yapay Zeka Destekli Profesyonel Veri Analizi Platformu
        </p>
        <div style='background: linear-gradient(135deg, #00ffff20 0%, #ff00ff10 100%); border: 1px solid #00ffff; border-radius: 15px; padding: 1.5rem; margin: 2rem auto; max-width: 800px;'>
            <h3 style='color: #00ffff; margin-bottom: 1rem;'>✨ Platform Özellikleri</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                <div style='text-align: center; padding: 1rem;'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🤖</div>
                    <h4 style='color: #ffffff; margin: 0.5rem 0;'>AI Destekli Analiz</h4>
                    <p style='color: #cccccc; font-size: 0.9rem;'>Doğal dille sorularınızı sorun, AI anında analiz yapsın</p>
                </div>
                <div style='text-align: center; padding: 1rem;'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
                    <h4 style='color: #ffffff; margin: 0.5rem 0;'>İsteme Özel Grafikler</h4>
                    <p style='color: #cccccc; font-size: 0.9rem;'>İstediğinize göre profesyonel grafikler oluşturulur ve otomatik analiz yapılır</p>
                </div>
                <div style='text-align: center; padding: 1rem;'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🔧</div>
                    <h4 style='color: #ffffff; margin: 0.5rem 0;'>Akıllı Veri Temizleme</h4>
                    <p style='color: #cccccc; font-size: 0.9rem;'>Verileriniz para sembolleri, yüzde işaretleri, eksik değerler gibi problemlerden otomatik temizlenir ve analiz için optimize edilir</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data analysis examples section
    st.markdown("""
    <div style='margin: 3rem 0;'>
        <h2 style='text-align: center; color: #00ffff; margin-bottom: 2rem;'>📈 Hangi Veri Türlerini Analiz Edebilirsiniz?</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create example visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales Data Example
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #00ffff; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;'>
            <h3 style='color: #00ffff; text-align: center; margin-bottom: 0.5rem;'>💰 Satış Verileri</h3>
            <p style='color: #ffffff; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;'>Gelir trendleri, ürün performansı, müşteri segmentasyonu</p>
        """, unsafe_allow_html=True)
        
        # Create sample sales chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a202c')
        ax.set_facecolor('#1a202c')
        
        # Sample sales data
        months = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz']
        sales_2023 = [150000, 165000, 180000, 175000, 200000, 220000]
        sales_2024 = [180000, 195000, 210000, 205000, 235000, 250000]
        
        x = np.arange(len(months))
        ax.plot(x, sales_2023, marker='o', linewidth=2, color='#ff6b6b', label='2023', markersize=6)
        ax.plot(x, sales_2024, marker='s', linewidth=2, color='#00ffff', label='2024', markersize=6)
        
        ax.set_xticks(x)
        ax.set_xticklabels(months, color='white', fontsize=8)
        ax.set_ylabel('Satış (₺)', color='white', fontsize=9)
        ax.set_title('Aylık Satış Trendi', color='white', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, color='#333333')
        
        # Style the plot
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.tick_params(colors='white', labelsize=8)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Marketing Data Example  
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #ff00ff; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;'>
            <h3 style='color: #ff00ff; text-align: center; margin-bottom: 0.5rem;'>📢 Pazarlama Verileri</h3>
            <p style='color: #ffffff; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;'>Kampanya performansı, ROI analizi, kanal etkinliği</p>
        """, unsafe_allow_html=True)
        
        # Create sample marketing chart
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a202c')
        ax.set_facecolor('#1a202c')
        
        # Sample marketing data
        channels = ['Google Ads', 'Facebook', 'Instagram', 'Email', 'Organic']
        roi = [3.2, 2.8, 4.1, 5.5, 2.1]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        
        bars = ax.bar(channels, roi, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, roi):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{value}x', ha='center', va='bottom', color='white', fontweight='bold', fontsize=8)
        
        ax.set_ylabel('ROI Oranı', color='white', fontsize=9)
        ax.set_title('Pazarlama Kanalları ROI Analizi', color='white', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, color='#333333', axis='y')
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.tick_params(colors='white', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Financial Data Example
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #ffff00; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;'>
            <h3 style='color: #ffff00; text-align: center; margin-bottom: 0.5rem;'>💳 Finansal Veriler</h3>
            <p style='color: #ffffff; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;'>Gelir-gider analizi, karlılık raporları, bütçe takibi</p>
        """, unsafe_allow_html=True)
        
        # Create sample financial chart
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a202c')
        ax.set_facecolor('#1a202c')
        
        # Sample financial data
        categories = ['Gelir', 'Operasyonel\nGiderler', 'Pazarlama', 'AR-GE', 'Net Kar']
        values = [1000000, -400000, -150000, -100000, 350000]
        colors = ['#00ff00', '#ff6b6b', '#ff6b6b', '#ff6b6b', '#00ffff']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            label = f'₺{abs(value):,.0f}'
            if value >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000, 
                       label, ha='center', va='bottom', color='white', fontweight='bold', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 20000, 
                       label, ha='center', va='top', color='white', fontweight='bold', fontsize=8)
        
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.set_ylabel('Tutar (₺)', color='white', fontsize=9)
        ax.set_title('Aylık Finansal Özet', color='white', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, color='#333333', axis='y')
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.tick_params(colors='white', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Customer Data Example
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #00ff00; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;'>
            <h3 style='color: #00ff00; text-align: center; margin-bottom: 0.5rem;'>👥 Müşteri Verileri</h3>
            <p style='color: #ffffff; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;'>Demografik analiz, davranış kalıpları, segmentasyon</p>
        """, unsafe_allow_html=True)
        
        # Create sample customer chart - SAME SIZE AS OTHERS
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1a202c')
        ax.set_facecolor('#1a202c')
        
        # Sample customer data
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        customers = [1200, 2800, 3200, 1800, 900]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(customers, labels=age_groups, colors=colors, autopct='%1.1f%%',
                                         startangle=90, pctdistance=0.85)
        
        # Make it a donut chart
        centre_circle = plt.Circle((0,0), 0.60, fc='#1a202c')
        fig.gca().add_artist(centre_circle)
        
        # Style the text - SMALLER FONTS TO MATCH OTHERS
        for text in texts:
            text.set_color('white')
            text.set_fontsize(7)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(6)
        
        ax.set_title('Müşteri Yaş Dağılımı', color='white', fontsize=10, fontweight='bold', pad=10)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional data types section
    st.markdown("""
    <div style='margin: 3rem 0;'>
        <h2 style='text-align: center; color: #00ffff; margin-bottom: 2rem;'>🎯 Daha Fazla Veri Türü Desteği</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 2rem 0;'>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #ff6b6b; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #ff6b6b;'>🏥</div>
                <h3 style='color: #ff6b6b; margin-bottom: 1rem;'>Sağlık Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Hasta demografikleri, tedavi sonuçları, hastane kapasitesi, epidemi takibi</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #4ecdc4; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #4ecdc4;'>🚚</div>
                <h3 style='color: #4ecdc4; margin-bottom: 1rem;'>Lojistik Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Teslimat süreleri, rota optimizasyonu, maliyet analizi, stok takibi</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #feca57; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #feca57;'>🏭</div>
                <h3 style='color: #feca57; margin-bottom: 1rem;'>Üretim Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Kalite kontrol, makine performansı, üretim kapasitesi, hata analizi</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #a8e6cf; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #a8e6cf;'>🎓</div>
                <h3 style='color: #a8e6cf; margin-bottom: 1rem;'>Eğitim Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Öğrenci performansı, sınav sonuçları, devam durumu, başarı analizi</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #ff9ff3; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #ff9ff3;'>🌐</div>
                <h3 style='color: #ff9ff3; margin-bottom: 1rem;'>Web Analitik</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Ziyaretçi trafiği, dönüşüm oranları, sayfa performansı, kullanıcı davranışı</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #74b9ff; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #74b9ff;'>📱</div>
                <h3 style='color: #74b9ff; margin-bottom: 1rem;'>Sosyal Medya</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Engagement analizi, takipçi büyümesi, içerik performansı, sentiment analizi</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #fd79a8; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #fd79a8;'>🛒</div>
                <h3 style='color: #fd79a8; margin-bottom: 1rem;'>Perakende Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Satış analizi, müşteri davranışları, sezonsal trendler, mağaza performansı</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #00b894; border-radius: 12px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem; color: #00b894;'>📦</div>
                <h3 style='color: #00b894; margin-bottom: 1rem;'>Stok Verileri</h3>
                <p style='color: #ffffff; line-height: 1.6;'>Envanter takibi, stok döngüleri, tedarik zinciri analizi, sipariş optimizasyonu</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("""
    <div style='margin: 3rem 0;'>
        <h2 style='text-align: center; color: #00ffff; margin-bottom: 2rem;'>🚀 Nasıl Çalışır?</h2>
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border: 1px solid #00ffff; border-radius: 15px; padding: 2rem; margin: 2rem 0;'>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;'>
                <div style='text-align: center;'>
                    <div style='background: #00ffff; color: #000; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: bold;'>1</div>
                    <h4 style='color: #00ffff; margin-bottom: 1rem;'>📁 Veri Yükle</h4>
                    <p style='color: #ffffff; font-size: 0.9rem;'>CSV dosyanızı sürükleyip bırakın veya seçin</p>
                </div>
                <div style='text-align: center;'>
                    <div style='background: #00ffff; color: #000; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: bold;'>2</div>
                    <h4 style='color: #00ffff; margin-bottom: 1rem;'>🔧 Otomatik Temizleme</h4>
                    <p style='color: #ffffff; font-size: 0.9rem;'>Yapay zeka verilerinizi otomatik olarak temizler, para sembolleri ve yüzde işaretlerini düzeltir, eksik değerleri doldurur ve veri tiplerini optimize eder</p>
                </div>
                <div style='text-align: center;'>
                    <div style='background: #00ffff; color: #000; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: bold;'>3</div>
                    <h4 style='color: #00ffff; margin-bottom: 1rem;'>💬 Soru Sor</h4>
                    <p style='color: #ffffff; font-size: 0.9rem;'>Doğal dille istediğiniz analizi isteyin</p>
                </div>
                <div style='text-align: center;'>
                    <div style='background: #00ffff; color: #000; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: bold;'>4</div>
                    <h4 style='color: #00ffff; margin-bottom: 1rem;'>📊 Sonuçları Gör</h4>
                    <p style='color: #ffffff; font-size: 0.9rem;'>Anında grafikler ve analiz sonuçları alın</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_home_page_no_file():
    """Display home page when no file is loaded"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h1 style='color: #00ffff; margin-bottom: 1rem; text-shadow: 0 0 10px #00ffff; font-size: 3rem;'>TuckerMind</h1>
        <p style='font-size: 1.4rem; color: #ffffff; margin: 2rem 0; font-weight: 300;'>
            Yapay Zeka ile Veri Analizinde Yeni Bir Çağ
        </p>
        <div style='background: linear-gradient(135deg, #00ffff20 0%, #ff00ff10 100%); border: 2px solid #00ffff; border-radius: 20px; padding: 3rem; margin: 3rem auto; max-width: 900px;'>
            <h2 style='color: #00ffff; margin-bottom: 2rem;'>📊 Veri Analizini Yeniden Tanımlıyoruz</h2>
            <p style='font-size: 1.2rem; color: #ffffff; line-height: 1.8; margin-bottom: 2rem;'>
                Programlama bilgisine ihtiyaç duymadan, sadece doğal dille sorularınızı sorun. 
                Yapay zeka sizin için profesyonel analizler yapsın, grafikler oluştursun.
            </p>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; margin-top: 2rem;'>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
                    <h4 style='color: #ffffff; margin-bottom: 0.5rem;'>AI Destekli</h4>
                    <p style='color: #cccccc;'>Gelişmiş yapay zeka modelleri</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>⚡</div>
                    <h4 style='color: #ffffff; margin-bottom: 0.5rem;'>Hızlı Sonuç</h4>
                    <p style='color: #cccccc;'>Saniyeler içinde analiz</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🎨</div>
                    <h4 style='color: #ffffff; margin-bottom: 0.5rem;'>Güzel Grafikler</h4>
                    <p style='color: #cccccc;'>Profesyonel görselleştirmeler</p>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🔧</div>
                    <h4 style='color: #ffffff; margin-bottom: 0.5rem;'>Otomatik Temizleme</h4>
                    <p style='color: #cccccc;'>Veri ön işleme otomatiği</p>
                </div>
            </div>
        </div>
        
        <div style='text-align: center; margin: 3rem 0; padding: 2rem; background: linear-gradient(135deg, #00ffff10 0%, #ff00ff10 100%); border: 2px solid #00ffff; border-radius: 20px;'>
            <h2 style='color: #00ffff; margin-bottom: 1rem;'>🎯 Başlamaya Hazır mısınız?</h2>
            <p style='color: #ffffff; font-size: 1.1rem; margin-bottom: 1.5rem;'>
                Sol panelden CSV dosyanızı yükleyin ve yapay zeka destekli veri analizinin gücünü keşfedin.
            </p>
            <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>
                <div style='background: rgba(0, 255, 255, 0.2); padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #00ffff;'>
                    <span style='color: #00ffff; font-weight: bold;'>✨ Kod Gerektirmez</span>
                </div>
                <div style='background: rgba(255, 0, 255, 0.2); padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #ff00ff;'>
                    <span style='color: #ff00ff; font-weight: bold;'>🚀 Anında Sonuç</span>
                </div>
                <div style='background: rgba(255, 255, 0, 0.2); padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #ffff00;'>
                    <span style='color: #ffff00; font-weight: bold;'>🎨 Profesyonel Grafikler</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the main chat interface for data analysis"""
    # Display report button in top right
    display_report_button()
    
    st.header("💬 Yapay Zeka Destekli Veri Analizi")
    
    # Display preprocessing status if applicable
    if st.session_state.preprocessing_done:
        st.success("✅ İşlenmiş veri seti kullanılıyor - Veri ön işleme sekmesinde detayları görebilirsiniz")
    
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            
            # Handle plot display
            if msg.get("plot_index") is not None:
                idx = msg["plot_index"]
                if 0 <= idx < len(st.session_state.plots):
                    # Display the plot
                    st.pyplot(st.session_state.plots[idx], use_container_width=True)
                    
                    # Show enhanced analysis for plots
                    if idx < len(st.session_state.plot_data_history) and st.session_state.plot_data_history[idx]:
                        try:
                            # Get user query from previous message
                            prev_msg = st.session_state.messages[i-1] if i > 0 else None
                            if prev_msg and prev_msg.get("role") == "user":
                                analysis = analyze_chart_data(st.session_state.plot_data_history[idx], prev_msg["content"])
                                st.markdown(f"""
                                <div class='analysis-result'>
                                    <h4>📊 Uzman Görselleştirme Analizi</h4>
                                    {analysis}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Analiz hatası: {e}")
                    
                    # Show technical details
                    with st.expander("🔬 Teknik Detaylar ve Veriler"):
                        if idx < len(st.session_state.plot_data_history) and st.session_state.plot_data_history[idx]:
                            try:
                                safe_plot_data = safe_json_serialize(st.session_state.plot_data_history[idx])
                                st.json(safe_plot_data)
                            except Exception:
                                st.write("Plot data görüntülenemiyor")
            
            # Handle non-plot results
            elif msg.get("result_data") is not None:
                with st.expander("🔬 Analiz Detayları"):
                    if isinstance(msg["result_data"], (pd.DataFrame, pd.Series)):
                        st.dataframe(msg["result_data"])
                    else:
                        st.write(f"**Sonuç:** {msg['result_data']}")
                    if msg.get("code_used"):
                        st.code(msg["code_used"], language='python')

    # Chat input with validation
    if user_query := st.chat_input("Veri setiniz hakkında profesyonel bir analiz sorusu sorun..."):
        
        # Validate query
        if not validate_query(user_query):
            st.error("⚠️ Lütfen veri setinizle ilgili bir analiz sorusu sorun. Genel sorular ve veri analizi dışındaki konular desteklenmemektedir.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.spinner("🧠 Uzman analiz yapılıyor..."):
            try:
                # Generate and execute code
                code, should_plot, _ = code_generation_agent(user_query, st.session_state.df)
                
                if not code:
                    st.error("Kod üretimi başarısız oldu. Lütfen soruyu yeniden ifade edin.")
                    return
                
                result, plot_data = execution_agent(code, st.session_state.df, should_plot)
                
            except Exception as e:
                st.error(f"Analiz sırasında hata oluştu: {e}")
                return

        # Handle plot results
        plot_idx = None
        if hasattr(result, 'figure') or str(type(result)) == "<class 'matplotlib.figure.Figure'>":
            fig = result.figure if hasattr(result, 'figure') else result
            
            # Store plot in session state immediately
            st.session_state.plots.append(fig)
            st.session_state.plot_data_history.append(plot_data)
            plot_idx = len(st.session_state.plots) - 1
            
            # Track analysis for reporting
            chart_analysis = ""
            if plot_data:
                try:
                    chart_analysis = analyze_chart_data(plot_data, user_query)
                except:
                    chart_analysis = "Grafik analizi yapıldı"
            
            track_analysis_query(
                question=user_query,
                result_type="visualization", 
                summary=f"Grafik oluşturuldu: {type(result).__name__}",
                chart_analysis=chart_analysis
            )
            
            # Clear the current figure to prevent interference
            plt.close('all')
            
            # Store message first
            st.session_state.messages.append({
                "role": "assistant",
                "content": "📊 **Profesyonel Görselleştirme Oluşturuldu**",
                "plot_index": plot_idx
            })
            
            st.rerun()
            
        else:
            # For non-plot results, generate reasoning
            reasoning = reasoning_agent(user_query, result, plot_data)
            
            # Track analysis for reporting
            track_analysis_query(
                question=user_query,
                result_type="statistical",
                summary=reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            )
            
            # Store message first
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""
                <div class='analysis-result'>
                    <h4>📊 Analiz Sonucu</h4>
                    {reasoning}
                </div>
                """,
                "result_data": result,
                "code_used": code
            })
            
            st.rerun()

def display_preprocessing_interface():
    """Display the preprocessing interface"""
    st.header("🔧 AI-Powered Veri Ön İşleme")
    
    if st.session_state.raw_df is None:
        st.warning("⚠️ Önce bir veri seti yükleyin.")
        return
    
    # Show data quality analysis
    raw_df = st.session_state.raw_df
    
    st.markdown("### 📊 Veri Kalitesi Analizi")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        missing_count = raw_df.isnull().sum().sum()
        st.metric("Eksik Değer", f"{missing_count:,}")
    
    with col2:
        duplicate_count = raw_df.duplicated().sum()
        st.metric("Duplikasyon", f"{duplicate_count:,}")
    
    with col3:
        object_cols = len([col for col in raw_df.columns if str(raw_df[col].dtype) == 'object'])
        st.metric("Object Sütun", f"{object_cols}")
    
    with col4:
        memory_mb = raw_df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Hafıza", f"{memory_mb:.1f}MB")
    
    # Missing data details
    if missing_count > 0:
        st.markdown("### 🔍 Eksik Veri Detayları")
        missing_info = []
        for col in raw_df.columns:
            missing = raw_df[col].isnull().sum()
            if missing > 0:
                pct = (missing / len(raw_df)) * 100
                missing_info.append({
                    "Sütun": col,
                    "Eksik Sayı": missing,
                    "Eksik %": f"{pct:.1f}%",
                    "Veri Tipi": str(raw_df[col].dtype)
                })
        
        if missing_info:
            missing_df = pd.DataFrame(missing_info)
            st.dataframe(missing_df, use_container_width=True)
    
    # Preprocessing controls
    st.markdown("### 🤖 AI Veri Ön İşleme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Veri Ön İşleme Başlat", type="primary", use_container_width=True):
            with st.spinner("🤖 AI ile veri ön işleme yapılıyor..."):
                try:
                    processed_df, preprocessing_report = preprocess_dataframe(raw_df)
                    
                    # Update session state
                    st.session_state.df = processed_df
                    st.session_state.preprocessing_report = preprocessing_report
                    st.session_state.preprocessing_done = True
                    
                    # Track preprocessing for reporting
                    track_preprocessing(preprocessing_report)
                    
                    # Regenerate insights with processed data
                    st.session_state.insights = data_insight_agent(processed_df)
                    
                    # Clear chat history since data changed
                    st.session_state.messages = []
                    st.session_state.plots = []
                    st.session_state.plot_data_history = []
                    
                    st.success("✅ Veri ön işleme tamamlandı!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Veri ön işleme hatası: {e}")
    
    with col2:
        if st.session_state.preprocessing_done:
            if st.button("🔄 Orijinal Veriye Dön", use_container_width=True):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.preprocessing_done = False
                st.session_state.preprocessing_report = ""
                st.session_state.insights = data_insight_agent(st.session_state.df)
                st.session_state.messages = []
                st.session_state.plots = []
                st.session_state.plot_data_history = []
                st.success("✅ Orijinal veriye dönüldü!")
                st.rerun()
    
    # Show preprocessing report
    if st.session_state.preprocessing_done and st.session_state.preprocessing_report:
        st.markdown("### 📋 Veri Ön İşleme Raporu")
        st.markdown(f"""
        <div class='insight-box'>
            {st.session_state.preprocessing_report}
        </div>
        """, unsafe_allow_html=True)
        
        # Download processed data
        st.markdown("### 💾 İşlenmiş Veri İndirme")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_data = convert_df_to_csv(st.session_state.df)
        
        st.download_button(
            label="📥 İşlenmiş Veri Setini İndir (CSV)",
            data=csv_data,
            file_name=f"processed_{st.session_state.current_file}",
            mime="text/csv",
            use_container_width=True
        )
        
        # Show before/after comparison
        st.markdown("### 📊 Karşılaştırma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Orijinal Veri**")
            st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
            
            # Add original data info
            st.markdown("**📋 Orijinal Veri Tipleri**")
            
            # Create df.info() equivalent table for raw data
            raw_info_data = []
            for i, col in enumerate(st.session_state.raw_df.columns):
                dtype = str(st.session_state.raw_df[col].dtype)
                non_null = st.session_state.raw_df[col].count()
                null_count = st.session_state.raw_df[col].isnull().sum()
                memory_usage = st.session_state.raw_df[col].memory_usage(deep=True)
                
                raw_info_data.append({
                    "#": i,
                    "Sütun": col,
                    "Non-Null Count": f"{non_null} non-null",
                    "Dtype": dtype,
                    "Null Count": null_count,
                    "Memory": f"{memory_usage} bytes"
                })
            
            raw_info_df = pd.DataFrame(raw_info_data)
            st.dataframe(raw_info_df, use_container_width=True)
        
        with col2:
            st.markdown("**🟢 İşlenmiş Veri**")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # Add processed data info
            st.markdown("**📋 İşlenmiş Veri Tipleri**")
            
            # Create df.info() equivalent table for processed data
            processed_info_data = []
            for i, col in enumerate(st.session_state.df.columns):
                dtype = str(st.session_state.df[col].dtype)
                non_null = st.session_state.df[col].count()
                null_count = st.session_state.df[col].isnull().sum()
                memory_usage = st.session_state.df[col].memory_usage(deep=True)
                
                processed_info_data.append({
                    "#": i,
                    "Sütun": col,
                    "Non-Null Count": f"{non_null} non-null",
                    "Dtype": dtype,
                    "Null Count": null_count,
                    "Memory": f"{memory_usage} bytes"
                })
            
            processed_info_df = pd.DataFrame(processed_info_data)
            st.dataframe(processed_info_df, use_container_width=True)
        
        # Summary comparison table
        st.markdown("### 🔄 Veri Tipi Değişiklikleri")
        
        changes_data = []
        for col in st.session_state.df.columns:
            if col in st.session_state.raw_df.columns:
                old_type = str(st.session_state.raw_df[col].dtype)
                new_type = str(st.session_state.df[col].dtype)
                
                if old_type != new_type:
                    status = "🔄 Değişti"
                else:
                    status = "✅ Aynı"
                
                changes_data.append({
                    "📋 Sütun": col,
                    "🔴 Orijinal Tip": old_type,
                    "🟢 İşlenmiş Tip": new_type,
                    "📊 Durum": status
                })
        
        if changes_data:
            changes_df = pd.DataFrame(changes_data)
            st.dataframe(changes_df, use_container_width=True)
        else:
            st.info("📊 Veri tiplerinde değişiklik yapılmadı.")

def main():
    st.set_page_config(
        page_title="TuckerMind",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Initialize session state
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "plot_data_history" not in st.session_state:
        st.session_state.plot_data_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "insights" not in st.session_state:
        st.session_state.insights = ""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "preprocessing_done" not in st.session_state:
        st.session_state.preprocessing_done = False
    if "preprocessing_report" not in st.session_state:
        st.session_state.preprocessing_report = ""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None

    # Sidebar for model selection and file upload
    with st.sidebar:
        display_model_sidebar()
        
        st.divider()
        st.header("📁 Veri Yükleme")
        
        file = st.file_uploader("CSV Dosyası Seçin", type=["csv"], help="Analiz edilecek veri setini yükleyin")
        
        if file:
            # Load and cache dataset
            if ("raw_df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                with st.spinner("🔄 Veri seti yükleniyor..."):
                    try:
                        # Load raw data
                        st.session_state.raw_df = pd.read_csv(file)
                        st.session_state.current_file = file.name
                        
                        # Track dataset upload for reporting
                        track_dataset_upload(st.session_state.raw_df, file.name)
                        
                        # Initially use raw data for analysis
                        if not st.session_state.preprocessing_done:
                            st.session_state.df = st.session_state.raw_df.copy()
                        
                        # Reset states when new file is loaded
                        st.session_state.messages = []
                        st.session_state.plots = []
                        st.session_state.plot_data_history = []
                        
                        # Generate insights for current dataframe
                        st.session_state.insights = data_insight_agent(st.session_state.df)
                        
                    except Exception as e:
                        st.error(f"Dosya yükleme hatası: {e}")
                        return
            
            # Dataset metrics
            df = st.session_state.df
            st.markdown("### 📈 Veri Seti Özeti")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📊 Satır Sayısı</h4>
                    <h2>{len(df):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📋 Sütun Sayısı</h4>
                    <h2>{len(df.columns)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            st.markdown("### 👀 Veri Önizleme")
            st.dataframe(df.head(), use_container_width=True)
            
            # Dataset insights
            if st.session_state.insights:
                st.markdown("### 🔍 Veri Setine İlk Bakış")
                st.markdown(f"""
                <div class='insight-box'>
                    {st.session_state.insights}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("📊 Veri analizi yükleniyor...")
            
            # Chat controls
            st.markdown("### 🗂️ Sohbet Kontrolleri")
            if st.button("🗑️ Sohbeti Temizle", help="Tüm sohbet geçmişini sil", use_container_width=True):
                clear_chat()
        
        else:
            st.info("📤 Profesyonel analiz için bir CSV dosyası yükleyin")

    # Main content area with tabs - ALWAYS SHOW TABS
    # Create tabs with Home tab added
    tab1, tab2, tab3 = st.tabs(["🏠 Ana Sayfa", "💬 Veri Analizi", "🔧 Veri Ön İşleme"])
    
    with tab1:
        display_home_page()
    
    with tab2:
        if file and 'df' in st.session_state and st.session_state.df is not None:
            display_chat_interface()
        else:
            st.info("📊 Veri analizi için önce sol panelden bir CSV dosyası yükleyin.")
    
    with tab3:
        if file and 'df' in st.session_state and st.session_state.df is not None:
            display_preprocessing_interface()
        else:
            st.info("🔧 Veri ön işleme için önce sol panelden bir CSV dosyası yükleyin.")

if __name__ == "__main__":
    main()