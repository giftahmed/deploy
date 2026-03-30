import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="Flood Forecasting ANN", page_icon="🌊", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; background-color: #0066cc; color: white; }
    .stSuccess { background-color: #d4edda; }
    .stError { background-color: #f8d7da; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌊 Flood Forecasting in Lokoja (ANN Model)")
st.markdown("---")
st.write("Enter monthly hydrological parameters to predict flood occurrence.")

# Load model with error handling
@st.cache_resource
def load_flood_model():
    try:
        model = load_model("flood_ann_model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model, scaler = load_flood_model()

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input(
        "🌧️ Rainfall (mm)", 
        min_value=0.0, 
        max_value=50000.0, 
        value=100.0,
        step=1.0,
        help="Monthly rainfall in millimeters"
    )

with col2:
    water_level = st.number_input(
        "📊 Water Level (m)", 
        min_value=0.0, 
        max_value=50000.0, 
        value=5.0,
        step=0.1,
        help="Average water level in meters"
    )

with col3:
    discharge = st.number_input(
        "💧 Discharge (cumecs)", 
        min_value=0.0, 
        max_value=50000.0, 
        value=500.0,
        step=10.0,
        help="Average discharge in cubic meters per second"
    )

# Validation check
if rainfall == 0 and water_level == 0 and discharge == 0:
    st.warning("⚠️ Please enter values greater than zero for accurate prediction.")

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Flood", type="primary"):
    try:
        # Prepare input
        input_data = np.array([[rainfall, water_level, discharge]])
        input_scaled = scaler.transform(input_data)
        
        # Prediction
        prob = model.predict(input_scaled, verbose=0)[0][0]
        
        # Determine result
        is_flood = prob > 0.5
        
        # Display results in columns
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Flood Probability", f"{prob:.4f}", f"{(prob-0.5)*100:+.1f}% vs threshold")
        
        with res_col2:
            confidence = "HIGH" if abs(prob - 0.5) > 0.3 else "MEDIUM" if abs(prob - 0.5) > 0.15 else "LOW"
            st.metric("Confidence Level", confidence)
        
        # Result message
        st.markdown("---")
        
        if is_flood:
            st.error(f"""
                ### 🌊 FLOOD RISK DETECTED
                
                **Probability:** {prob:.2%}  
                **Risk Level:** {"CRITICAL" if prob > 0.8 else "HIGH" if prob > 0.7 else "MODERATE"}
                
                ⚠️ **Recommendation:** Activate flood preparedness protocols immediately.
            """)
        else:
            st.success(f"""
                ### ✅ NO FLOOD RISK
                
                **Probability:** {prob:.2%}  
                **Safety Margin:** {(0.5-prob)*100:.1f}% below threshold
                
                ✓ **Status:** Conditions are within normal parameters.
            """)
        
        # Progress bar visualization
        st.progress(float(prob))
        st.caption(f"Flood Risk Meter: {prob:.1%}")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Artificial Neural Network (ANN) | Lokoja Flood Forecasting System")