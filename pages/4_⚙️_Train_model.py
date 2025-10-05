import streamlit as st
from recutils.model import train_model, has_model

st.title("⚙️ Train rating model")
st.markdown("Trains a LightGBM regressor to predict your 1–5⭐ from audio features.")

if st.button("Train / Retrain"):
    n = train_model()

    if n == 0:
        st.warning("Not enough rated tracks (need ~50+). Rate more and try again.")
    else:
        st.success(f"Trained on {n} rated tracks.")

st.write("Model present:", has_model())
st.caption("After training, the 'Similar to…' page can re-rank neighbors by predicted stars.")
