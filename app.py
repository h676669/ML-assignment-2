# for å kunne bruke dette lokalt, kjør: streamlit run app.py
# for å installere alt riktig aktiver py inviroment: .venv\Scripts\Activate.ps1
# så installer streamlit: pip install streamlit
# og pandas og numpy: pip install pandas numpy

import streamlit as st

st.set_page_config(
    page_title="Alzheimer's Risk Assessment",
    page_icon="🧠",
)

st.title("Alzheimer's Risk Assessment")
st.write("Welcome! This application provides two models for assessing the likelihood of Alzheimer's disease.")
st.sidebar.success("Select an assessment tool.")

st.markdown(
    """
    Please choose the appropriate assessment tool from the sidebar on the left:

    - **General User Assessment:** A simplified model based on general health and lifestyle factors.
    - **Healthcare Professional Assessment:** A more comprehensive model that includes detailed medical data.

    **Disclaimer:** This is a machine learning prediction and not a medical diagnosis. Please consult a healthcare professional for any health concerns.
    """
)
