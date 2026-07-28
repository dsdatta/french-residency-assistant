import streamlit as st
from generate_result import answer_query

st.set_page_config(page_title="French Residency Assistant", page_icon="FR")
st.title("🇫🇷 French Residency Assistant")
st.markdown(
    "Ask any question about French residency documents, and I'll answer based on official sources."
)


query = st.text_input(
    "Your question:", placeholder="e.g. What is the validity length of carte de séjour?"
)

if st.button("Ask") and query:
    with st.spinner("Searching documents and generating answer..."):
        answer = answer_query(query)
    st.markdown("### Answer")
    st.write(answer)
