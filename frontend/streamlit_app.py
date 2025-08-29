"""
Streamlit-based frontend for the Enterprise Document Q&A Assistant.
Backend is started separately (e.g., via run_app.py).
"""

import streamlit as st
import requests
import pandas as pd
from io import BytesIO

# ---------------------------
# Backend configuration
# ---------------------------
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000   # ✅ Backend runs here
SERVER_IP = "192.168.0.230"  # LAN IP for coworkers
BACKEND_URL = f"http://{SERVER_IP}:{FASTAPI_PORT}"


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="Enterprise Document Q&A Assistant",
        page_icon="🔍",
        layout="centered"
    )

    st.title("🏢 Enterprise Document Q&A Assistant")
    st.markdown("""
    Ask questions about your company's documents (Invoices, Purchase Orders, Shipping Orders, etc.)

    This system uses Retrieval-Augmented Generation to find relevant information from your document repository.
    """)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "sources" in message and message["sources"]:
                    with st.expander("📄 Sources"):
                        for source in message["sources"]:
                            st.write(f"- **{source['filename']}** ({source['category']})")

                # Show structured data with download options
                if "structured_data" in message and message["structured_data"]:
                    df = pd.DataFrame(message["structured_data"])
                    st.dataframe(df)

                    # Excel export
                    buffer = BytesIO()
                    df.to_excel(buffer, index=False, engine="openpyxl")
                    buffer.seek(0)
                    st.download_button(
                        label="⬇ Download Excel",
                        data=buffer,
                        file_name="structured_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # CSV export
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download CSV",
                        data=csv_data,
                        file_name="structured_data.csv",
                        mime="text/csv"
                    )

    # Input box
    prompt = st.chat_input("Ask a question about your company documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    placeholder.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result.get("sources", []),
                        "structured_data": result.get("structured_data", [])
                    })

                else:
                    err = f"Error {response.status_code}: {response.text}"
                    placeholder.markdown(f"❌ {err}")
                    st.session_state.messages.append({"role": "assistant", "content": err})

            except Exception as e:
                placeholder.markdown(f"❌ Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": str(e)})


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
