"""
Streamlit frontend for the TTB label compliance tool.

Reuses the existing pipeline (triage -> extract -> validate -> report)
unchanged -- this file is presentation only. Results are persisted via
app.storage the same way main.py does, so a check made here shows up in
scripts/view_results.py too.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from app.batch import process_single_application
from app.storage import save_result

load_dotenv()

st.set_page_config(page_title="TTB Label Compliance Check", page_icon="🏷️")

st.title("TTB Label Compliance Check")
st.caption(
    "Upload a label image and its expected application data to check "
    "compliance against the 7 required TTB fields."
)

uploaded_image = st.file_uploader("Label image", type=["jpg", "jpeg", "png"])

st.subheader("Expected application data")
col1, col2 = st.columns(2)
with col1:
    brand_name = st.text_input("Brand name")
    class_type = st.text_input("Class/type designation")
    alcohol_content = st.text_input("Alcohol content (e.g. '45%')")
    net_contents = st.text_input("Net contents (e.g. '750 mL')")
with col2:
    bottler_name_addr = st.text_input("Bottler name & address")
    country_of_origin = st.text_input("Country of origin (imports only)")
    is_import = st.checkbox("This is an imported product")
    abv_required = st.checkbox("Alcohol content must be stated", value=True)

government_warning = st.text_area(
    "Government warning statement (full text, including the header)",
    height=100,
)

if st.button("Check compliance", type="primary", disabled=uploaded_image is None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_image.getvalue())
        tmp_path = tmp.name

    expected = {
        "brand_name": brand_name or None,
        "class_type": class_type or None,
        "alcohol_content": alcohol_content or None,
        "net_contents": net_contents or None,
        "bottler_name_addr": bottler_name_addr or None,
        "country_of_origin": country_of_origin or None,
        "government_warning": government_warning or None,
        "government_warning_header": "GOVERNMENT WARNING:",
    }
    context = {"is_import": is_import, "abv_required": abv_required}

    try:
        with st.spinner("Checking label..."):
            result = process_single_application(tmp_path, expected, context)
            save_result(result)
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong while processing this label: {e}")
        os.unlink(tmp_path)
        st.stop()

    os.unlink(tmp_path)

    if result.get("overall") == "approve":
        st.success("✅ Approved — all checks passed.")
    else:
        st.warning("⚠️ Flagged for review.")

    st.image(uploaded_image, caption="Submitted label", width=300)

    if "fields" in result:
        st.subheader("Field-by-field results")
        for f in result["fields"]:
            icon = {"pass": "✅", "fail": "❌", "not_applicable": "➖"}.get(f["status"], "❓")
            with st.expander(f"{icon} {f['field'].replace('_', ' ').title()}"):
                st.write(f["reason"])
                if f["status"] == "fail":
                    st.text(f"Extracted: {f['extracted']!r}")
                    st.text(f"Expected: {f['expected']!r}")
    else:
        st.write(result.get("reason", "No further details available."))