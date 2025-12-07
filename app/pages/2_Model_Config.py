"""
Model configuration page.
"""

import streamlit as st
import yaml
import json
from pathlib import Path

st.set_page_config(page_title="Model Config", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Configuration")

st.markdown("""
View and edit model configurations. Changes can be exported for training.
""")

# Configuration editor
config_type = st.selectbox(
    "Configuration Type",
    options=["Training Config", "AWS A10 Config", "Data Config"]
)

if config_type == "Training Config":
    config_path = Path("training/train_config.yaml")
elif config_type == "AWS A10 Config":
    config_path = Path("configs/aws_g5_a10.yaml")
else:
    config_path = Path("configs/data_config.yaml")

# Load config
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    st.subheader("Current Configuration")

    # Display as JSON for editing
    config_json = json.dumps(config, indent=2)
    edited_config = st.text_area(
        "Edit Configuration (JSON format)",
        value=config_json,
        height=400
    )

    # Validate and save
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Validate Config"):
            try:
                new_config = json.loads(edited_config)
                st.success("✓ Configuration is valid!")
                st.json(new_config)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    with col2:
        if st.button("Export Config"):
            st.download_button(
                label="Download YAML",
                data=yaml.dump(json.loads(edited_config)),
                file_name=config_path.name,
                mime="text/yaml"
            )
else:
    st.error(f"Configuration file not found: {config_path}")
