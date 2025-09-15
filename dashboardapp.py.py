import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline
from fpdf import FPDF
# ---------------------------------------
# 🎨 Page Config + Background Gradient
# ---------------------------------------
st.set_page_config(page_title="Auto Analytica Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

         .stApp {
            background-color: #000000;
            color: white;  /* Optional: makes default text white for readability */
        }
    </style>
""", unsafe_allow_html=True)

# Load Hugging Face Summarizer
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.warning(f"Summarizer failed: {e}")
        return None

summarizer = load_summarizer()

def sanitize_for_arrow(df):
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if pd.api.types.is_integer_dtype(df_fixed[col].dtype):
            df_fixed[col] = df_fixed[col].astype("int64")
        elif pd.api.types.is_float_dtype(df_fixed[col].dtype):
            df_fixed[col] = df_fixed[col].astype("float64")
        elif pd.api.types.is_bool_dtype(df_fixed[col].dtype):
            df_fixed[col] = df_fixed[col].astype("bool")
        else:
            df_fixed[col] = df_fixed[col].astype("object")
    return df_fixed

def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return sanitize_for_arrow(df)
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

def show_upload_card():
    st.markdown("""
    <h1 style="
        background: linear-gradient(45deg, #ff6a00, #ee0979, #ff512f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        text-align: center;
        margin-bottom: 20px;
    "> 🚀 Upload Your Dataset </h1>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Upload A CSV File  📂", type=["csv"], key="custom_uploader")
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
        return uploaded_file


def show_basic_info(df):
    st.write(f"**Shape:** `{df.shape[0]} rows × {df.shape[1]} columns`")
    st.dataframe(df.head(10))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column Data Types")
        st.dataframe(df.dtypes.astype(str))
    with col2:
        st.subheader("Unique Values per Column")
        st.dataframe(df.nunique())
    st.subheader('Memory Usage Check In Bytes')
    st.dataframe(df.memory_usage(deep=True))

def show_summary_stats(df):
    st.subheader("Summary Statistics")
    try:
        st.dataframe(df.describe(include="all").transpose())
    except Exception:
        st.warning("⚠️ Could not compute summary statistics for all columns.")
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0])
    else:
        st.info("✅ No missing values detected.")


def show_correlation_heatmap(df):
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap="magma", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation.")


def show_categorical_value_counts(df: pd.DataFrame, top_n=10):
    st.subheader("Top Value Counts for Categorical Columns")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns detected.")
        return
    for col in cat_cols:
        series = df[col].value_counts(dropna=False).head(top_n)
        st.markdown(f"**Column:** `{col}`")
        st.dataframe(series)
 

def plot_column(df, column):
    st.subheader(f"Visualization: {column}")
    chart_type = st.radio(
        "Choose chart type",
        ["Bar", "Histogram", "Line", "Pie", "Scatter", "Box", "Violin", "Pairplot"],
        horizontal=True
    )

    if chart_type == "Bar" and df[column].dtype == "object":
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        st.plotly_chart(px.bar(counts, x=column, y="count", text="count"))

    elif chart_type == "Histogram" and np.issubdtype(df[column].dtype, np.number):
        st.plotly_chart(px.histogram(df, x=column, nbins=30))

    elif chart_type == "Line" and np.issubdtype(df[column].dtype, np.number):
        st.plotly_chart(px.line(df, y=column))

    elif chart_type == "Pie" and df[column].dtype == "object":
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        st.plotly_chart(px.pie(counts, names=column, values="count"))

    elif chart_type == "Scatter":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)
            st.plotly_chart(px.scatter(df, x=x_col, y=y_col, color=column if df[column].dtype == "object" else None))
        else:
            st.info("Need at least two numeric columns.")

    elif chart_type == "Box":
        st.plotly_chart(px.box(df, y=column))

    elif chart_type == "Violin":
        st.plotly_chart(px.violin(df, y=column, box=True, points="all"))

    elif chart_type == "Pairplot":
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            st.info("⚡ Generating pairplot (might take time)...")
            fig = sns.pairplot(num_df)
            st.pyplot(fig)
        else:
            st.info("Need at least two numeric columns.")

    elif chart_type == "Map":
        lat_cols = [c for c in df.columns if "lat" in c.lower()]
        lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
        if lat_cols and lon_cols:
            st.plotly_chart(px.scatter_mapbox(
                df, lat=lat_cols[0], lon=lon_cols[0], zoom=3, mapbox_style="carto-positron",
                hover_name=column if column in df.columns else None
            ))
        else:
            st.info("No latitude/longitude columns found.")

# ---------------------------------------
# 🧠 Insights
# ---------------------------------------

def generate_nlp_insight(df, return_text=False):
    insights = []
    # Health Checks
    # -------------------------
    health_checks = []
    if df.duplicated().sum() > 0:
        health_checks.append(f"Dataset has {df.duplicated().sum()} duplicate rows.")
    if df.isnull().sum().sum() > 0:
        health_checks.append(f"Dataset has {df.isnull().sum().sum()} missing values.")
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        health_checks.append(f"Columns with constant values: {', '.join(constant_cols)}")
    if not health_checks:
        health_checks.append("No major data health issues detected.")
    insights.append("### Health Checks:\n" + "\n".join(f"- {h}" for h in health_checks))
    # Numerical Insights
    # -------------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        insights.append(f"### Numerical Columns ({len(num_cols)} found):\n- " + ", ".join(num_cols))
        num_insights = []
        for col in num_cols[:5]:  # Limit to first 5 to keep summary short
            num_insights.append(
                f"{col}: "
                f"Mean={df[col].mean():.2f}, "
                f"Median={df[col].median():.2f}, "
                f"Std={df[col].std():.2f}, "
                f"Min={df[col].min():.2f}, "
                f"Max={df[col].max():.2f}"
            )
        insights.append("### Numerical Insights (sample up to 5 columns):\n" + "\n".join(f"- {n}" for n in num_insights))
    else:
        insights.append("### Numerical Insights:\n- No numerical columns found.")

    # Categorical Insights
    # -------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        insights.append(f"### Categorical Columns ({len(cat_cols)} found):\n- " + ", ".join(cat_cols))

        cat_insights = []
        for col in cat_cols[:5]:  # Limit to first 5 columns for brevity
            top_vals = df[col].value_counts().head(3)
            cat_insights.append(
                f"{col}: Top categories → " + ", ".join([f"{i} ({v})" for i, v in top_vals.items()])
            )
        insights.append("### Categorical Insights (sample up to 5 columns):\n" + "\n".join(f"- {c}" for c in cat_insights))
    else:
        insights.append("### Categorical Insights:\n- No categorical columns found.")

    # Correlations
    # -------------------------
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        strong_corr = [
            f"{i} ↔ {j}: {corr.loc[i,j]:.2f}"
            for i in num_cols for j in num_cols
            if i != j and abs(corr.loc[i,j]) > 0.6
        ]
        if strong_corr:
            insights.append("### Correlations:\n" + "\n".join(f"- {c}" for c in strong_corr))
        else:
            insights.append("### Correlations:\n- No strong correlations detected.")
    else:
        insights.append("### Correlations:\n- Not enough numerical columns for correlation analysis.")
    # AI Narrative (HuggingFace model)
    # -------------------------
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        narrative_text = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
        narrative_text += "It includes numerical and categorical features. "
        if len(num_cols) > 0:
            narrative_text += f"Some numerical features are {', '.join(num_cols[:3])}. "
        if len(cat_cols) > 0:
            narrative_text += f"Some categorical features are {', '.join(cat_cols[:3])}. "

        ai_summary = summarizer(narrative_text, max_length=80, min_length=30, do_sample=False)
        insights.append("### A Simple Narration:\n- " + ai_summary[0]['summary_text'])
    except Exception as e:
        insights.append(f"### A Simple Narration:\n- Could not generate narrative ({e})")
    # Return result
    # -------------------------
    final_text = "\n\n".join(insights)

    if return_text:
        return final_text

    # Streamlit display
    st.subheader("🔎 NLP Insights")
    for section in insights:
        st.markdown(section)

# Report Generation 
def generate_pdf_report(df):
    import tempfile, os
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    st.subheader("📑Generate Report")
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # -------------------------
        # Font Handling
        # -------------------------
        def register_unicode_font(pdf):
            try_paths = []
            import matplotlib.font_manager as fm
            try:
                dejavu = fm.findfont("DejaVu Sans", fallback_to_default=True)
                if dejavu and os.path.exists(dejavu):
                    try_paths.append(dejavu)
            except:
                pass
            try_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arialuni.ttf",
            ])
            for p in try_paths:
                if os.path.exists(p):
                    try:
                        pdf.add_font("DejaVu", "", p, uni=True)
                        pdf.add_font("DejaVu", "B", p, uni=True)
                        pdf.set_font("DejaVu", size=12)
                        return True
                    except:
                        continue
            return False
        unicode_ok = register_unicode_font(pdf)
        if not unicode_ok:
            pdf.set_font("Arial", size=12)

        def sanitize_text(s: str):
            if s is None:
                return ""
            s = str(s)
            if unicode_ok:
                return s
            repl = {"→": "->", "•": "-", "✓": "v", "✔": "v", "–": "-", "—": "-"}
            for k, v in repl.items():
                s = s.replace(k, v)
            return s.encode("latin-1", "ignore").decode("latin-1")
        # Title
        # -------------------------
        pdf.set_font(pdf.font_family, style="B", size=16)
        pdf.cell(0, 10, sanitize_text("InsightLens Report"), ln=True, align="C")
        pdf.ln(6)
        # Dataset Overview
        # -------------------------
        rows, cols = df.shape
        pdf.set_font(pdf.font_family, style="B", size=13)
        pdf.cell(0, 8, "Dataset Overview", ln=True)
        pdf.set_font(pdf.font_family, size=11)
        pdf.multi_cell(0, 7, sanitize_text(f"Shape: {rows} rows × {cols} columns"))
        col_list = ", ".join(df.columns[:12])
        if len(df.columns) > 12:
            col_list += ", ..."
        pdf.multi_cell(0, 7, sanitize_text(f"Columns (first 12): {col_list}"))
        pdf.ln(4)
        # NLP Insights Section
        # -------------------------
        insights_str = generate_nlp_insight(df, return_text=True)
        sections = insights_str.split("\n")
        # Group by keywords
        def add_section(title, lines):
            pdf.set_font(pdf.font_family, style="B", size=13)
            pdf.cell(0, 8, sanitize_text(title), ln=True)
            pdf.set_font(pdf.font_family, size=11)
            for line in lines:
                pdf.multi_cell(0, 6, sanitize_text(f"• {line}"))
            pdf.ln(2)

        # Partition text
        num_lines = []
        cat_lines = []
        corr_lines = []
        health_lines = []
        ai_lines = []

        for line in sections:
            if line.startswith("Numerical Insights:") or "Mean" in line:
                num_lines.append(line.replace("Numerical Insights:", "").strip())
            elif line.startswith("Categorical Insights:") or ":" in line and "%" in line:
                cat_lines.append(line.replace("Categorical Insights:", "").strip())
            elif "↔" in line or "Correlation" in line:
                corr_lines.append(line.strip())
            elif "Duplicate" in line or "Missing" in line or "Constant" in line:
                health_lines.append(line.strip())
            elif "AI Narrative" in line or "dataset" in line.lower():
                ai_lines.append(line.strip())

        if health_lines:
            add_section("Health Checks", health_lines)
        if num_lines:
            add_section("Numerical Insights", num_lines)
        if cat_lines:
            add_section("Categorical Insights", cat_lines)
        if corr_lines:
            add_section("Correlations", corr_lines)
        if ai_lines:
            add_section("AI Narrative", ai_lines)
        # Visualizations
        # -------------------------
        pdf.set_font(pdf.font_family, style="B", size=13)
        pdf.cell(0, 8, "Visualizations", ln=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Histogram
            if num_cols:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[num_cols[0]].dropna(), kde=True)
                plt.title(f"Distribution of {num_cols[0]}")
                hist_path = os.path.join(tmpdir, "hist.png")
                plt.savefig(hist_path, bbox_inches="tight")
                plt.close()
                pdf.add_page()
                pdf.cell(0, 8, sanitize_text(f"Distribution of {num_cols[0]}"), ln=True)
                pdf.image(hist_path, x=20, w=170)

            # Correlation heatmap
            if len(num_cols) > 1:
                plt.figure(figsize=(6, 4))
                sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f")
                plt.title("Correlation Heatmap")
                heatmap_path = os.path.join(tmpdir, "heatmap.png")
                plt.savefig(heatmap_path, bbox_inches="tight")
                plt.close()
                pdf.add_page()
                pdf.cell(0, 8, "Correlation Heatmap", ln=True)
                pdf.image(heatmap_path, x=20, w=170)

            # Top categories
            if cat_cols:
                plt.figure(figsize=(6, 4))
                top_vals = df[cat_cols[0]].value_counts().head(10)
                sns.barplot(x=top_vals.values, y=top_vals.index)
                plt.title(f"Top Categories in {cat_cols[0]}")
                cat_path = os.path.join(tmpdir, "cat.png")
                plt.savefig(cat_path, bbox_inches="tight")
                plt.close()
                pdf.add_page()
                pdf.cell(0, 8, sanitize_text(f"Top Categories in {cat_cols[0]}"), ln=True)
                pdf.image(cat_path, x=20, w=170)
        # Save + Download
        # -------------------------
        pdf_output = "report.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            st.download_button("📥 Download Report (PDF)", f, "report.pdf", "application/pdf")

        st.download_button(
            "📥 Download Cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_data.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"PDF Report generation failed: {e}")
# 🚀 Streamlit UI
# ---------------------------------------
st.markdown("""
    <style>
        .gradient-title {
            font-size: 56px;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(270deg, #ff6a00, #ee0979, #4facfe, #43e97b, #f9d423);
            background-size: 600% 600%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: animatedGradient 10s ease infinite;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        @keyframes animatedGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @media (max-width: 768px) {
            .gradient-title {
                font-size: 36px;
            }
        }
    </style>
    <h1 class="gradient-title"> InsightLens Dashboard </h1>
""", unsafe_allow_html=True)

uploaded_file = show_upload_card()
if uploaded_file:
    df = load_csv(uploaded_file)
    if df is not None:
        tabs = st.tabs(["📄 Overview", "📊 EDA", "📈 Visualizations", "🧠 Insights", "📋 Report"])
        with tabs[0]:
            show_basic_info(df)
        with tabs[1]:
            show_summary_stats(df)
            show_categorical_value_counts(df)
            show_correlation_heatmap(df)
        with tabs[2]:
            column = st.selectbox("Choose a column to visualize", df.columns)
            if column:
                plot_column(df, column)
        with tabs[3]:
            generate_nlp_insight(df)
        with tabs[4]:
            generate_pdf_report(df)
else:
    st.info("⬅️ Upload a CSV file to get started.")