import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import glob

BASE_DIR = Path(__file__).parent
CLEAN_DATA = BASE_DIR / "CleanData" / "CleanedDataSet" / "cleaned_autos.csv"
DATA_FOR_ANALYSIS = BASE_DIR / "CleanData" / "DataForAnalysis"

st.set_page_config(page_title="Used Car Analysis", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA, encoding="latin-1")


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Used Car Analysis")
st.sidebar.markdown("Select an analysis to view:")

analysis = st.sidebar.radio(
    "Analyses",
    options=[
        "Analysis 1 – Vehicle Distribution & Price by Type",
        "Analysis 2 – Brand Count & Price by Gearbox",
        "Analysis 3 – Price & Power by Fuel Type",
        "Analysis 4 – Price Heatmap (Brand × Type)",
        "Analysis 5 – Days Online by Brand",
    ],
    label_visibility="collapsed",
)

df = load_data()
sns.set(style="white")


# ── Analysis 1 ────────────────────────────────────────────────────────────────
if analysis.startswith("Analysis 1"):
    st.title("Analysis 1 – Vehicle Distribution & Price by Type")

    st.subheader("Distribution of Vehicles by Year of Registration")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["yearOfRegistration"], color="#33cc33", kde=True, ax=ax)
    ax.set_title("Distribution of Vehicles Based on Year of Registration", fontsize=15)
    ax.set_ylabel("Count / Density (KDE)", fontsize=13)
    ax.set_xlabel("Year of Registration", fontsize=13)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Price Range by Vehicle Type (after removing outliers)")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="vehicleType", y="price", data=df, ax=ax)
    ax.set_title("Price Distribution by Vehicle Type", fontsize=15)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Total Count of Vehicles by Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df["vehicleType"].value_counts().index
    sns.countplot(x="vehicleType", data=df, palette="BuPu", order=order, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + 0.1, p.get_height() + 200))
    ax.set_title("Count of Vehicles by Type", fontsize=15)
    st.pyplot(fig)
    plt.close(fig)


# ── Analysis 2 ────────────────────────────────────────────────────────────────
elif analysis.startswith("Analysis 2"):
    st.title("Analysis 2 – Brand Count & Price by Gearbox")

    st.subheader("Number of Vehicles by Brand Available on eBay")
    fig, ax = plt.subplots(figsize=(9, 10))
    order = df["brand"].value_counts().index
    sns.countplot(y="brand", data=df, palette="Reds_r", order=order, ax=ax)
    ax.set_title("Count of Vehicles by Brand", fontsize=18)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Price by Vehicle Type and Gearbox")
    colors = ["#00e600", "#ff8c1a", "#a180cc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="vehicleType", y="price", hue="gearbox", palette=colors, data=df, ax=ax)
    ax.set_title("Average Price of Vehicles by Vehicle Type and Gearbox", fontsize=13)
    st.pyplot(fig)
    plt.close(fig)


# ── Analysis 3 ────────────────────────────────────────────────────────────────
elif analysis.startswith("Analysis 3"):
    st.title("Analysis 3 – Price & Power by Fuel Type")

    st.subheader("Average Price by Fuel Type and Gearbox")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="fuelType", y="price", hue="gearbox", palette="husl", data=df, ax=ax)
    ax.set_title("Average Price of Vehicles by Fuel Type and Gearbox", fontsize=13)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Power (PS) by Vehicle Type and Gearbox")
    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_palette(sns.xkcd_palette(colors))
    sns.barplot(x="vehicleType", y="powerPS", hue="gearbox", data=df, ax=ax)
    ax.set_title("Average Power of Vehicles by Vehicle Type and Gearbox", fontsize=13)
    st.pyplot(fig)
    plt.close(fig)


# ── Analysis 4 ────────────────────────────────────────────────────────────────
elif analysis.startswith("Analysis 4"):
    st.title("Analysis 4 – Price Heatmap (Brand × Vehicle Type)")

    st.info("This chart may take a moment to render due to its size.")

    rows = []
    for b in df["brand"].unique():
        for v in df["vehicleType"].unique():
            avg = df[(df["brand"] == b) & (df["vehicleType"] == v)]["price"].mean()
            rows.append({"brand": b, "vehicleType": v, "avgPrice": int(avg) if not pd.isna(avg) else 0})
    trial = pd.DataFrame(rows)
    tri = trial.pivot(index="brand", columns="vehicleType", values="avgPrice").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(15, 20))
    sns.heatmap(tri, linewidths=1, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
    ax.set_title("Average Price of Vehicles by Brand and Vehicle Type", fontsize=18)
    st.pyplot(fig)
    plt.close(fig)


# ── Analysis 5 ────────────────────────────────────────────────────────────────
elif analysis.startswith("Analysis 5"):
    st.title("Analysis 5 – Days Online Before Sale by Brand")

    brands = sorted([d.name for d in DATA_FOR_ANALYSIS.iterdir() if d.is_dir()])
    selected_brand = st.selectbox("Select a car brand:", brands)

    if st.button("Run Analysis"):
        brand_path = DATA_FOR_ANALYSIS / selected_brand
        all_files = list(brand_path.glob("*.csv"))

        if not all_files:
            st.error(f"No CSV files found for brand: {selected_brand}")
        else:
            frames = [pd.read_csv(f, index_col=None, header=0) for f in all_files]
            frame = pd.concat(frames, ignore_index=True)

            if "NoOfDaysOnline" not in frame.columns:
                st.error("Column 'NoOfDaysOnline' not found in data for this brand.")
            else:
                colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.set_palette(sns.xkcd_palette(colors))
                sns.stripplot(
                    x="vehicleType", y="NoOfDaysOnline", hue="gearbox",
                    dodge=True, data=frame, size=8, alpha=0.5, jitter=True, ax=ax
                )
                ax.set_title(
                    f"No. of Days Online Before Sale – {selected_brand.replace('_', ' ').title()}",
                    fontsize=13
                )
                st.pyplot(fig)
                plt.close(fig)
