
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib

from pathlib import Path
from datetime import datetime


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "dataset.csv"

RECOMMENDATIONS_PATH = (
    OUTPUT_DIR / "business_recommendations.csv"
)

ACTION_PLAN_PATH = (
    OUTPUT_DIR / "action_plan.csv"
)

TRACKING_PATH = (
    OUTPUT_DIR / "action_tracking.csv"
)

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. PROFESSIONAL DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    h1, h2, h3 {
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Date",
        "Revenue",
        "Ad Spend",
        "Conversions",
        "Service Type",
        "Customer Type",
        "Channel",
        "Region"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Revenue",
        "Ad Spend",
        "Conversions"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    categorical_columns = [
        "Service Type",
        "Customer Type",
        "Channel",
        "Region"
    ]

    for col in categorical_columns:

        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
        )

    data = data.dropna(
        subset=required_columns
    )

    return data


@st.cache_data
def load_csv_output(path_string):

    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


try:

    df = load_data()

    recommendations = load_csv_output(
        str(RECOMMENDATIONS_PATH)
    )

    action_plan = load_csv_output(
        str(ACTION_PLAN_PATH)
    )

except Exception as error:

    st.error("Unable to load project data.")

    st.exception(error)

    st.stop()


if df.empty:

    st.error("The dataset has no valid records.")

    st.stop()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def safe_divide(numerator, denominator):

    if denominator == 0:
        return np.nan

    return numerator / denominator


def format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def calculate_growth(current, previous):

    if pd.isna(previous) or previous == 0:
        return None

    return (
        (current - previous) /
        abs(previous)
    ) * 100


def calculate_kpis(data):

    revenue = data["Revenue"].sum()

    spend = data["Ad Spend"].sum()

    conversions = data["Conversions"].sum()

    transactions = len(data)

    return {
        "Revenue": revenue,
        "Ad Spend": spend,
        "Conversions": conversions,
        "Transactions": transactions,

        "ROAS": safe_divide(
            revenue,
            spend
        ),

        "Average Revenue per Transaction": safe_divide(
            revenue,
            transactions
        ),

        "Revenue per Conversion": safe_divide(
            revenue,
            conversions
        )
    }


def monthly_analysis(data):

    monthly = (
        data.assign(
            Month=data["Date"].dt.to_period("M")
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Ad_Spend=("Ad Spend", "sum"),
            Conversions=("Conversions", "sum")
        )
        .reset_index()
    )

    monthly["Month"] = (
        monthly["Month"]
        .dt.to_timestamp()
    )

    monthly["ROAS"] = np.where(
        monthly["Ad_Spend"] > 0,
        monthly["Revenue"] / monthly["Ad_Spend"],
        np.nan
    )

    monthly["Revenue Growth %"] = (
        monthly["Revenue"]
        .pct_change(fill_method=None) * 100
    )

    monthly["Ad Spend Growth %"] = (
        monthly["Ad_Spend"]
        .pct_change(fill_method=None) * 100
    )

    monthly["Conversions Growth %"] = (
        monthly["Conversions"]
        .pct_change(fill_method=None) * 100
    )

    monthly["ROAS Growth %"] = (
        monthly["ROAS"]
        .pct_change(fill_method=None) * 100
    )

    monthly.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return monthly


def create_action_id(row):

    fields = [
        "Category",
        "Finding",
        "Recommendation"
    ]

    raw = "|".join(
        str(row.get(field, ""))
        for field in fields
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:12]


def show_empty_message(data):

    if data.empty:

        st.warning(
            "No data available for the selected filters."
        )

        return True

    return False


# =========================================================
# 5. ACTION TRACKING DATA
# =========================================================

def prepare_action_tracking(action_data):

    if action_data.empty:
        return pd.DataFrame()

    tracking = action_data.copy()

    if "Action ID" not in tracking.columns:

        tracking["Action ID"] = tracking.apply(
            create_action_id,
            axis=1
        )

    tracking["Action ID"] = (
        tracking["Action ID"].astype(str)
    )

    defaults = {
        "Status": "Pending",
        "Actual Result": "",
        "Notes": "",
        "Updated At": ""
    }

    for column, default in defaults.items():

        if column not in tracking.columns:
            tracking[column] = default

    if TRACKING_PATH.exists():

        try:

            saved = pd.read_csv(
                TRACKING_PATH,
                dtype={"Action ID": str}
            )

            saved_columns = [
                "Action ID",
                "Status",
                "Actual Result",
                "Notes",
                "Updated At"
            ]

            saved = saved[
                [
                    col for col in saved_columns
                    if col in saved.columns
                ]
            ]

            tracking = tracking.drop(
                columns=[
                    col for col in [
                        "Status",
                        "Actual Result",
                        "Notes",
                        "Updated At"
                    ]
                    if col in tracking.columns
                ]
            )

            tracking = tracking.merge(
                saved,
                on="Action ID",
                how="left"
            )

            for column, default in defaults.items():

                if column not in tracking.columns:
                    tracking[column] = default

                tracking[column] = (
                    tracking[column].fillna(default)
                )

        except Exception as error:

            st.warning(
                f"Could not load saved tracking data: {error}"
            )

    return tracking


def save_action_tracking(data):

    OUTPUT_DIR.mkdir(exist_ok=True)

    data.to_csv(
        TRACKING_PATH,
        index=False,
        encoding="utf-8-sig"
    )


tracking_df = prepare_action_tracking(
    action_plan
)


# =========================================================
# 6. SIDEBAR AND NAVIGATION
# =========================================================

st.sidebar.title("📊 BI Platform")

st.sidebar.caption(
    "Business Intelligence & Recommendation Engine"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Revenue Analysis",
        "Marketing Performance",
        "Customer Analysis",
        "Service & Regional Analysis",
        "Recommendations & Action Tracking",
        "Data Quality"
    ]
)


# =========================================================
# 7. GLOBAL FILTERS
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("Global Filters")

min_date = df["Date"].min().date()

max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date, end_date = date_range

elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:

    start_date = date_range[0]
    end_date = date_range[0]

else:

    start_date = min_date
    end_date = max_date


if start_date > end_date:

    st.error("Invalid date range.")

    st.stop()


date_filtered = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
].copy()


def sidebar_filter(column):

    options = sorted(
        date_filtered[column]
        .dropna()
        .unique()
        .tolist()
    )

    return st.sidebar.multiselect(
        column,
        options,
        default=options
    )


selected_regions = sidebar_filter("Region")

selected_channels = sidebar_filter("Channel")

selected_services = sidebar_filter("Service Type")

selected_customers = sidebar_filter("Customer Type")


filter_mask = (
    df["Region"].isin(selected_regions) &
    df["Channel"].isin(selected_channels) &
    df["Service Type"].isin(selected_services) &
    df["Customer Type"].isin(selected_customers)
)


filtered_df = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date) &
    filter_mask
].copy()


# =========================================================
# 8. PREVIOUS PERIOD COMPARISON
# =========================================================

selected_days = (
    end_date - start_date
).days + 1

previous_end = (
    pd.Timestamp(start_date) -
    pd.Timedelta(days=1)
).date()

previous_start = (
    pd.Timestamp(previous_end) -
    pd.Timedelta(days=selected_days - 1)
).date()


previous_df = df[
    (df["Date"].dt.date >= previous_start) &
    (df["Date"].dt.date <= previous_end) &
    filter_mask
].copy()


current_kpis = calculate_kpis(
    filtered_df
)

previous_kpis = calculate_kpis(
    previous_df
)


# =========================================================
# 9. HEADER
# =========================================================

st.title("Business Intelligence Platform")

st.markdown(
    """
    **Business Intelligence & Rule-Based Recommendation Engine**

    Monitor business performance, understand trends,
    evaluate marketing efficiency, and manage business actions.
    """
)

st.caption(
    f"Current Period: {start_date} → {end_date} | "
    f"Records: {len(filtered_df):,}"
)

st.divider()


# =========================================================
# 10. EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    if show_empty_message(filtered_df):
        st.stop()

    st.subheader("Key Performance Indicators")

    st.caption(
        f"Previous Period: {previous_start} → {previous_end}"
    )

    revenue_growth = calculate_growth(
        current_kpis["Revenue"],
        previous_kpis["Revenue"]
    )

    spend_growth = calculate_growth(
        current_kpis["Ad Spend"],
        previous_kpis["Ad Spend"]
    )

    conversion_growth = calculate_growth(
        current_kpis["Conversions"],
        previous_kpis["Conversions"]
    )

    average_growth = calculate_growth(
        current_kpis["Average Revenue per Transaction"],
        previous_kpis["Average Revenue per Transaction"]
    )

    roas_growth = calculate_growth(
        current_kpis["ROAS"],
        previous_kpis["ROAS"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Revenue",
        format_number(current_kpis["Revenue"]),
        delta=(
            f"{revenue_growth:+.2f}%"
            if revenue_growth is not None
            else "N/A"
        )
    )

    col2.metric(
        "Total Ad Spend",
        format_number(current_kpis["Ad Spend"]),
        delta=(
            f"{spend_growth:+.2f}%"
            if spend_growth is not None
            else "N/A"
        ),
        delta_color="inverse"
    )

    col3.metric(
        "Conversions",
        f"{current_kpis['Conversions']:,.0f}",
        delta=(
            f"{conversion_growth:+.2f}%"
            if conversion_growth is not None
            else "N/A"
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall ROAS",
        (
            f"{current_kpis['ROAS']:.2f}x"
            if pd.notna(current_kpis["ROAS"])
            else "N/A"
        ),
        delta=(
            f"{roas_growth:+.2f}%"
            if roas_growth is not None
            else "N/A"
        )
    )

    col2.metric(
        "Average Revenue / Transaction",
        format_number(
            current_kpis["Average Revenue per Transaction"]
        ),
        delta=(
            f"{average_growth:+.2f}%"
            if average_growth is not None
            else "N/A"
        )
    )

    col3.metric(
        "Total Transactions",
        f"{current_kpis['Transactions']:,}"
    )

    st.divider()

    st.subheader("Business Performance Trends")

    monthly = monthly_analysis(
        filtered_df
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            title="Monthly Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.line(
            monthly,
            x="Month",
            y="Ad_Spend",
            markers=True,
            title="Monthly Ad Spend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        channel_data = (
            filtered_df.groupby(
                "Channel",
                as_index=False
            )["Revenue"]
            .sum()
            .sort_values(
                "Revenue",
                ascending=False
            )
        )

        fig = px.bar(
            channel_data,
            x="Channel",
            y="Revenue",
            color="Channel",
            title="Revenue by Channel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        customer_data = (
            filtered_df.groupby(
                "Customer Type",
                as_index=False
            )["Revenue"]
            .sum()
        )

        fig = px.pie(
            customer_data,
            names="Customer Type",
            values="Revenue",
            title="Customer Revenue Contribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Recent Business Records")

    st.dataframe(
        filtered_df.sort_values(
            "Date",
            ascending=False
        ).head(10),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 11. REVENUE ANALYSIS
# =========================================================

elif page == "Revenue Analysis":

    st.header("Revenue Analysis")

    if show_empty_message(filtered_df):
        st.stop()

    monthly = monthly_analysis(
        filtered_df
    )

    st.subheader("Monthly Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            monthly,
            x="Month",
            y="Revenue",
            title="Monthly Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            monthly,
            x="Month",
            y="Ad_Spend",
            title="Monthly Advertising Spend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            monthly,
            x="Month",
            y="Conversions",
            markers=True,
            title="Monthly Conversions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.line(
            monthly,
            x="Month",
            y="ROAS",
            markers=True,
            title="Monthly ROAS"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Monthly Growth Analysis")

    display_monthly = monthly.copy()

    display_monthly["Month"] = (
        display_monthly["Month"]
        .dt.strftime("%Y-%m")
    )

    growth_columns = [
        "Revenue Growth %",
        "Ad Spend Growth %",
        "Conversions Growth %",
        "ROAS Growth %"
    ]

    for col in growth_columns:

        display_monthly[col] = (
            display_monthly[col].apply(
                lambda x: (
                    f"{x:+.2f}%"
                    if pd.notna(x)
                    else "N/A"
                )
            )
        )

    display_monthly = display_monthly.rename(
        columns={
            "Month": "Month",
            "Revenue": "Total Revenue",
            "Ad_Spend": "Total Ad Spend",
            "Conversions": "Total Conversions"
        }
    )

    st.dataframe(
        display_monthly,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Monthly Analysis",
        data=monthly.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="monthly_performance.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Revenue by Region and Service")

    col1, col2 = st.columns(2)

    with col1:

        region_data = (
            filtered_df.groupby(
                "Region",
                as_index=False
            )["Revenue"]
            .sum()
        )

        fig = px.bar(
            region_data,
            x="Region",
            y="Revenue",
            title="Revenue by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        service_data = (
            filtered_df.groupby(
                "Service Type",
                as_index=False
            )["Revenue"]
            .sum()
        )

        fig = px.pie(
            service_data,
            names="Service Type",
            values="Revenue",
            title="Revenue Share by Service"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# 12. MARKETING PERFORMANCE
# =========================================================

elif page == "Marketing Performance":

    st.header("Marketing Performance")

    if show_empty_message(filtered_df):
        st.stop()

    channel_performance = (
        filtered_df.groupby("Channel")
        .agg(
            Total_Revenue=("Revenue", "sum"),
            Total_Ad_Spend=("Ad Spend", "sum"),
            Total_Conversions=("Conversions", "sum")
        )
        .reset_index()
    )

    channel_performance["ROAS"] = np.where(
        channel_performance["Total_Ad_Spend"] > 0,
        (
            channel_performance["Total_Revenue"] /
            channel_performance["Total_Ad_Spend"]
        ),
        np.nan
    )

    channel_performance["Revenue per Conversion"] = np.where(
        channel_performance["Total_Conversions"] > 0,
        (
            channel_performance["Total_Revenue"] /
            channel_performance["Total_Conversions"]
        ),
        np.nan
    )

    total_spend = channel_performance[
        "Total_Ad_Spend"
    ].sum()

    total_revenue = channel_performance[
        "Total_Revenue"
    ].sum()

    channel_performance["Revenue Share %"] = (
        channel_performance["Total_Revenue"] /
        total_revenue * 100
        if total_revenue != 0
        else 0
    )

    channel_performance["Spend Share %"] = (
        channel_performance["Total_Ad_Spend"] /
        total_spend * 100
        if total_spend != 0
        else 0
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            channel_performance,
            x="Channel",
            y="ROAS",
            color="Channel",
            title="ROAS by Channel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            channel_performance,
            x="Total_Ad_Spend",
            y="Total_Revenue",
            size="Total_Conversions",
            color="Channel",
            hover_name="Channel",
            title="Ad Spend vs Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Marketing Channel Efficiency")

    st.dataframe(
        channel_performance.sort_values(
            "ROAS",
            ascending=False,
            na_position="last"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Channel Performance",
        data=channel_performance.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="channel_performance.csv",
        mime="text/csv"
    )


# =========================================================
# 13. CUSTOMER ANALYSIS
# =========================================================

elif page == "Customer Analysis":

    st.header("Customer Analysis")

    if show_empty_message(filtered_df):
        st.stop()

    customer_data = (
        filtered_df.groupby("Customer Type")
        .agg(
            Total_Revenue=("Revenue", "sum"),
            Total_Conversions=("Conversions", "sum"),
            Transactions=("Revenue", "count")
        )
        .reset_index()
    )

    total_customer_revenue = (
        customer_data["Total_Revenue"].sum()
    )

    customer_data["Revenue Share %"] = np.where(
        total_customer_revenue > 0,
        (
            customer_data["Total_Revenue"] /
            total_customer_revenue
        ) * 100,
        0
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            customer_data,
            x="Customer Type",
            y="Total_Revenue",
            color="Customer Type",
            title="Revenue by Customer Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            customer_data,
            names="Customer Type",
            values="Total_Revenue",
            title="Customer Revenue Share"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Customer Segment Performance")

    st.dataframe(
        customer_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Revenue Share measures the contribution of each customer "
        "type to total revenue. It is not a customer retention rate."
    )


# =========================================================
# 14. SERVICE & REGIONAL ANALYSIS
# =========================================================

elif page == "Service & Regional Analysis":

    st.header("Service & Regional Analysis")

    if show_empty_message(filtered_df):
        st.stop()

    service_data = (
        filtered_df.groupby("Service Type")
        .agg(
            Revenue=("Revenue", "sum"),
            Ad_Spend=("Ad Spend", "sum"),
            Conversions=("Conversions", "sum")
        )
        .reset_index()
    )

    service_data["ROAS"] = np.where(
        service_data["Ad_Spend"] > 0,
        (
            service_data["Revenue"] /
            service_data["Ad_Spend"]
        ),
        np.nan
    )

    region_data = (
        filtered_df.groupby("Region")
        .agg(
            Revenue=("Revenue", "sum"),
            Ad_Spend=("Ad Spend", "sum"),
            Conversions=("Conversions", "sum")
        )
        .reset_index()
    )

    region_data["ROAS"] = np.where(
        region_data["Ad_Spend"] > 0,
        (
            region_data["Revenue"] /
            region_data["Ad_Spend"]
        ),
        np.nan
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            service_data,
            x="Service Type",
            y="Revenue",
            color="Service Type",
            title="Revenue by Service"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            region_data,
            x="Region",
            y="Revenue",
            color="Region",
            title="Revenue by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Service ROAS")

    fig = px.bar(
        service_data,
        x="Service Type",
        y="ROAS",
        title="ROAS by Service"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Regional Performance")

    st.dataframe(
        region_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Service vs Customer Type")

    cross_data = pd.pivot_table(
        filtered_df,
        values="Revenue",
        index="Service Type",
        columns="Customer Type",
        aggfunc="sum",
        fill_value=0
    )

    fig = px.imshow(
        cross_data,
        text_auto=True,
        aspect="auto",
        title="Revenue Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 15. RECOMMENDATIONS & ACTION TRACKING
# =========================================================

elif page == "Recommendations & Action Tracking":

    st.header("Recommendations & Action Tracking")

    tab1, tab2 = st.tabs(
        [
            "Business Recommendations",
            "Action Tracking"
        ]
    )

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    with tab1:

        st.subheader("Generated Business Recommendations")

        if recommendations.empty:

            st.warning(
                "No recommendations file found or file is empty."
            )

        else:

            display_recommendations = recommendations.copy()

            if "Priority" in display_recommendations.columns:

                priorities = (
                    display_recommendations["Priority"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_priorities = st.multiselect(
                    "Filter by Priority",
                    priorities,
                    default=priorities,
                    key="recommendation_filter"
                )

                display_recommendations = (
                    display_recommendations[
                        display_recommendations["Priority"].isin(
                            selected_priorities
                        )
                    ]
                )

            st.metric(
                "Total Recommendations",
                len(display_recommendations)
            )

            st.dataframe(
                display_recommendations,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Recommendations",
                data=display_recommendations.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="business_recommendations.csv",
                mime="text/csv"
            )

    # -----------------------------------------------------
    # ACTION TRACKING
    # -----------------------------------------------------

    with tab2:

        st.subheader("Action Plan Execution Tracking")

        if tracking_df.empty:

            st.warning(
                "No action plan available. Check action_plan.csv."
            )

        else:

            tracking_view = tracking_df.copy()

            if "Priority" in tracking_view.columns:

                priorities = (
                    tracking_view["Priority"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_priorities = st.multiselect(
                    "Filter by Priority",
                    priorities,
                    default=priorities,
                    key="tracking_priority"
                )

                tracking_view = tracking_view[
                    tracking_view["Priority"].isin(
                        selected_priorities
                    )
                ]

            statuses = [
                "Pending",
                "In Progress",
                "Completed",
                "On Hold"
            ]

            selected_statuses = st.multiselect(
                "Filter by Status",
                statuses,
                default=statuses
            )

            tracking_view = tracking_view[
                tracking_view["Status"].isin(
                    selected_statuses
                )
            ]

            total_actions = len(tracking_view)

            completed_actions = (
                tracking_view["Status"] == "Completed"
            ).sum()

            in_progress_actions = (
                tracking_view["Status"] == "In Progress"
            ).sum()

            pending_actions = (
                tracking_view["Status"] == "Pending"
            ).sum()

            completion_rate = (
                completed_actions / total_actions * 100
                if total_actions > 0
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Actions",
                total_actions
            )

            col2.metric(
                "Completed",
                int(completed_actions)
            )

            col3.metric(
                "In Progress",
                int(in_progress_actions)
            )

            col4.metric(
                "Completion Rate",
                f"{completion_rate:.1f}%"
            )

            st.progress(
                completion_rate / 100
            )

            st.divider()

            st.markdown(
                """
                Edit the task status, actual result, and notes.
                Save your changes to keep them between sessions.
                """
            )

            editable_columns = [
                "Action ID",
                "Priority",
                "Category",
                "Finding",
                "Recommendation",
                "Action Plan",
                "Expected Impact",
                "Responsible Team",
                "Success KPI",
                "Measurement Method",
                "Timeframe",
                "Status",
                "Actual Result",
                "Notes",
                "Updated At"
            ]

            visible_columns = [
                col for col in editable_columns
                if col in tracking_view.columns
            ]

            editor_data = tracking_view[
                visible_columns
            ].copy()

            column_config = {
                "Action ID": st.column_config.TextColumn(
                    "Action ID",
                    disabled=True
                ),

                "Priority": st.column_config.TextColumn(
                    "Priority",
                    disabled=True
                ),

                "Category": st.column_config.TextColumn(
                    "Category",
                    disabled=True
                ),

                "Finding": st.column_config.TextColumn(
                    "Finding",
                    disabled=True,
                    width="large"
                ),

                "Recommendation": st.column_config.TextColumn(
                    "Recommendation",
                    disabled=True,
                    width="large"
                ),

                "Action Plan": st.column_config.TextColumn(
                    "Action Plan",
                    disabled=True,
                    width="large"
                ),

                "Expected Impact": st.column_config.TextColumn(
                    "Expected Impact",
                    disabled=True
                ),

                "Responsible Team": st.column_config.TextColumn(
                    "Responsible Team"
                ),

                "Success KPI": st.column_config.TextColumn(
                    "Success KPI",
                    disabled=True
                ),

                "Measurement Method": st.column_config.TextColumn(
                    "Measurement Method",
                    disabled=True
                ),

                "Timeframe": st.column_config.TextColumn(
                    "Timeframe",
                    disabled=True
                ),

                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=statuses,
                    required=True
                ),

                "Actual Result": st.column_config.TextColumn(
                    "Actual Result",
                    width="large"
                ),

                "Notes": st.column_config.TextColumn(
                    "Notes",
                    width="large"
                ),

                "Updated At": st.column_config.TextColumn(
                    "Updated At",
                    disabled=True
                )
            }

            edited_data = st.data_editor(
                editor_data,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="action_tracking_editor"
            )

            if st.button(
                "💾 Save Action Tracking",
                type="primary",
                use_container_width=True
            ):

                updated_tracking = tracking_df.copy()

                edited_data = edited_data.copy()

                if "Action ID" in edited_data.columns:

                    edited_ids = edited_data["Action ID"].astype(str)

                    for _, edited_row in edited_data.iterrows():

                        action_id = str(
                            edited_row["Action ID"]
                        )

                        mask = (
                            updated_tracking["Action ID"].astype(str)
                            == action_id
                        )

                        for col in [
                            "Status",
                            "Actual Result",
                            "Notes",
                            "Responsible Team"
                        ]:

                            if (
                                col in edited_data.columns and
                                col in updated_tracking.columns
                            ):

                                updated_tracking.loc[
                                    mask, col
                                ] = edited_row[col]

                        updated_tracking.loc[
                            mask, "Updated At"
                        ] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                save_action_tracking(
                    updated_tracking
                )

                st.success(
                    "Action tracking saved successfully."
                )

                st.rerun()

            st.divider()

            st.subheader("Export Action Tracking")

            st.download_button(
                "Download Full Action Tracking CSV",
                data=tracking_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="action_tracking.csv",
                mime="text/csv"
            )


# =========================================================
# 16. DATA QUALITY
# =========================================================

elif page == "Data Quality":

    st.header("Data Quality & Validation")

    st.markdown(
        "Review the quality of the source dataset and identify "
        "potential data issues."
    )

    total_rows = len(df)

    duplicate_rows = int(
        df.duplicated().sum()
    )

    negative_revenue = int(
        (df["Revenue"] < 0).sum()
    )

    negative_spend = int(
        (df["Ad Spend"] < 0).sum()
    )

    zero_conversions = int(
        (df["Conversions"] == 0).sum()
    )

    revenue_without_conversions = int(
        (
            (df["Revenue"] > 0) &
            (df["Conversions"] == 0)
        ).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Valid Records Loaded",
        f"{total_rows:,}"
    )

    col2.metric(
        "Duplicate Rows",
        duplicate_rows
    )

    col3.metric(
        "Zero Conversion Records",
        zero_conversions
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Negative Revenue",
        negative_revenue
    )

    col2.metric(
        "Negative Ad Spend",
        negative_spend
    )

    col3.metric(
        "Revenue Without Conversions",
        revenue_without_conversions
    )

    st.divider()

    st.subheader("Missing Values")

    missing_data = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing_data.columns = [
        "Column",
        "Missing Values"
    ]

    missing_data["Missing %"] = (
        missing_data["Missing Values"] /
        len(df) * 100
    )

    st.dataframe(
        missing_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Data Overview")

    st.write(
        f"Dataset start date: {df['Date'].min().date()}"
    )

    st.write(
        f"Dataset end date: {df['Date'].max().date()}"
    )

    st.write(
        f"Number of columns: {len(df.columns)}"
    )

    st.write(
        f"Number of records: {len(df):,}"
    )

    st.info(
        "These checks identify potential issues. Negative values "
        "or zero conversions are not automatically errors; "
        "review their business meaning before changing the data."
    )

    st.subheader("Source Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 17. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Business Intelligence Platform"
)

st.sidebar.caption(
    "Python | Pandas | Plotly | Streamlit"
)

st.sidebar.caption(
    "Analytics • Recommendations • Action Tracking"
)
