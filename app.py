import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import t, norm


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Petrol Consumption Analytics",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

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

    .dashboard-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    else:
        possible_paths = [
            "DATASETS/Petrol_Consumption.csv",
            "../DATASETS/Petrol_Consumption.csv",
            "Petrol_Consumption.csv"
        ]

        data = None

        for path in possible_paths:
            try:
                data = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue

        if data is None:
            return None

    # Rename the column used in the notebook
    if "Population_Driver_licence(%)" in data.columns:
        data = data.rename(
            columns={
                "Population_Driver_licence(%)":
                "Population_Driver_licence"
            }
        )

    return data


# ============================================================
# OUTLIER REMOVAL
# ============================================================

def remove_iqr_outliers(data, column):

    df = data.copy()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    mask = (
        (df[column] >= lower_limit)
        &
        (df[column] <= upper_limit)
    )

    cleaned = df[mask].copy()

    removed = len(df) - len(cleaned)

    return cleaned, removed, q1, q3, iqr, lower_limit, upper_limit


def sequential_outlier_removal(data, columns):

    df = data.copy()
    report = []

    for column in columns:

        before = len(df)

        cleaned, removed, q1, q3, iqr, lower, upper = (
            remove_iqr_outliers(df, column)
        )

        after = len(cleaned)

        report.append(
            {
                "Column": column,
                "Rows Before": before,
                "Rows Removed": removed,
                "Rows After": after,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper
            }
        )

        df = cleaned

    return df, pd.DataFrame(report)


# ============================================================
# CORRELATION
# ============================================================

def correlation_strength(value):

    absolute = abs(value)

    if absolute >= 0.8:
        return "Very Strong"

    elif absolute >= 0.6:
        return "Strong"

    elif absolute >= 0.4:
        return "Moderate"

    elif absolute >= 0.2:
        return "Weak"

    return "Very Weak"


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def calculate_confidence_interval(values, confidence):

    values = np.asarray(values)

    n = len(values)

    mean = np.mean(values)

    # Reproduces the notebook's np.std() approach
    std = np.std(values)

    degrees_of_freedom = n - 1

    alpha = 1 - confidence

    t_score = t.ppf(
        1 - alpha / 2,
        degrees_of_freedom
    )

    margin_error = (
        t_score * std / np.sqrt(n)
    )

    lower = mean - margin_error
    upper = mean + margin_error

    return {
        "mean": mean,
        "std": std,
        "n": n,
        "dof": degrees_of_freedom,
        "t_score": t_score,
        "margin_error": margin_error,
        "lower": lower,
        "upper": upper
    }


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">⛽ Petrol Consumption Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-subtitle">
    Interactive statistical analysis dashboard based on the
    Petrol Consumption analysis notebook.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("### Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Petrol Consumption CSV",
    type=["csv"]
)

df_original = load_data(uploaded_file)

if df_original is None:

    st.error(
        """
        Dataset not found.

        Please place `Petrol_Consumption.csv` inside the
        `DATASETS` folder or upload the CSV using the sidebar.
        """
    )

    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "Petrol_tax",
    "Average_income",
    "Paved_Highways",
    "Population_Driver_licence",
    "Petrol_Consumption"
]

missing_columns = [
    col for col in required_columns
    if col not in df_original.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# OUTLIER SETTINGS
# ============================================================

st.sidebar.markdown("### Data Cleaning")

apply_outliers = st.sidebar.checkbox(
    "Remove IQR outliers",
    value=False
)

outlier_columns = st.sidebar.multiselect(
    "Columns for IQR filtering",
    required_columns,
    default=[
        "Petrol_tax",
        "Paved_Highways",
        "Population_Driver_licence",
        "Petrol_Consumption"
    ],
    disabled=not apply_outliers
)


if apply_outliers and outlier_columns:

    df, outlier_report = sequential_outlier_removal(
        df_original,
        outlier_columns
    )

else:

    df = df_original.copy()

    outlier_report = pd.DataFrame()


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Total Records",
        len(df)
    )

with c2:
    st.metric(
        "Variables",
        len(df.columns)
    )

with c3:
    st.metric(
        "Avg. Petrol Consumption",
        f"{df['Petrol_Consumption'].mean():.2f}"
    )

with c4:
    st.metric(
        "Median Consumption",
        f"{df['Petrol_Consumption'].median():.2f}"
    )

with c5:
    st.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )


st.divider()


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📊 Overview",
        "📈 Distribution",
        "📦 Outliers",
        "🔗 Correlation",
        "🔬 Relationships",
        "📐 Confidence Intervals",
        "🧪 Hypothesis Testing"
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab1:

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

    st.subheader("Data Types")

    dtype_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique())
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.subheader("Mean / Median / Mode")

    statistics_df = pd.DataFrame(
        {
            "Mean": df.mean(numeric_only=True),
            "Median": df.median(numeric_only=True),
            "Mode": [
                (
                    df[col].mode().iloc[0]
                    if not df[col].mode().empty
                    else np.nan
                )
                for col in df.select_dtypes(
                    include=np.number
                ).columns
            ]
        },
        index=df.select_dtypes(
            include=np.number
        ).columns
    )

    st.dataframe(
        statistics_df,
        use_container_width=True
    )


# ============================================================
# TAB 2 - DISTRIBUTION
# ============================================================

with tab2:

    st.subheader("Distribution Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select variable",
        numeric_columns,
        key="distribution_column"
    )

    chart_type = st.radio(
        "Chart type",
        [
            "Histogram",
            "KDE / Density"
        ],
        horizontal=True
    )

    if chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=selected_column,
            nbins=15,
            marginal="box",
            title=f"Distribution of {selected_column}"
        )

    else:

        values = df[selected_column].dropna()

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=values,
                histnorm="probability density",
                opacity=0.45,
                name="Histogram"
            )
        )

        # Gaussian KDE approximation using scipy-free
        # Plotly-friendly rolling density calculation
        x = np.linspace(
            values.min(),
            values.max(),
            200
        )

        bandwidth = (
            1.06
            * values.std()
            * len(values) ** (-1 / 5)
        )

        if bandwidth <= 0:
            bandwidth = 1

        density = np.zeros(len(x))

        for value in values:
            density += (
                np.exp(
                    -0.5
                    * ((x - value) / bandwidth) ** 2
                )
                /
                (
                    bandwidth
                    * np.sqrt(2 * np.pi)
                )
            )

        density /= len(values)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=density,
                mode="lines",
                name="KDE"
            )
        )

        fig.update_layout(
            title=f"KDE of {selected_column}",
            xaxis_title=selected_column,
            yaxis_title="Density"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Distribution statistics
    st.subheader("Distribution Statistics")

    col1, col2, col3, col4 = st.columns(4)

    values = df[selected_column].dropna()

    with col1:
        st.metric(
            "Mean",
            f"{values.mean():.3f}"
        )

    with col2:
        st.metric(
            "Median",
            f"{values.median():.3f}"
        )

    with col3:
        st.metric(
            "Std. Deviation",
            f"{values.std():.3f}"
        )

    with col4:
        st.metric(
            "Variance",
            f"{values.var():.3f}"
        )


# ============================================================
# TAB 3 - OUTLIERS
# ============================================================

with tab3:

    st.subheader("IQR Outlier Analysis")

    selected_outlier_column = st.selectbox(
        "Select variable",
        required_columns,
        key="outlier_column"
    )

    values = df[selected_outlier_column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound)
        |
        (values > upper_bound)
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Q1",
            f"{q1:.3f}"
        )

    with c2:
        st.metric(
            "Q3",
            f"{q3:.3f}"
        )

    with c3:
        st.metric(
            "IQR",
            f"{iqr:.3f}"
        )

    with c4:
        st.metric(
            "Outliers",
            len(outliers)
        )

    fig = px.box(
        df,
        y=selected_outlier_column,
        points="outliers",
        title=f"Boxplot - {selected_outlier_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("IQR Boundaries")

    boundary_df = pd.DataFrame(
        {
            "Metric": [
                "Q1",
                "Q3",
                "IQR",
                "Lower Bound",
                "Upper Bound"
            ],
            "Value": [
                q1,
                q3,
                iqr,
                lower_bound,
                upper_bound
            ]
        }
    )

    st.dataframe(
        boundary_df,
        use_container_width=True
    )

    if apply_outliers and not outlier_report.empty:

        st.subheader(
            "Sequential Outlier Removal Report"
        )

        st.dataframe(
            outlier_report,
            use_container_width=True
        )


# ============================================================
# TAB 4 - CORRELATION
# ============================================================

with tab4:

    st.subheader("Correlation Analysis")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Correlation Table")

    st.dataframe(
        corr.style.format("{:.3f}"),
        use_container_width=True
    )

    st.subheader("Strongest Relationships")

    correlation_pairs = []

    columns = corr.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = corr.iloc[i, j]

            correlation_pairs.append(
                {
                    "Variable 1": columns[i],
                    "Variable 2": columns[j],
                    "Correlation": value,
                    "Strength": correlation_strength(value)
                }
            )

    pairs_df = pd.DataFrame(
        correlation_pairs
    ).sort_values(
        "Correlation",
        key=lambda x: abs(x),
        ascending=False
    )

    st.dataframe(
        pairs_df,
        use_container_width=True
    )


# ============================================================
# TAB 5 - RELATIONSHIPS
# ============================================================

with tab5:

    st.subheader("Interactive Scatter Plot")

    x_column = st.selectbox(
        "X-axis",
        numeric_columns,
        index=0,
        key="scatter_x"
    )

    y_column = st.selectbox(
        "Y-axis",
        numeric_columns,
        index=min(
            1,
            len(numeric_columns) - 1
        ),
        key="scatter_y"
    )

    show_trendline = st.checkbox(
        "Show regression trendline",
        value=True
    )

    if show_trendline:

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            trendline="ols",
            title=f"{x_column} vs {y_column}"
        )

    else:

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}"
        )

    fig.update_traces(
        marker=dict(
            size=9,
            opacity=0.75
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    correlation = df[
        [x_column, y_column]
    ].corr().iloc[0, 1]

    st.metric(
        f"Correlation: {x_column} vs {y_column}",
        f"{correlation:.4f}"
    )

    st.subheader("Pairwise Analysis")

    pair_columns = st.multiselect(
        "Select variables for pairwise analysis",
        numeric_columns,
        default=numeric_columns
    )

    if len(pair_columns) >= 2:

        pair_df = df[pair_columns]

        # Plotly parallel coordinates gives an interactive
        # alternative to the notebook's static pairplot.
        normalized = pair_df.copy()

        for column in pair_columns:

            min_value = normalized[column].min()
            max_value = normalized[column].max()

            if max_value != min_value:

                normalized[column] = (
                    normalized[column] - min_value
                ) / (
                    max_value - min_value
                )

        fig = px.scatter_matrix(
            pair_df,
            dimensions=pair_columns,
            title="Interactive Pair Plot"
        )

        fig.update_traces(
            diagonal_visible=True,
            showupperhalf=True,
            showlowerhalf=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 6 - CONFIDENCE INTERVALS
# ============================================================

with tab6:

    st.subheader(
        "Confidence Interval Calculator"
    )

    ci_column = st.selectbox(
        "Select variable",
        numeric_columns,
        key="ci_column"
    )

    confidence = st.select_slider(
        "Confidence level",
        options=[
            0.90,
            0.95,
            0.99,
            0.995
        ],
        value=0.95,
        format_func=lambda x: f"{x * 100:.1f}%"
    )

    ci_result = calculate_confidence_interval(
        df[ci_column].dropna(),
        confidence
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Sample Mean",
            f"{ci_result['mean']:.4f}"
        )

    with c2:
        st.metric(
            "Std. Deviation",
            f"{ci_result['std']:.4f}"
        )

    with c3:
        st.metric(
            "Sample Size",
            ci_result["n"]
        )

    st.markdown("### Confidence Interval")

    lower = ci_result["lower"]
    upper = ci_result["upper"]

    st.success(
        f"""
        **{confidence * 100:.1f}% Confidence Interval**

        **{lower:.4f} ≤ μ ≤ {upper:.4f}**
        """
    )

    ci_table = pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Standard Deviation",
                "Sample Size",
                "Degrees of Freedom",
                "t Score",
                "Margin of Error",
                "Lower Bound",
                "Upper Bound"
            ],
            "Value": [
                ci_result["mean"],
                ci_result["std"],
                ci_result["n"],
                ci_result["dof"],
                ci_result["t_score"],
                ci_result["margin_error"],
                ci_result["lower"],
                ci_result["upper"]
            ]
        }
    )

    st.dataframe(
        ci_table,
        use_container_width=True
    )


# ============================================================
# TAB 7 - HYPOTHESIS TESTING
# ============================================================

with tab7:

    st.subheader("Hypothesis Testing")

    test_type = st.radio(
        "Select hypothesis test",
        [
            "Notebook Example 1",
            "Notebook Example 2",
            "Custom One-Sample Test"
        ],
        horizontal=True
    )


    # --------------------------------------------------------
    # NOTEBOOK EXAMPLE 1
    # --------------------------------------------------------

    if test_type == "Notebook Example 1":

        st.markdown(
            "### Hypothesis Test Example 1"
        )

        theory_mean = 1.8 + 0.02 * 450
        sample_mean = 10.8
        sample_std = 25
        sample_size = 35
        alpha = 0.05

        standard_error = (
            sample_std / np.sqrt(sample_size)
        )

        test_statistic = (
            sample_mean - theory_mean
        ) / standard_error

        critical_value = norm.ppf(
            1 - alpha
        )

        if test_statistic > critical_value:
            decision = "Reject H0 / Support H1"
        else:
            decision = "Fail to Reject H0"

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Theoretical Mean",
                f"{theory_mean:.4f}"
            )

        with c2:
            st.metric(
                "Test Statistic",
                f"{test_statistic:.4f}"
            )

        with c3:
            st.metric(
                "Critical Value",
                f"{critical_value:.4f}"
            )

        if test_statistic > critical_value:
            st.error(decision)
        else:
            st.success(decision)


    # --------------------------------------------------------
    # NOTEBOOK EXAMPLE 2
    # --------------------------------------------------------

    elif test_type == "Notebook Example 2":

        st.markdown(
            "### Hypothesis Test Example 2"
        )

        theory_mean = 10 + 0.08 * 75
        sample_mean = 16.2
        sample_std = 6
        sample_size = 30
        alpha = 0.05

        standard_error = (
            sample_std / np.sqrt(sample_size)
        )

        test_statistic = (
            sample_mean - theory_mean
        ) / standard_error

        critical_value = norm.ppf(
            1 - alpha
        )

        if test_statistic > critical_value:
            decision = "Reject H0 / Support H1"
        else:
            decision = "Fail to Reject H0"

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Theoretical Mean",
                f"{theory_mean:.4f}"
            )

        with c2:
            st.metric(
                "Test Statistic",
                f"{test_statistic:.4f}"
            )

        with c3:
            st.metric(
                "Critical Value",
                f"{critical_value:.4f}"
            )

        if test_statistic > critical_value:
            st.error(decision)
        else:
            st.success(decision)


    # --------------------------------------------------------
    # CUSTOM TEST
    # --------------------------------------------------------

    else:

        st.markdown(
            "### Custom One-Sample Hypothesis Test"
        )

        st.info(
            """
            H0: μ = theoretical mean

            H1: μ > theoretical mean

            The calculation follows the same
            one-sided normal-critical-value approach
            used in the notebook examples.
            """
        )

        theoretical_mean = st.number_input(
            "Theoretical Mean",
            value=10.0
        )

        sample_mean = st.number_input(
            "Sample Mean",
            value=12.0
        )

        sample_std = st.number_input(
            "Sample Standard Deviation",
            min_value=0.0001,
            value=2.0
        )

        sample_size = st.number_input(
            "Sample Size",
            min_value=2,
            value=30,
            step=1
        )

        alpha = st.selectbox(
            "Significance Level",
            [0.01, 0.05, 0.10],
            index=1
        )

        if st.button(
            "Run Hypothesis Test",
            type="primary"
        ):

            standard_error = (
                sample_std
                / np.sqrt(sample_size)
            )

            test_statistic = (
                sample_mean - theoretical_mean
            ) / standard_error

            critical_value = norm.ppf(
                1 - alpha
            )

            p_value = 1 - norm.cdf(
                test_statistic
            )

            st.markdown("### Results")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Test Statistic",
                    f"{test_statistic:.4f}"
                )

            with c2:
                st.metric(
                    "Critical Value",
                    f"{critical_value:.4f}"
                )

            with c3:
                st.metric(
                    "P-value",
                    f"{p_value:.6f}"
                )

            with c4:
                st.metric(
                    "Alpha",
                    f"{alpha:.2f}"
                )

            if test_statistic > critical_value:

                st.error(
                    "Reject H0. There is evidence "
                    "supporting H1."
                )

            else:

                st.success(
                    "Fail to reject H0. There is "
                    "not enough evidence supporting H1."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Petrol Consumption Statistical Analysis Dashboard • "
    "Built with Streamlit, Pandas, SciPy and Plotly"
)
