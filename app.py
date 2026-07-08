import streamlit as st
import pandas as pd
import numpy as np

from io import BytesIO

import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy.interpolate import CubicSpline

# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="Compressor Curve Regression",
    layout="wide"
)

st.title("Compressor Curve Regression Tool")

# ==========================================================
# SIDEBAR
# ==========================================================

fit_method = st.sidebar.selectbox(
    "Regression Method",
    [
        "Auto Best Fit",
        "Linear",
        "Quadratic",
        "Cubic",
        "4th Order",
        "5th Order",
        "Spline"
    ]
)

generated_points = st.sidebar.slider(
    "Generated Points",
    min_value=10,
    max_value=200,
    value=15
)

uploaded_file = st.file_uploader(
    "Upload Workbook",
    type=["xlsx"]
)

# ==========================================================
# HELPERS
# ==========================================================

def clean_parameter_name(name):

    name = str(name).strip()
    low = name.lower()

    if "head" in low:
        return "Head"

    if "eff" in low:
        return "Efficiency"

    if "power" in low:
        return "Power"

    return name


def detect_triplet_blocks(df):

    blocks = []

    cols = [str(c).strip() for c in df.columns]

    for i in range(len(cols) - 2):

        c1 = cols[i].lower()
        c2 = cols[i + 1].lower()
        c3 = cols[i + 2]

        if (
            ("speed" in c1)
            and
            (
                "flow" in c2
                or "volumeflow" in c2
            )
        ):

            blocks.append(
                {
                    "speed_col": df.columns[i],
                    "flow_col": df.columns[i + 1],
                    "value_col": df.columns[i + 2],
                    "parameter": clean_parameter_name(c3)
                }
            )

    return blocks


# ==========================================================
# REGRESSION METHODS
# ==========================================================

def polynomial_fit(x, y, degree, npoints):

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(
        x.reshape(-1, 1)
    )

    model = LinearRegression()

    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)

    x_new = np.linspace(
        x.min(),
        x.max(),
        npoints
    )

    y_new = model.predict(
        poly.transform(
            x_new.reshape(-1, 1)
        )
    )

    return x_new, y_new, r2


def spline_fit(x, y, npoints):

    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]

    spline = CubicSpline(x, y)

    y_pred = spline(x)

    r2 = r2_score(y, y_pred)

    x_new = np.linspace(
        x.min(),
        x.max(),
        npoints
    )

    y_new = spline(x_new)

    return x_new, y_new, r2


def run_method(method, x, y, npoints):

    if method == "Linear":
        return polynomial_fit(x, y, 1, npoints)

    if method == "Quadratic":
        return polynomial_fit(x, y, 2, npoints)

    if method == "Cubic":
        return polynomial_fit(x, y, 3, npoints)

    if method == "4th Order":
        return polynomial_fit(x, y, 4, npoints)

    if method == "5th Order":
        return polynomial_fit(x, y, 5, npoints)

    if method == "Spline":
        return spline_fit(x, y, npoints)

    raise ValueError("Unknown Method")


def auto_best_fit(x, y, npoints):

    methods = [
        "Linear",
        "Quadratic",
        "Cubic",
        "4th Order",
        "5th Order",
        "Spline"
    ]

    best_method = None
    best_result = None
    best_r2 = -999

    for method in methods:

        try:

            result = run_method(
                method,
                x,
                y,
                npoints
            )

            if result[2] > best_r2:

                best_r2 = result[2]
                best_method = method
                best_result = result

        except Exception:
            pass

    return best_method, best_result


# ==========================================================
# MAIN
# ==========================================================

if uploaded_file:

    workbook = pd.read_excel(
        uploaded_file,
        sheet_name=None
    )

    summary_rows = []

    overview_rows = []

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter"
    ) as writer:

        for stage_name, df in workbook.items():

            st.header(stage_name)

            blocks = detect_triplet_blocks(df)

            if len(blocks) == 0:

                st.warning(
                    f"No Speed-Flow-Parameter blocks found in {stage_name}"
                )

                continue

            overview_rows.append(
                {
                    "Stage": stage_name,
                    "Blocks Found": len(blocks),
                    "Parameters":
                    ", ".join(
                        [b["parameter"] for b in blocks]
                    )
                }
            )

            tabs = st.tabs(
                [b["parameter"] for b in blocks]
            )

            for block, tab in zip(blocks, tabs):

                with tab:

                    parameter = block["parameter"]

                    temp = df[
                        [
                            block["speed_col"],
                            block["flow_col"],
                            block["value_col"]
                        ]
                    ].copy()

                    temp.columns = [
                        "Speed",
                        "Flow",
                        "Value"
                    ]

                    temp["Speed"] = pd.to_numeric(
                        temp["Speed"],
                        errors="coerce"
                    )

                    temp["Flow"] = pd.to_numeric(
                        temp["Flow"],
                        errors="coerce"
                    )

                    temp["Value"] = pd.to_numeric(
                        temp["Value"],
                        errors="coerce"
                    )

                    temp = temp.dropna()

                    fig = go.Figure()

                    export_rows = []

                    speeds = sorted(
                        temp["Speed"].unique()
                    )

                    for speed in speeds:

                        sdf = temp[
                            temp["Speed"] == speed
                        ].copy()

                        if len(sdf) < 4:
                            continue

                        x = sdf["Flow"].values
                        y = sdf["Value"].values

                        if fit_method == "Auto Best Fit":

                            best_method, result = (
                                auto_best_fit(
                                    x,
                                    y,
                                    generated_points
                                )
                            )

                            xfit, yfit, r2 = result

                        else:

                            xfit, yfit, r2 = (
                                run_method(
                                    fit_method,
                                    x,
                                    y,
                                    generated_points
                                )
                            )

                            best_method = fit_method

                        summary_rows.append(
                            {
                                "Stage": stage_name,
                                "Speed": speed,
                                "Parameter": parameter,
                                "Method": best_method,
                                "R2": round(r2, 6)
                            }
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                name=f"{speed} Original"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=xfit,
                                y=yfit,
                                mode="lines",
                                name=f"{speed} Fit"
                            )
                        )

                        export_rows.append(
                            pd.DataFrame(
                                {
                                    "Speed": speed,
                                    "Flow": xfit,
                                    parameter: yfit
                                }
                            )
                        )

                    fig.update_layout(
                        title=f"{stage_name} - {parameter}",
                        height=650,
                        xaxis_title="Flow",
                        yaxis_title=parameter
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    if len(export_rows):

                        export_df = pd.concat(
                            export_rows,
                            ignore_index=True
                        )

                        sheet_name_export = (
                            f"{stage_name}_{parameter}"
                        )[:31]

                        export_df.to_excel(
                            writer,
                            sheet_name=sheet_name_export,
                            index=False
                        )

        overview_df = pd.DataFrame(
            overview_rows
        )

        r2_df = pd.DataFrame(
            summary_rows
        )

        overview_df.to_excel(
            writer,
            sheet_name="Workbook_Overview",
            index=False
        )

        r2_df.to_excel(
            writer,
            sheet_name="Summary_R2",
            index=False
        )

    st.success("Regression Completed")

    if len(summary_rows):

        st.subheader("R² Summary")

        st.dataframe(
            r2_df,
            use_container_width=True
        )

    st.download_button(
        label="Download Regression Workbook",
        data=output.getvalue(),
        file_name="Regression_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
