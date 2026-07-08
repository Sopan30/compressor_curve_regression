import streamlit as st
import pandas as pd
import numpy as np

from io import BytesIO

import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from scipy.interpolate import CubicSpline


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Compressor Curve Regression",
    layout="wide"
)

st.title("Compressor Curve Regression Tool")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

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
    10,
    25,
    1
)

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"]
)


# ---------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------

def detect_columns(df):

    speed = None
    flow = None
    head = None
    eff = None
    power = None

    for col in df.columns:

        c = str(col).lower()

        if speed is None and "speed" in c:
            speed = col

        if flow is None and (
            "flow" in c
            or "volumeflow" in c
            or "inlet1_volumeflowrate" in c
        ):
            flow = col

        if head is None and "head" in c:
            head = col

        if eff is None and (
            "eff" in c
            or "efficiency" in c
        ):
            eff = col

        if power is None and (
            "power" in c
            or "bhp" in c
            or "kw" in c
        ):
            power = col

    return {
        "speed": speed,
        "flow": flow,
        "head": head,
        "eff": eff,
        "power": power
    }


# ---------------------------------------------------
# REGRESSION
# ---------------------------------------------------

def polynomial_fit(x, y, degree, points):

    poly = PolynomialFeatures(degree)

    X = poly.fit_transform(
        x.reshape(-1, 1)
    )

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    x_new = np.linspace(
        x.min(),
        x.max(),
        points
    )

    y_new = model.predict(
        poly.transform(
            x_new.reshape(-1, 1)
        )
    )

    return x_new, y_new, r2


def spline_fit(x, y, points):

    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]

    spline = CubicSpline(x, y)

    y_pred = spline(x)

    r2 = r2_score(y, y_pred)

    x_new = np.linspace(
        x.min(),
        x.max(),
        points
    )

    y_new = spline(x_new)

    return x_new, y_new, r2


def run_method(method, x, y, points):

    if method == "Linear":
        return polynomial_fit(x, y, 1, points)

    elif method == "Quadratic":
        return polynomial_fit(x, y, 2, points)

    elif method == "Cubic":
        return polynomial_fit(x, y, 3, points)

    elif method == "4th Order":
        return polynomial_fit(x, y, 4, points)

    elif method == "5th Order":
        return polynomial_fit(x, y, 5, points)

    elif method == "Spline":
        return spline_fit(x, y, points)


def auto_best_fit(x, y, points):

    methods = [
        "Linear",
        "Quadratic",
        "Cubic",
        "4th Order",
        "5th Order",
        "Spline"
    ]

    best_r2 = -999

    best_method = None

    best_result = None

    for method in methods:

        try:

            result = run_method(
                method,
                x,
                y,
                points
            )

            if result[2] > best_r2:

                best_r2 = result[2]

                best_method = method

                best_result = result

        except:
            pass

    return best_method, best_result


# ---------------------------------------------------
# PROCESS WORKBOOK
# ---------------------------------------------------

if uploaded_file:

    workbook = pd.read_excel(
        uploaded_file,
        sheet_name=None
    )

    overview_rows = []

    r2_rows = []

    output_buffer = BytesIO()

    with pd.ExcelWriter(
        output_buffer,
        engine="xlsxwriter"
    ) as writer:

        for stage_name, df in workbook.items():

            st.header(stage_name)

            cols = detect_columns(df)

            if cols["speed"] is None:
                continue

            if cols["flow"] is None:
                continue

            speed_col = cols["speed"]
            flow_col = cols["flow"]

            parameters = []

            if cols["head"]:
                parameters.append(cols["head"])

            if cols["eff"]:
                parameters.append(cols["eff"])

            if cols["power"]:
                parameters.append(cols["power"])

            if len(parameters) == 0:
                continue

            tabs = st.tabs(parameters)

            merged_output = None

            for t, parameter in zip(tabs, parameters):

                with t:

                    work = df[
                        [
                            speed_col,
                            flow_col,
                            parameter
                        ]
                    ].dropna()

                    work.columns = [
                        "Speed",
                        "Flow",
                        "Value"
                    ]

                    fig = go.Figure()

                    output_rows = []

                    speeds = sorted(
                        work["Speed"].unique()
                    )

                    for spd in speeds:

                        spd_df = work[
                            work["Speed"] == spd
                        ]

                        x = spd_df["Flow"].values
                        y = spd_df["Value"].values

                        if len(x) < 4:
                            continue

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

                        r2_rows.append(
                            [
                                stage_name,
                                spd,
                                parameter,
                                best_method,
                                round(r2, 6)
                            ]
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                name=f"{spd} Original"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=xfit,
                                y=yfit,
                                mode="lines",
                                name=f"{spd} Fit"
                            )
                        )

                        temp = pd.DataFrame({
                            "Speed": spd,
                            "Flow": xfit,
                            parameter: yfit
                        })

                        output_rows.append(temp)

                    fig.update_layout(
                        title=f"{stage_name} - {parameter}",
                        height=650
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    stage_param_df = pd.concat(
                        output_rows,
                        ignore_index=True
                    )

                    if merged_output is None:

                        merged_output = stage_param_df

                    else:

                        merged_output = pd.merge(
                            merged_output,
                            stage_param_df,
                            on=["Speed", "Flow"],
                            how="outer"
                        )

            if merged_output is not None:

                merged_output.to_excel(
                    writer,
                    sheet_name=f"{stage_name}"[:31],
                    index=False
                )

            overview_rows.append(
                [
                    stage_name,
                    ",".join(
                        [str(i) for i in parameters]
                    ),
                    len(parameters)
                ]
            )

        overview_df = pd.DataFrame(
            overview_rows,
            columns=[
                "Stage",
                "Parameters",
                "Count"
            ]
        )

        r2_df = pd.DataFrame(
            r2_rows,
            columns=[
                "Stage",
                "Speed",
                "Parameter",
                "Method",
                "R2"
            ]
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

    st.success("Processing Completed")

    st.subheader("R² Summary")

    st.dataframe(r2_df)

    st.download_button(
        label="Download Regression Workbook",
        data=output_buffer.getvalue(),
        file_name="Regression_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
