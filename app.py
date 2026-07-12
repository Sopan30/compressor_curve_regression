"""Streamlit entry point for compressor curve regression."""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from compressor_curve_regression.constants import G, R_UNIVERSAL
from compressor_curve_regression.parser import SheetProcessor
from compressor_curve_regression.regression import RegressionModelBuilder
from compressor_curve_regression.units import UnitConverter


class CompressorCurveApp:
    def __init__(self):
        self.converter = UnitConverter()
        self.parser = SheetProcessor(self.converter)
        self.regression = RegressionModelBuilder()

    def compute_missing_parameter(self, available: dict[str, np.ndarray], mass_flow_kg_s: np.ndarray):
        have = set(available.keys())
        needed = {'Head', 'Efficiency', 'Power'} - have
        if len(needed) != 1:
            return None, None
        missing = needed.pop()

        if missing == 'Power':
            eff = available['Efficiency'] / 100.0
            power_kw = mass_flow_kg_s * G * available['Head'] / eff / 1000.0
            return 'Power', power_kw

        if missing == 'Head':
            eff = available['Efficiency'] / 100.0
            head_m = available['Power'] * 1000.0 * eff / (mass_flow_kg_s * G)
            return 'Head', head_m

        if missing == 'Efficiency':
            eff_pct = (mass_flow_kg_s * G * available['Head']) / (available['Power'] * 1000.0) * 100.0
            return 'Efficiency', eff_pct

        return None, None

    def run(self):
        st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
        st.title('Compressor Curve Regression Tool')

        method = st.sidebar.selectbox('Regression Method', ['Auto Best Fit', 'Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline'])
        points = st.sidebar.slider('Generated Points', 15, 50, 15)
        file = st.file_uploader('Upload Workbook', type=['xlsx'])

        if not file:
            return

        xls = pd.ExcelFile(file)
        output = BytesIO()
        r2_rows = []
        overview = []
        property_rows = []
        fatal_error = None

        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for stage in xls.sheet_names:
                    st.header(stage)
                    try:
                        raw = pd.read_excel(file, sheet_name=stage, header=None)
                        blocks = self.parser.detect_triplet_blocks(raw)
                        prop_block = self.parser.detect_property_block(raw)
                        prop_df = self.parser.extract_property_block(raw, prop_block)
                        gas_props = None

                        if not prop_df.empty:
                            st.subheader('Operating Conditions')
                            st.dataframe(prop_df, use_container_width=True)
                            for _, row in prop_df.iterrows():
                                property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                            gas_props = self.parser.gas_properties_from_df(prop_df)

                            if gas_props is not None:
                                t_k = self.converter.c_to_k(gas_props['temperature_c'])
                                p_pa = self.converter.kg_cm2a_to_pa(gas_props['pressure_kg_cm2a'])
                                acoustic_vel = np.sqrt((gas_props['k'] * gas_props['z'] * R_UNIVERSAL * t_k) / gas_props['mw'])
                                spec_vol = (gas_props['z'] * R_UNIVERSAL * t_k) / (p_pa * gas_props['mw'])
                                rho = 1.0 / spec_vol
                                speed_factor = (2 * np.pi * gas_props['diameter_m']) / (60.0 * acoustic_vel)
                                flow_factor = 1.0 / (acoustic_vel * gas_props['diameter_m'] ** 2)
                                head_factor = 1000.0 / (acoustic_vel ** 2)
                                power_factor = (1000.0 * 30 * spec_vol) / (np.pi * acoustic_vel ** 2 * gas_props['diameter_m'] ** 3)

                                derived_df = pd.DataFrame([
                                    {'Parameter': 'Acoustic Velocity', 'Value': round(acoustic_vel, 2), 'Units': 'm/s'},
                                    {'Parameter': 'Specific Volume', 'Value': round(spec_vol, 5), 'Units': 'm3/kg'},
                                    {'Parameter': 'Rotational Speed', 'Value': f"{speed_factor:.5e}", 'Units': 'rpm'},
                                    {'Parameter': 'Volumetric Flow', 'Value': f"{flow_factor:.5e}", 'Units': 'm3/s'},
                                    {'Parameter': 'Polytropic Head', 'Value': f"{head_factor:.5e}", 'Units': 'kJ/kg'},
                                    {'Parameter': 'Power', 'Value': f"{power_factor:.5e}", 'Units': 'kW'},
                                ])
                                st.dataframe(derived_df, use_container_width=True)
                                for _, row in derived_df.iterrows():
                                    property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                            else:
                                st.info('Could not read Pressure/Temperature/MW/Compressibility as numbers — skipping derived calculations and missing-parameter steps.')
                        else:
                            st.warning(f'No operating-conditions block found in {stage}')

                        if not blocks:
                            st.warning(f'No blocks found in {stage}')
                            overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0, 'Calculated Parameter': '', 'Status': 'no blocks found'})
                            continue

                        clean_blocks = []
                        for block in blocks:
                            looks_numeric = False
                            for unit_name in (block['speed_unit'], block['flow_unit'], block['value_unit']):
                                try:
                                    float(unit_name)
                                    looks_numeric = True
                                except (ValueError, TypeError):
                                    pass
                            if looks_numeric:
                                st.warning(f"Skipping a detected block for '{block['parameter']}' at column {block['start_col']} — its units row looks like data, not units ('{block['speed_unit']}', '{block['flow_unit']}', '{block['value_unit']}').")
                            else:
                                clean_blocks.append(block)
                        blocks = clean_blocks

                        if not blocks:
                            st.warning(f'No valid blocks remained after validation in {stage}')
                            overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0, 'Calculated Parameter': '', 'Status': 'blocks failed validation'})
                            continue

                        block_summary = pd.DataFrame([
                            {'parameter': b['parameter'], 'speed_unit': b['speed_unit'], 'flow_unit': b['flow_unit'], 'value_unit': b['value_unit']}
                            for b in blocks
                        ])
                        st.dataframe(block_summary)

                        stage_models = {}
                        stage_parameters = []
                        tabs = st.tabs([b['parameter'] for b in blocks])

                        for tab, block in zip(tabs, blocks):
                            with tab:
                                param = block['parameter']
                                df, flow_conv, val_conv = self.parser.extract_block_data(raw, block)
                                stage_parameters.append(param)

                                unit_note = []
                                if not flow_conv:
                                    unit_note.append(f"flow unit '{block['flow_unit']}' not recognized, left as-is")
                                if not val_conv:
                                    unit_note.append(f"{param} unit '{block['value_unit']}' not recognized, left as-is")
                                if unit_note:
                                    st.caption('⚠ ' + '; '.join(unit_note))
                                else:
                                    st.caption(f"Converted to Flow: m³/hr, {param}: {self.converter.get_target_unit(param)}")

                                fig = go.Figure()
                                if param not in stage_models:
                                    stage_models[param] = {}

                                for speed in sorted(df['Speed'].unique()):
                                    sdf = df[df['Speed'] == speed]
                                    if len(sdf) < 4:
                                        continue

                                    x = sdf['Flow'].values.astype(float)
                                    y = sdf['Value'].values.astype(float)

                                    if method == 'Auto Best Fit':
                                        used, mdl = self.regression.auto_best(x, y)
                                    else:
                                        mdl = self.regression.build_model(x, y, method)
                                        used = method

                                    if mdl is None:
                                        continue

                                    stage_models[param][speed] = mdl
                                    r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                                    flow_fit = np.linspace(x.min(), x.max(), points)
                                    y_fit = self.regression.predict_model(mdl, flow_fit)

                                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{speed} Original'))
                                    fig.add_trace(go.Scatter(x=flow_fit, y=y_fit, mode='lines', name=f'{speed} Fit'))

                                st.plotly_chart(fig, use_container_width=True)

                        speeds = set()
                        for parameter_name in stage_models:
                            speeds.update(stage_models[parameter_name].keys())

                        export_rows = []
                        computed_param_name = None
                        stage_status = 'ok'

                        for speed in sorted(speeds):
                            available = []
                            available_params = []
                            for p in stage_models:
                                if speed in stage_models[p]:
                                    available.append(stage_models[p][speed])
                                    available_params.append(p)

                            if len(available) < 1:
                                continue

                            common_min = max(m['xmin'] for m in available)
                            common_max = min(m['xmax'] for m in available)
                            if common_max <= common_min:
                                err_msg = f"Flow values do not overlap for Stage: **{stage}** at Speed: **{speed}** across parameters ({', '.join(available_params)})."
                                st.error(err_msg)
                                stage_status = f"error: flow overlap failed at speed {speed}"
                                continue

                            common_flow = np.linspace(common_min, common_max, points)
                            temp = {'Speed': [speed] * points, 'Flow (m3/hr)': common_flow}
                            predicted = {}

                            for p in stage_models:
                                if speed in stage_models[p]:
                                    vals = self.regression.predict_model(stage_models[p][speed], common_flow)
                                    predicted[p] = vals
                                    unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(p, '')
                                    temp[f'{p} ({unit_label})'] = vals

                            if gas_props is not None:
                                try:
                                    rho = self.converter.gas_density_kg_m3(
                                        gas_props['pressure_kg_cm2a'],
                                        gas_props['temperature_c'],
                                        gas_props['mw'],
                                        gas_props['z'],
                                    )
                                    mass_flow_kg_s = common_flow * rho / 3600.0
                                    name, values = self.compute_missing_parameter(predicted, mass_flow_kg_s)
                                    if name is not None:
                                        computed_param_name = name
                                        unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(name, '')
                                        temp[f'{name} ({unit_label}, calculated)'] = values
                                        predicted[name] = values

                                    head_meters = predicted['Head']
                                    head_kj_kg = (head_meters * G) / 1000.0
                                    eff_pct = predicted['Efficiency']
                                    k_val = gas_props['k']
                                    L5 = (k_val * (eff_pct / 100.0)) / (k_val - 1.0)
                                    M5 = (acoustic_vel ** 2) / k_val
                                    pressure_ratio = (1.0 + (1000.0 * head_kj_kg) / (M5 * L5)) ** L5
                                    temp['Pressure Ratio'] = pressure_ratio
                                except (ZeroDivisionError, ValueError, KeyError) as e:
                                    st.warning(f"Could not compute missing parameter/pressure ratio for {stage} @ speed {speed}: {e}")

                            export_rows.append(pd.DataFrame(temp))

                        if computed_param_name and stage_status == 'ok':
                            st.success(f"Calculated missing parameter **{computed_param_name}** and **Pressure Ratio** for {stage} using gas density from Operating Conditions.")

                        if export_rows:
                            final_df = pd.concat(export_rows, ignore_index=True)
                            final_df.to_excel(writer, sheet_name=stage[:31], index=False)

                        overview.append({
                            'Stage': stage,
                            'Parameters': ','.join(stage_parameters),
                            'Blocks Found': len(blocks),
                            'Calculated Parameter': computed_param_name or '',
                            'Status': stage_status,
                        })
                    except Exception as e:
                        st.error(f"Error processing '{stage}': {e}")
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0, 'Calculated Parameter': '', 'Status': f'error: {e}'})

                pd.DataFrame(r2_rows, columns=['Stage', 'Speed', 'Parameter', 'Method', 'R2']).to_excel(writer, sheet_name='Summary_R2', index=False)
                pd.DataFrame(overview).to_excel(writer, sheet_name='Workbook_Overview', index=False)

                if property_rows:
                    pd.DataFrame(property_rows, columns=['Stage', 'Parameter', 'Value', 'Units']).to_excel(writer, sheet_name='Operating_Conditions', index=False)
        except Exception as e:
            fatal_error = e

        if fatal_error is not None:
            st.error(f"Could not build the output workbook: {fatal_error}")
            st.info('Nothing to download — see the error above.')
        else:
            output.seek(0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_filename = os.path.splitext(file.name)[0]
            output_filename = f'{input_filename}_Regression_Output_{timestamp}.xlsx'
            st.download_button(
                label='Download Regression Workbook',
                data=output.getvalue(),
                file_name=output_filename,
                mime='application/vnd.openxmlformats-officedocument/spreadsheetml.sheet',
            )


if __name__ == '__main__':
    app = CompressorCurveApp()
    app.run()
