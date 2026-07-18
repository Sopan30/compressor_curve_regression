import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from datetime import datetime
import os
import re
from xml.sax.saxutils import escape

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')

# --- Refactored Layout Configurations ---
compressor_type = st.radio("Select Compressor Type", ["Centrifugal Compressor", "Multi-Side Stream Compressor"], horizontal=True)

no_of_stages = 1
if compressor_type == "Multi-Side Stream Compressor":
    no_of_stages = st.selectbox(label="Select Number of Stages for Multi-Side Stream Compressor", options=list(range(1, 11)))

method = st.sidebar.selectbox('Regression Method', ['Auto Best Fit', 'Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline'])
points = st.sidebar.slider('Number of Points', 10, 50, 15)
file = st.file_uploader('Upload Workbook', type=['xlsx'])

# ---------------------------------------------------------------------------
# Physical constants and unit conversions
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.462618   # J/(kmol.K)
G = 9.80665                 # m/s^2
P_ATM_KG_CM2 = 1.033227     # Standard Atmospheric Pressure in kg/cm2 a

def normalize_unit(u):
    s = str(u).strip().lower()
    s = s.replace('³', '3').replace('²', '2').replace('^3', '3').replace('^2', '2')
    s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '').replace('cu', '')
    return s

FLOW_TO_M3HR = {
    'cfm': 1.699010796, 'acfm': 1.699010796, 'icfm': 1.699010796,
    'ft3/min': 1.699010796, 'ft3min': 1.699010796, 'ft/min': 1.699010796, 'cf/min': 1.699010796,
    'cfh': 0.028316847, 'ft3/hr': 0.028316847, 'ft3/h': 0.028316847, 'ft3hr': 0.028316847,
    'ft/hr': 0.028316847, 'cf/hr': 0.028316847,
    'cfs': 101.9406, 'ft3/s': 101.9406, 'ft3s': 101.9406, 'ft/s': 101.9406, 'cf/s': 101.9406,
    'm3/hr': 1.0, 'm3/h': 1.0, 'm3hr': 1.0, 'm3h': 1.0,
    'm3/min': 60.0, 'm3min': 60.0,
    'm3/s': 3600.0, 'm3s': 3600.0,
    'l/min': 0.06, 'lpm': 0.06,
    'l/s': 3.6, 'lps': 3.6, 'l/hr': 0.001, 'lph': 0.001,
    'gpm': 0.227124707, 'usgpm': 0.227124707, 'galmin': 0.227124707,
    'igpm': 0.272765, 'ukgpm': 0.272765, 'impgpm': 0.272765,
    'gph': 0.003785412, 'usgph': 0.003785412,
    'bbl/day': 0.006624459, 'bpd': 0.006624459, 'bbl/d': 0.006624459,
    'mmscfd': 1179.874,
}

HEAD_TO_M = {
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048,
    'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048, 'lbft/lb': 0.3048,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0,
    'mm': 0.001, 'kj/kg': 101.9716, 'j/kg': 0.1019716, 'btu/lb': 237.2075,
}

POWER_TO_KW = {
    'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
    'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499,
    'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
    'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071,
    'btu/s': 1.055056, 'btus': 1.055056, 'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818,
    'kcal/hr': 0.001163, 'kcal/h': 0.001163,
}

EFF_TO_PCT = {
    '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0,
    'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
}

DIAMETER_TO_M = {
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'mm': 0.001, 'milimeter': 0.001, 'milimeters': 0.001,
    'cm': 0.01, 'centimeter': 0.01, 'm': 1.0, 'meter': 1.0, 'meters': 1.0
}

def convert_temperature_to_c(val, unit_str):
    u = str(unit_str).strip().lower().replace('°', '').replace('degree', '').replace('deg', '').replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    if u in ['f', 'fahrenheit']: return (val - 32) * 5.0 / 9.0, True
    if u in ['k', 'kelvin']: return val - 273.15, True
    if u in ['r', 'rankine']: return (val - 491.67) * 5.0 / 9.0, True
    if u in ['c', 'celsius', 'centigrade']: return val, True
    return val, False

def convert_pressure_to_kg_cm2a(val, unit_str):
    u = normalize_unit(unit_str)
    multipliers = {
        'psi': 0.070306958, 'psia': 0.070306958, 'psig': 0.070306958,
        'bar': 1.01971621, 'bara': 1.01971621, 'barg': 1.01971621,
        'kpa': 0.010197162, 'kpaa': 0.010197162, 'kpag': 0.010197162,
        'mpa': 10.1971621, 'mpaa': 10.1971621, 'mpag': 10.1971621,
        'kg/cm2': 1.0, 'kg/cm2a': 1.0, 'kg/cm2g': 1.0,
        'kgf/cm2': 1.0, 'kgf/cm2a': 1.0, 'kgf/cm2g': 1.0,
        'atm': 1.033227, 'atmg': 1.033227, 'pa': 0.00001019716, 'pag': 0.00001019716
    }
    if u not in multipliers: return val, False
    kg_cm2_val = val * multipliers[u]
    if u.endswith('g'): return kg_cm2_val + P_ATM_KG_CM2, True
    return kg_cm2_val, True

def convert_unit(value, unit_str, table, label):
    key = normalize_unit(unit_str)
    if key in table: return value * table[key], True
    return value, False

def kg_cm2a_to_pa(kg_cm2a): return kg_cm2a * 98066.5
def c_to_k(deg_c): return deg_c + 273.15

def gas_density_kg_m3(pressure_kg_cm2a, temperature_c, mw, z):
    p_pa = kg_cm2a_to_pa(pressure_kg_cm2a)
    t_k = c_to_k(temperature_c)
    return (p_pa * mw) / (z * R_UNIVERSAL * t_k)

def clean_parameter_name(name):
    n = str(name).lower()
    if 'head' in n: return 'Head'
    if 'eff' in n: return 'Efficiency'
    if 'power' in n or 'bhp' in n or 'kw' in n: return 'Power'
    return str(name)

def calculate_scaling_factors(final_df, gas_props, acoustic_vel, spec_vol):
    if gas_props is None or not {"Speed", "Flow (m3/hr)"}.issubset(final_df.columns): return None
    diameter = gas_props["diameter_m"]
    q_factor = 1.0 / (acoustic_vel * diameter**2)
    p_factor = (1000.0 * spec_vol / (acoustic_vel**2 * diameter**3 * (2 * np.pi / 60.0)))
    
    df = final_df.copy()
    flow_m3s = df["Flow (m3/hr)"] / 3600.0
    df["Qr"] = (df["Speed"] * flow_m3s * q_factor)
    
    if "Head (m)" in df.columns:
        df["Hpr"] = (df["Head (m)"] * G / (acoustic_vel**2))
    if "Power (kW)" in df.columns:
        df["Pnr"] = (df["Power (kW)"] * p_factor)

    scaling = {}
    if "Hpr" in df.columns:
        qr_max, hpr_max = df["Qr"].max(), df["Hpr"].max()
        qr_scale_head = round(hpr_max / qr_max, 4) if hpr_max > qr_max else 1.0
        hpr_scale = round(qr_max / hpr_max, 4) if qr_max > hpr_max else 1.0
        scaling.update({"Qr_Max_Head": qr_max, "Hpr_Max": hpr_max, "Qr_Scale_Head": qr_scale_head, "Hpr_Scale": hpr_scale})

    if "Pnr" in df.columns:
        qr_max, pnr_max = df["Qr"].max(), df["Pnr"].max()
        qr_scale_power = round(pnr_max / qr_max, 4) if pnr_max > qr_max else 1.0
        pnr_scale = round(qr_max / pnr_max, 4) if qr_max > pnr_max else 1.0
        scaling.update({"Qr_Max_Power": qr_max, "Pnr_Max": pnr_max, "Qr_Scale_Power": qr_scale_power, "Pnr_Scale": pnr_scale})

    return df, scaling

def detect_triplet_blocks(raw_df):
    blocks = []
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip()
            if v1 == 'speed' and 'flow' in v2:
                p = clean_parameter_name(v3)
                if p.lower() != 'nan':
                    speed_unit = flow_unit = value_unit = ''
                    if r + 1 < rows:
                        speed_unit = str(raw_df.iloc[r + 1, c]).strip()
                        flow_unit = str(raw_df.iloc[r + 1, c + 1]).strip()
                        value_unit = str(raw_df.iloc[r + 1, c + 2]).strip()
                    blocks.append({'parameter': p, 'header_row': r, 'start_col': c, 'speed_unit': speed_unit, 'flow_unit': flow_unit, 'value_unit': value_unit})
    uniq = []
    seen = set()
    for b in blocks:
        if b['start_col'] not in seen:
            seen.add(b['start_col'])
            uniq.append(b)
    return uniq

def extract_block_data(raw_df, block):
    r, c = block['header_row'], block['start_col']
    data = []
    for row in range(r + 1, len(raw_df)):
        try:
            data.append([float(raw_df.iloc[row, c]), float(raw_df.iloc[row, c + 1]), float(raw_df.iloc[row, c + 2])])
        except: pass
    df = pd.DataFrame(data, columns=['Speed', 'Flow', 'Value'])
    if df.empty: return df, False, False

    df['Flow'], flow_converted = convert_unit(df['Flow'], block['flow_unit'], FLOW_TO_M3HR, 'flow')
    param = block['parameter']
    if param == 'Head': df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], HEAD_TO_M, 'head')
    elif param == 'Power': df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], POWER_TO_KW, 'power')
    elif param == 'Efficiency': df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], EFF_TO_PCT, 'efficiency')
    else: value_converted = False

    return df, flow_converted, value_converted

def detect_property_block(raw_df):
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            if str(raw_df.iloc[r, c]).strip().lower() == 'parameter' and str(raw_df.iloc[r, c+1]).strip().lower() == 'value' and str(raw_df.iloc[r, c+2]).strip().lower() == 'units':
                return {'header_row': r, 'start_col': c}
    return None

def extract_property_block(raw_df, block):
    if block is None: return pd.DataFrame(columns=['Parameter', 'Value', 'Units'])
    r, c = block['header_row'], block['start_col']
    rows_out = []
    for row in range(r + 1, len(raw_df)):
        param = raw_df.iloc[row, c]
        value = raw_df.iloc[row, c + 1]
        units = raw_df.iloc[row, c + 2]
        if pd.isna(param) or str(param).strip() == '': break
        p_name = str(param).strip()
        v_val = value.item() if hasattr(value, 'item') else value
        u_str = '' if pd.isna(units) else str(units).strip()
        
        try:
            if 'diameter' in p_name.lower():
                v_val, succ = convert_unit(float(re.split(r'[,/]', str(v_val))[0]), u_str, DIAMETER_TO_M, 'diameter')
                if succ: u_str = 'm'
            elif 'pressure' in p_name.lower():
                v_val, succ = convert_pressure_to_kg_cm2a(float(re.split(r'[,/]', str(v_val))[0]), u_str)
                if succ: u_str = 'kg/cm2a'
            elif 'temperature' in p_name.lower():
                v_val, succ = convert_temperature_to_c(float(re.split(r'[,/]', str(v_val))[0]), u_str)
                if succ: u_str = 'deg C'
        except: pass

        rows_out.append({'Parameter': p_name, 'Value': v_val, 'Units': u_str})
    return pd.DataFrame(rows_out, columns=['Parameter', 'Value', 'Units'])

def build_model(x, y, meth):
    if meth == 'Spline':
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        s = CubicSpline(x, y)
        return {'type': 'spline', 'model': s, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2_score(y, s(x))}
    deg = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3, '4th Order': 4, '5th Order': 5}[meth]
    poly = PolynomialFeatures(deg)
    X = poly.fit_transform(x.reshape(-1, 1))
    lr = LinearRegression().fit(X, y)
    return {'type': 'poly', 'poly': poly, 'model': lr, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2_score(y, lr.predict(X))}

def predict_model(obj, flow):
    if obj['type'] == 'spline': return obj['model'](flow)
    return obj['model'].predict(obj['poly'].transform(flow.reshape(-1, 1)))

def auto_best(x, y):
    best = None; best_name = None; best_r2 = -1e9
    for m in ['Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline']:
        try:
            mdl = build_model(x, y, m)
            if mdl['r2'] > best_r2: best_r2 = mdl['r2']; best = mdl; best_name = m
        except: pass
    return best_name, best

def gas_properties_from_df(prop_df):
    lookup = {}
    for _, row in prop_df.iterrows():
        name = str(row['Parameter']).lower()
        if 'pressure' in name: lookup['pressure'] = row['Value']
        elif 'temperature' in name: lookup['temperature'] = row['Value']
        elif 'molecular weight' in name or 'mw' in name: lookup['mw'] = row['Value']
        elif 'compressibility' in name or 'z' in name: lookup['z'] = row['Value']
        elif 'isentropic' in name or 'k' == name.strip(): lookup['k'] = row['Value']
        elif 'diameter' in name: lookup['diameter'] = row['Value']
    try:
        return {'pressure_kg_cm2a': float(lookup['pressure']), 'temperature_c': float(lookup['temperature']), 'mw': float(lookup['mw']), 'z': float(lookup['z']), 'k': float(lookup.get('k', 1.4)), 'diameter_m': float(lookup.get('diameter', 1.0))}
    except: return None

def compute_missing_parameter(available, mass_flow_kg_s):
    have = set(available.keys())
    needed = {'Head', 'Efficiency', 'Power'} - have
    if len(needed) != 1: return None, None
    missing = needed.pop()
    if missing == 'Power': return 'Power', mass_flow_kg_s * G * available['Head'] / (available['Efficiency'] / 100.0) / 1000.0
    if missing == 'Head': return 'Head', available['Power'] * 1000.0 * (available['Efficiency'] / 100.0) / (mass_flow_kg_s * G)
    if missing == 'Efficiency': return 'Efficiency', (mass_flow_kg_s * G * available['Head']) / (available['Power'] * 1000.0) * 100.0
    return None, None

def infer_tabular_data_type(series):
    if pd.api.types.is_bool_dtype(series): return 'Boolean'
    if pd.api.types.is_integer_dtype(series): return 'Int32'
    if pd.api.types.is_float_dtype(series): return 'Double'
    if pd.api.types.is_datetime64_any_dtype(series): return 'DateTime'
    return 'String'

def _format_xml_scalar(value):
    if pd.isna(value): return ''
    if isinstance(value, (bool, np.bool_)): return 'true' if value else 'false'
    if isinstance(value, (np.integer, int)): return str(int(value))
    if isinstance(value, (np.floating, float)): return format(float(value), '.15g')
    if isinstance(value, datetime): return value.isoformat()
    return str(value)

def dataframe_to_tabular_xml(df):
    column_mapping = {"Speed": "Speed", "Flow (m3/hr)": "Inlet1_ActualVolumetricFlow", "Head (m)": "OperatingPolyHead", "Efficiency (%)": "OperatingPolyEff", "Efficiency (%, calculated)": "OperatingPolyEff", "Pressure Ratio": "PresRatio"}
    required_cols = ["Speed", "Inlet1_ActualVolumetricFlow", "OperatingPolyHead", "OperatingPolyEff", "PresRatio"]
    df = df.rename(columns=column_mapping).drop(columns=["Power"], errors="ignore")
    for missing in [c for c in required_cols if c not in df.columns]: df[missing] = np.nan
    df = df[required_cols]
    
    parts = ['<?xml version="1.0" encoding="utf-16"?>', '<TabularData xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">', '<Columns>']
    for col_name in df.columns:
        parts.extend([f'<TabularDataColumn><Name>{escape(str(col_name))}</Name>', f'<DataType>{infer_tabular_data_type(df[col_name])}</DataType>', '<Values>'])
        for value in df[col_name].tolist(): parts.append(f'<string>{escape(_format_xml_scalar(value))}</string>')
        parts.extend(['</Values>', '</TabularDataColumn>'])
    parts.extend(['</Columns>', '</TabularData>'])
    return ''.join(parts)


if file:
    xls = pd.ExcelFile(file)
    output = BytesIO()
    r2_rows, overview, property_rows, stage_xml_exports = [], [], [], []
    fatal_error = None
    
    # Track the consolidated data row-by-row across stages
    all_stages_consolidated_df = pd.DataFrame()

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            scaling_rows = []
            
            # Dynamically resolve sheets to match specified configuration metrics
            target_sheets = xls.sheet_names[:no_of_stages] if compressor_type == "Multi-Side Stream Compressor" else xls.sheet_names

            for idx, stage in enumerate(target_sheets):
                st.header(f"Processing: {stage}")
                stage_idx_num = idx + 1
                try:
                    raw = pd.read_excel(file, sheet_name=stage, header=None)
                    blocks = detect_triplet_blocks(raw)
                    prop_block = detect_property_block(raw)
                    prop_df = extract_property_block(raw, prop_block)
                    gas_props = None
                    
                    if not prop_df.empty:
                        st.subheader('Operating Conditions')
                        st.dataframe(prop_df, use_container_width=True)
                        for _, row in prop_df.iterrows():
                            property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                        gas_props = gas_properties_from_df(prop_df)
                        
                        if gas_props is not None:
                            t_k = c_to_k(gas_props['temperature_c'])
                            p_pa = kg_cm2a_to_pa(gas_props['pressure_kg_cm2a'])
                            acoustic_vel = np.sqrt((gas_props['k'] * gas_props['z'] * R_UNIVERSAL * t_k) / gas_props['mw'])
                            spec_vol = (gas_props['z'] * R_UNIVERSAL * t_k) / (p_pa * gas_props['mw'])
                            
                            speed_factor = (2 * np.pi * gas_props['diameter_m']) / (60.0 * acoustic_vel)
                            flow_factor = 1.0 / (acoustic_vel * gas_props['diameter_m']**2)
                            head_factor = 1000.0 / (acoustic_vel**2)
                            power_factor = (1000.0 * 30 * spec_vol) / (np.pi * acoustic_vel**2 * gas_props['diameter_m']**3)
                            
                            derived_df = pd.DataFrame([
                                {'Parameter': 'Acoustic Velocity', 'Value': round(acoustic_vel, 2), 'Units': 'm/s'},
                                {'Parameter': 'Specific Volume', 'Value': round(spec_vol, 5), 'Units': 'm3/kg'},
                                {'Parameter': 'Rotational Speed', 'Value': f"{speed_factor:.5e}", 'Units': 'rpm'},
                                {'Parameter': 'Volumetric Flow', 'Value': f"{flow_factor:.5e}", 'Units': 'm3/s'},
                                {'Parameter': 'Polytropic Head', 'Value': f"{head_factor:.5e}", 'Units': 'kJ/kg'},
                                {'Parameter': 'Power', 'Value': f"{power_factor:.5e}", 'Units': 'kW'}
                            ])
                            st.dataframe(derived_df, use_container_width=True)
                            for _, row in derived_df.iterrows():
                                property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])

                    if not blocks:
                        overview.append({'Stage': stage, 'Status': 'No Valid Blocks Found'})
                        continue

                    stage_models = {}
                    stage_parameters = []
                    tabs = st.tabs([b['parameter'] for b in blocks])

                    for tab, block in zip(tabs, blocks):
                        with tab:
                            param = block['parameter']
                            df, flow_conv, val_conv = extract_block_data(raw, block)
                            stage_parameters.append(param)
                            
                            if param not in stage_models: stage_models[param] = {}
                            fig = go.Figure()

                            for speed in sorted(df['Speed'].unique()):
                                sdf = df[df['Speed'] == speed]
                                if len(sdf) < 4: continue
                                x, y = sdf['Flow'].values.astype(float), sdf['Value'].values.astype(float)

                                used, mdl = auto_best(x, y) if method == 'Auto Best Fit' else (method, build_model(x, y, method))
                                if mdl is None: continue
                                stage_models[param][speed] = mdl
                                r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                                flow_fit = np.linspace(x.min(), x.max(), points)
                                y_fit = predict_model(mdl, flow_fit)
                                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{speed} Original'))
                                fig.add_trace(go.Scatter(x=flow_fit, y=y_fit, mode='lines', name=f'{speed} Fit'))

                            st.plotly_chart(fig, use_container_width=True)

                    speeds = set()
                    for p in stage_models: speeds.update(stage_models[p].keys())
                    export_rows = []

                    for speed in sorted(speeds):
                        available = [stage_models[p][speed] for p in stage_models if speed in stage_models[p]]
                        if not available: continue
                        common_min, common_max = max(m['xmin'] for m in available), min(m['xmax'] for m in available)
                        if common_max <= common_min: continue

                        common_flow = np.linspace(common_min, common_max, points)
                        temp = {'Speed': [speed] * points, 'Flow (m3/hr)': common_flow}
                        predicted = {}

                        for p in stage_models:
                            if speed in stage_models[p]:
                                vals = predict_model(stage_models[p][speed], common_flow)
                                predicted[p] = vals
                                temp[f'{p} ({"m" if p=="Head" else "kW" if p=="Power" else "%"})'] = vals

                        if gas_props is not None:
                            try:
                                rho = gas_density_kg_m3(gas_props['pressure_kg_cm2a'], gas_props['temperature_c'], gas_props['mw'], gas_props['z'])
                                mass_flow_kg_s = common_flow * rho / 3600.0
                                name, values = compute_missing_parameter(predicted, mass_flow_kg_s)
                                if name:
                                    temp[f'{name} ({"m" if name=="Head" else "kW" if name=="Power" else "%"}, calculated)'] = values
                                    predicted[name] = values
                                
                                head_kj_kg = (predicted['Head'] * G) / 1000.0
                                L5 = (gas_props['k'] * (predicted['Efficiency'] / 100.0)) / (gas_props['k'] - 1.0)
                                temp['Pressure Ratio'] = (1.0 + (1000.0 * head_kj_kg) / (((acoustic_vel ** 2) / gas_props['k']) * L5)) ** L5
                            except: pass

                        export_rows.append(pd.DataFrame(temp))

                    if export_rows:
                        stage_df = pd.concat(export_rows, ignore_index=True)
                        
                        # Apply global tracking filters across structures
                        if compressor_type == "Multi-Side Stream Compressor":
                            stage_df.insert(0, 'Stage_Number', stage_idx_num)
                        
                        all_stages_consolidated_df = pd.concat([all_stages_consolidated_df, stage_df], ignore_index=True)
                        stage_df.to_excel(writer, sheet_name=stage[:31], index=False)
                        
                        xml_content = dataframe_to_tabular_xml(stage_df)
                        stage_xml_exports.append({'Stage': stage, 'XML': xml_content})

                    overview.append({'Stage': stage, 'Parameters': ','.join(stage_parameters), 'Blocks Found': len(blocks), 'Status': 'Success'})
                except Exception as e:
                    st.error(f"Error processing context sheet '{stage}': {e}")

            # Append the structured dataset safely to tracking workbooks
            if not all_stages_consolidated_df.empty:
                all_stages_consolidated_df.to_excel(writer, sheet_name='Consolidated_Regression', index=False)
            if stage_xml_exports:
                pd.DataFrame(stage_xml_exports).to_excel(writer, sheet_name='PolyPerformanceData', index=False)
                
            pd.DataFrame(overview).to_excel(writer, sheet_name='Workbook_Overview', index=False)
            if property_rows:
                pd.DataFrame(property_rows, columns=['Stage', 'Parameter', 'Value', 'Units']).to_excel(writer, sheet_name='Operating_Conditions', index=False)

    except Exception as e:
        fatal_error = e

    if fatal_error:
        st.error(f"Could not build regression output structures: {fatal_error}")
    else:
        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{os.path.splitext(file.name)[0]}_Regression_Output_{timestamp}.xlsx"

        st.download_button(label="Download Regression Workbook", data=output.getvalue(), file_name=output_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
