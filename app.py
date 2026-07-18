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
compressor_type = st.radio("Select Compressor Type",["Centrifugal Compressor", "Multi-Side Stream Compressor"],horizontal=True)
# if compressor_type == "Multi-Side Stream Compressor":
#     no_of_stages=st.selectbox(label="Select Number of Stages for Multi-Side Stream Compressor",options=list(range(1,11)))

method = st.sidebar.selectbox('Regression Method',
    ['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline'])
points = st.sidebar.slider('Number of Points',10,50,15)
file = st.file_uploader('Upload Workbook', type=['xlsx'])

# ---------------------------------------------------------------------------
# Physical constants and unit conversions
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.462618   # J/(kmol.K)
G = 9.80665                 # m/s^2
P_ATM_KG_CM2 = 1.033227     # Standard Atmospheric Pressure in kg/cm2 a

def normalize_unit(u):
    """Fold formatting variants ('m³/hr', 'M3/HR', 'ft^3 / min', 'Ft3.Min') onto one key."""
    s = str(u).strip().lower()
    s = s.replace('³', '3').replace('²', '2')
    s = s.replace('^3', '3').replace('^2', '2')
    s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    s = s.replace('cu', '')  # 'cuft' -> 'ft'
    return s

# All target units: Flow -> m3/hr, Head -> m, Power -> kW, Efficiency -> %
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
    'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,'lbft/lb': 0.3048,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0,
    'mm': 0.001,
    'kj/kg': 101.9716, 'j/kg': 0.1019716,
    'btu/lb': 237.2075,
}

POWER_TO_KW = {
    'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
    'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499,
    'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
    'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071,
    'btu/s': 1.055056, 'btus': 1.055056,
    'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818,
    'kcal/hr': 0.001163, 'kcal/h': 0.001163,
}

EFF_TO_PCT = {
    '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0,
    'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
}

DIAMETER_TO_M = {
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'mm': 0.001, 'milimeter': 0.001, 'milimeters': 0.001,
    'cm': 0.01, 'centimeter': 0.01,
    'm': 1.0, 'meter': 1.0, 'meters': 1.0
}

def convert_temperature_to_c(val, unit_str):
    """Handles temperature offset scales directly instead of single scalar multipliers."""
    u = str(unit_str).strip().lower()
    u = u.replace('°', '').replace('degree', '').replace('deg', '')
    u = u.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    
    if u in ['f', 'fahrenheit']:
        return (val - 32) * 5.0 / 9.0, True
    if u in ['k', 'kelvin']:
        return val - 273.15, True
    if u in ['r', 'rankine']:
        return (val - 491.67) * 5.0 / 9.0, True
    if u in ['c', 'celsius', 'centigrade']:
        return val, True
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
        'atm': 1.033227, 'atmg': 1.033227,
        'pa': 0.00001019716, 'pag': 0.00001019716
    }

    if u not in multipliers:
        return val, False

    kg_cm2_val = val * multipliers[u]

    # Absolute is now the default. Atmospheric offset is added ONLY
    # when the unit explicitly says gauge (e.g. 'psig', 'barg').
    if u.endswith('g'):
        return kg_cm2_val + P_ATM_KG_CM2, True

    return kg_cm2_val, True

def convert_unit(value, unit_str, table, label):
    key = normalize_unit(unit_str)
    if key in table:
        return value * table[key], True
    return value, False

def kg_cm2a_to_pa(kg_cm2a):
    return kg_cm2a * 98066.5

def c_to_k(deg_c):
    return deg_c + 273.15

def gas_density_kg_m3(pressure_kg_cm2a, temperature_c, mw, z):
    """Inlet gas density using metric units: rho = P*MW / (Z*R*T)."""
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

    if gas_props is None:
        return None

    required_cols = {"Speed", "Flow (m3/hr)"}
    if not required_cols.issubset(final_df.columns):
        return None

    diameter = gas_props["diameter_m"]

    q_factor = 1.0 / (acoustic_vel * diameter**2)

    hp_factor = 1000.0 / (acoustic_vel**2)

    p_factor = (1000.0 * spec_vol/(acoustic_vel**2* diameter**3* (2 * np.pi / 60.0)))

    df = final_df.copy()

    flow_m3s = df["Flow (m3/hr)"] / 3600.0

    df["Qr"] = (df["Speed"]* flow_m3s* q_factor)
    if "Head (m)" in df.columns:
        df["Hpr"] = (df["Head (m)"]* G / (acoustic_vel**2))
    if "Power (kW)" in df.columns:
        df["Pnr"] = (df["Power (kW)"]* p_factor)

    scaling = {}

    if "Hpr" in df.columns:

        qr_max = df["Qr"].max()
        hpr_max = df["Hpr"].max()

        qr_scale_head = 1.0
        hpr_scale = 1.0

        if hpr_max > qr_max:
            qr_scale_head = round(hpr_max / qr_max, 4)

        elif qr_max > hpr_max:
            hpr_scale = round(qr_max / hpr_max, 4)

        scaling.update({
            "Qr_Max_Head": qr_max,
            "Hpr_Max": hpr_max,
            "Qr_Scale_Head": qr_scale_head,
            "Hpr_Scale": hpr_scale
        })

        df["Qr_Head_Scaled"] = df["Qr"] * qr_scale_head
        df["Hpr_Scaled"] = df["Hpr"] * hpr_scale

    if "Pnr" in df.columns:

        qr_max = df["Qr"].max()
        pnr_max = df["Pnr"].max()

        qr_scale_power = 1.0
        pnr_scale = 1.0

        if pnr_max > qr_max:
            qr_scale_power = round(pnr_max / qr_max, 4)

        elif qr_max > pnr_max:
            pnr_scale = round(qr_max / pnr_max, 4)

        scaling.update({
            "Qr_Max_Power": qr_max,
            "Pnr_Max": pnr_max,
            "Qr_Scale_Power": qr_scale_power,
            "Pnr_Scale": pnr_scale
        })

        df["Qr_Power_Scaled"] = df["Qr"] * qr_scale_power
        df["Pnr_Scaled"] = df["Pnr"] * pnr_scale

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
                    blocks.append({
                        'parameter': p, 'header_row': r, 'start_col': c,
                        'speed_unit': speed_unit, 'flow_unit': flow_unit,
                        'value_unit': value_unit
                    })
    uniq = []
    seen = set()
    for b in blocks:
        k = b['start_col']
        if k not in seen:
            seen.add(k)
            uniq.append(b)
    return uniq

def extract_block_data(raw_df, block):
    r = block['header_row']
    c = block['start_col']
    data = []
    for row in range(r + 1, len(raw_df)):
        try:
            sp = float(raw_df.iloc[row, c])
            fl = float(raw_df.iloc[row, c + 1])
            val = float(raw_df.iloc[row, c + 2])
            data.append([sp, fl, val])
        except:
            pass
    df = pd.DataFrame(data, columns=['Speed', 'Flow', 'Value'])
    if df.empty:
        return df, False, False

    flow_converted = value_converted = False
    df['Flow'], flow_converted = convert_unit(df['Flow'], block['flow_unit'], FLOW_TO_M3HR, 'flow')

    param = block['parameter']
    if param == 'Head':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], HEAD_TO_M, 'head')
    elif param == 'Power':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], POWER_TO_KW, 'power')
    elif param == 'Efficiency':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], EFF_TO_PCT, 'efficiency')

    return df, flow_converted, value_converted

def detect_property_block(raw_df):
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip().lower()
            if v1 == 'parameter' and v2 == 'value' and v3 == 'units':
                return {'header_row': r, 'start_col': c}
    return None

def extract_property_block(raw_df, block):
    if block is None:
        return pd.DataFrame(columns=['Parameter', 'Value', 'Units'])
    r = block['header_row']
    c = block['start_col']
    rows_out = []
    for row in range(r + 1, len(raw_df)):
        param = raw_df.iloc[row, c]
        value = raw_df.iloc[row, c + 1]
        units = raw_df.iloc[row, c + 2]
        if pd.isna(param) or str(param).strip() == '':
            break
        
        p_name = str(param).strip()
        v_val = value.item() if hasattr(value, 'item') else value
        u_str = '' if pd.isna(units) else str(units).strip()
        
        if 'diameter' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                converted_val, success = convert_unit(float(v_val), u_str, DIAMETER_TO_M, 'diameter')
                if success:
                    v_val = converted_val
                    u_str = 'm'
            except (ValueError, TypeError):
                pass
        elif 'pressure' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                converted_val, success = convert_pressure_to_kg_cm2a(float(v_val), u_str)
                if success:
                    v_val = converted_val
                    u_str = 'kg/cm2a'
            except (ValueError, TypeError):
                pass
        elif 'temperature' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                converted_val, success = convert_temperature_to_c(float(v_val), u_str)
                if success:
                    v_val = converted_val
                    u_str = 'deg C'
            except (ValueError, TypeError):
                pass

        rows_out.append({
            'Parameter': p_name,
            'Value': v_val,
            'Units': u_str
        })
    return pd.DataFrame(rows_out, columns=['Parameter', 'Value', 'Units'])

def build_model(x, y, meth):
    if meth == 'Spline':
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]
        s = CubicSpline(x, y)
        r2 = r2_score(y, s(x))
        return {'type': 'spline', 'model': s, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

    deg = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3, '4th Order': 4, '5th Order': 5}[meth]
    poly = PolynomialFeatures(deg)
    X = poly.fit_transform(x.reshape(-1, 1))
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))
    return {'type': 'poly', 'poly': poly, 'model': lr, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

def predict_model(obj, flow):
    if obj['type'] == 'spline':
        return obj['model'](flow)
    return obj['model'].predict(obj['poly'].transform(flow.reshape(-1, 1)))

def auto_best(x, y):
    best = None; best_name = None; best_r2 = -1e9
    for m in ['Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline']:
        try:
            mdl = build_model(x, y, m)
            if mdl['r2'] > best_r2:
                best_r2 = mdl['r2']; best = mdl; best_name = m
        except:
            pass
    return best_name, best

def gas_properties_from_df(prop_df):
    lookup = {}
    for _, row in prop_df.iterrows():
        name = str(row['Parameter']).lower()
        if 'pressure' in name:
            lookup['pressure'] = row['Value']
        elif 'temperature' in name:
            lookup['temperature'] = row['Value']
        elif 'molecular weight' in name or 'mw' in name:
            lookup['mw'] = row['Value']
        elif 'compressibility' in name or 'z' in name:
            lookup['z'] = row['Value']
        elif 'isentropic' in name or name.strip().lower() == 'k':
            lookup['k'] = float(row['Value'])
            if np.isclose(lookup['k'], 1.0):
                lookup['k'] = 1.001
        elif 'diameter' in name:
            lookup['diameter'] = row['Value']
            
    try:
        return {
            'pressure_kg_cm2a': float(lookup['pressure']),
            'temperature_c': float(lookup['temperature']),
            'mw': float(lookup['mw']),
            'z': float(lookup['z']),
            'k': float(lookup.get('k', 1.4)), 
            'diameter_m': float(lookup.get('diameter', 1.0))
        }
    except (KeyError, TypeError, ValueError):
        return None

def compute_missing_parameter(available, mass_flow_kg_s):
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


def infer_tabular_data_type(series):
    if pd.api.types.is_bool_dtype(series):
        return 'Boolean'
    if pd.api.types.is_integer_dtype(series):
        return 'Int32'
    if pd.api.types.is_float_dtype(series):
        return 'Double'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'DateTime'
    return 'String'


def _format_xml_scalar(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), '.15g')
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def dataframe_to_tabular_xml(df1, dtype):
    df = df1.copy()

    if compressor_type == "Centrifugal Compressor":
        column_mapping = {
            "Speed": "Speed",
            "Flow (m3/hr)": "Inlet1_ActualVolumetricFlow",
            "Head (m)": "OperatingPolyHead",
            "Efficiency (%, calculated)": "OperatingPolyEff",
            "Efficiency (%)": "OperatingPolyEff",
            "Power (kW)": "Power",
            "Power (kW, calculated)": "Power",
            "Pressure Ratio": "PresRatio"
        }
        required_cols = [
            "Speed",
            "Inlet1_ActualVolumetricFlow",
            "OperatingPolyHead",
            "OperatingPolyEff",
            "PresRatio"
        ]

        df.rename(columns=column_mapping, inplace=True)
        df.drop(
            columns=["Power"],
            inplace=True,
            errors="ignore"
        )
        
        missing_cols = [
            c for c in required_cols
            if c not in df.columns
        ]

        for col in missing_cols:
            df[col] = np.nan

        df = df[required_cols]

    elif str(dtype).lower() == "custom":
        df.drop(
            columns=["PressureRatio"],
            inplace=True,
            errors="ignore"
        )

    parts = []

    parts.append('<?xml version="1.0" encoding="utf-16"?>')
    parts.append(
        '<TabularData '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
    )

    parts.append('<Columns>')

    for col_name in df.columns:
        parts.append('<TabularDataColumn>')

        parts.append(
            f'<Name>{escape(str(col_name))}</Name>'
        )

        parts.append(
            f'<DataType>{infer_tabular_data_type(df[col_name])}</DataType>'
        )

        parts.append('<Values>')

        for value in df[col_name].tolist():
            parts.append(
                f'<string>{escape(_format_xml_scalar(value))}</string>'
            )

        parts.append('</Values>')
        parts.append('</TabularDataColumn>')

    parts.append('</Columns>')
    parts.append('</TabularData>')

    return ''.join(parts)

if file:
    xls = pd.ExcelFile(file)
    output = BytesIO()
    r2_rows = []
    overview = []
    property_rows = []
    stage_xml_exports = []
    final_df = pd.DataFrame()
    if compressor_type != "Centrifugal Compressor":
        side_stream_df = pd.DataFrame()
        NumberOfStages = 1
    fatal_error = None

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            scaling_rows = []
            for stage in xls.sheet_names:
                st.header(stage)
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
                        
                        temp2 = None

                        if gas_props is not None:
                            t_k = c_to_k(gas_props['temperature_c'])
                            p_pa = kg_cm2a_to_pa(gas_props['pressure_kg_cm2a'])
                            
                            # 1. Intermediate Properties
                            acoustic_vel = np.sqrt((gas_props['k'] * gas_props['z'] * R_UNIVERSAL * t_k) / gas_props['mw'])
                            spec_vol = (gas_props['z'] * R_UNIVERSAL * t_k) / (p_pa * gas_props['mw'])
                            rho = 1.0 / spec_vol
                            if compressor_type == "Centrifugal Compressor":    
                                # 2. Updated Nondimensionalization Display Values
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
                        else:
                            st.info('Could not read Pressure/Temperature/MW/Compressibility as numbers — '
                                    'skipping derived calculations and missing-parameter steps.')
                    else:
                        st.warning(f'No operating-conditions block found in {stage}')

                    if not blocks:
                        st.warning(f'No blocks found in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'no blocks found'})
                        continue

                    clean_blocks = []
                    for b in blocks:
                        looks_numeric = False
                        for u in (b['speed_unit'], b['flow_unit'], b['value_unit']):
                            try:
                                float(u)
                                looks_numeric = True
                            except (ValueError, TypeError):
                                pass
                        if looks_numeric:
                            st.warning(f"Skipping a detected block for '{b['parameter']}' at column "
                                       f"{b['start_col']} — its units row looks like data, not units "
                                       f"('{b['speed_unit']}', '{b['flow_unit']}', '{b['value_unit']}'). "
                                       f"This usually means a duplicated/mislabeled header row in the source sheet.")
                        else:
                            clean_blocks.append(b)
                    blocks = clean_blocks

                    if not blocks:
                        st.warning(f'No valid blocks remained after validation in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'blocks failed validation'})
                        continue

                    block_summary = pd.DataFrame([
                        {'parameter': b['parameter'], 'speed_unit': b['speed_unit'],
                         'flow_unit': b['flow_unit'], 'value_unit': b['value_unit']}
                        for b in blocks
                    ])
                    st.dataframe(block_summary)

                    stage_models = {}
                    stage_parameters = []

                    tabs = st.tabs([b['parameter'] for b in blocks])

                    for tab, block in zip(tabs, blocks):
                        with tab:
                            param = block['parameter']
                            df, flow_conv, val_conv = extract_block_data(raw, block)
                            stage_parameters.append(param)

                            unit_note = []
                            if not flow_conv:
                                unit_note.append(f"flow unit '{block['flow_unit']}' not recognized, left as-is")
                            if not val_conv:
                                unit_note.append(f"{param} unit '{block['value_unit']}' not recognized, left as-is")
                            if unit_note:
                                st.caption('⚠ ' + '; '.join(unit_note))
                            else:
                                target_unit = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(param, '')
                                st.caption(f"Converted to Flow: m³/hr, {param}: {target_unit}")

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
                                    used, mdl = auto_best(x, y)
                                else:
                                    mdl = build_model(x, y, method)
                                    used = method

                                if mdl is None:
                                    continue

                                stage_models[param][speed] = mdl

                                r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                                flow_fit = np.linspace(x.min(), x.max(), points)
                                y_fit = predict_model(mdl, flow_fit)

                                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{speed} Original'))
                                fig.add_trace(go.Scatter(x=flow_fit, y=y_fit, mode='lines', name=f'{speed} Fit'))

                            st.plotly_chart(fig, use_container_width=True)

                    speeds = set()
                    for p in stage_models:
                        speeds.update(stage_models[p].keys())

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
                                vals = predict_model(stage_models[p][speed], common_flow)
                                predicted[p] = vals
                                unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(p, '')
                                temp[f'{p} ({unit_label})'] = vals

                        if gas_props is not None:
                            try:
                                rho = gas_density_kg_m3(gas_props['pressure_kg_cm2a'], gas_props['temperature_c'],
                                                         gas_props['mw'], gas_props['z'])
                                mass_flow_kg_s = common_flow * rho / 3600.0
                                name, values = compute_missing_parameter(predicted, mass_flow_kg_s)
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
                                if compressor_type != "Centrifugal Compressor":
                                    temp2 = pd.DataFrame(temp).copy()
                                    temp2.drop(columns=['Flow (m3/hr)'],inplace=True)
                                    mass_flow_kg_hr = mass_flow_kg_s * 3600
                                    temp2.insert(1,f'Stage{NumberOfStages}_MassFlow',mass_flow_kg_hr)
                                    temp2.rename(columns={'Head (m)':f'Stage{NumberOfStages}_OperatingPolyHead',
                                                          'Head (m, calculated)':f'Stage{NumberOfStages}_OperatingPolyHead',
                                                          'Power (kW, calculated)':f'Stage{NumberOfStages}_OperatingShaftPower',
                                                          'Power (kW)':f'Stage{NumberOfStages}_OperatingShaftPower',
                                                          'Efficiency (%)':f'Stage{NumberOfStages}_OperatingPolyEfficiency',
                                                          'Efficiency (%, calculated)':f'Stage{NumberOfStages}_OperatingPolyEfficiency',
                                                          'Pressure Ratio' : 'PressureRatio'
                                                          },inplace=True)
                                
                            except (ZeroDivisionError, ValueError, KeyError) as e:
                                st.warning(f"Could not compute missing parameter/pressure ratio for {stage} @ speed {speed}: {e}")

                        if compressor_type == "Centrifugal Compressor":
                            export_rows.append(pd.DataFrame(temp))
                        else:
                            export_rows.append(temp2)

                    if computed_param_name and stage_status == 'ok':
                        st.success(f"Calculated missing parameter **{computed_param_name}** and **Pressure Ratio** for {stage} "
                                   f"using gas density from Operating Conditions.")

                    if export_rows:
                        final_df = pd.concat(export_rows, ignore_index=True)
                        if gas_props is not None and compressor_type == "Centrifugal Compressor":
                            scaling_result = calculate_scaling_factors(final_df, gas_props, acoustic_vel, spec_vol)
                            if scaling_result is not None:
                                final_df, scaling_info = scaling_result
                                scaling_info["Stage"] = stage
                                scaling_rows.append(scaling_info)
                        final_df.to_excel(writer, sheet_name=stage[:31], index=False)
                        # if scaling_rows:
                        #     pd.DataFrame(scaling_rows).to_excel(writer, sheet_name="Scaling_Factors", index=False)
                        if compressor_type == "Centrifugal Compressor":
                            xml_content = dataframe_to_tabular_xml(final_df,'poly')
                        else:
                            xml_content = dataframe_to_tabular_xml(final_df,'custom')
                            final_df.to_excel(writer, sheet_name=f'Stage{NumberOfStages}_CustomPerformanceData', index=False)
                        stage_xml_exports.append({'Stage': stage, 'XML': xml_content})
                    
                    if compressor_type != "Centrifugal Compressor":
                        final_df.insert(0,'NumberOfStages',[NumberOfStages] * len(final_df))
                        remap={f'Stage{NumberOfStages}_MassFlow':'Stage1_MassFlow',
                               f'Stage{NumberOfStages}_OperatingPolyHead':'OperatingPolyHead',
                               f'Stage{NumberOfStages}_OperatingShaftPower':'OperatingShaftPower',
                               f'Stage{NumberOfStages}_OperatingPolyEfficiency':'OperatingPolyEfficiency'}
                        final_df.rename(columns=remap,inplace=True)
                        final_df.drop(columns=['OperatingShaftPower'],inplace=True,errors='ignore')
                        side_stream_df = pd.concat([side_stream_df,final_df],ignore_index=True)
                        NumberOfStages = NumberOfStages + 1

                    overview.append({
                        'Stage': stage,
                        'Parameters': ','.join(stage_parameters),
                        'Blocks Found': len(blocks),
                        'Calculated Parameter': computed_param_name or '',
                        'Status': stage_status
                    })
                    

                except Exception as e:
                    st.error(f"Error processing '{stage}': {e}")
                    overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                      'Calculated Parameter': '', 'Status': f'error: {e}'})

            if not side_stream_df.empty and compressor_type != "Centrifugal Compressor":
                side_stream_df.to_excel(writer,sheet_name='SideStreamPerformanceData',index=False)
                xml=dataframe_to_tabular_xml(side_stream_df,'side')
                stage_xml_exports.append({'Stage': 'SideStreamPolyPerformanceData', 'XML': xml})

            if stage_xml_exports:
                if compressor_type == "Centrifugal Compressor":
                    sheet = 'PolyPerformanceData_XML'
                else:
                    sheet = 'CustomerCurveData_XML'
                pd.DataFrame(stage_xml_exports).to_excel(writer, sheet_name=sheet, index=False)

            if property_rows:
                pd.DataFrame(
                    property_rows, columns=['Stage', 'Parameter', 'Value', 'Units']
                ).to_excel(writer, sheet_name='Operating_Conditions', index=False)
                
            pd.DataFrame(overview).to_excel(writer, sheet_name='Workbook_Overview', index=False)
            pd.DataFrame(r2_rows, columns=['Stage', 'Speed', 'Parameter', 'Method', 'R2']).to_excel(writer, sheet_name='Summary_R2', index=False)

    except Exception as e:
        fatal_error = e

    if fatal_error is not None:
        st.error(f"Could not build the output workbook: {fatal_error}")
        st.info("Nothing to download — see the error above.")
    else:
        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.splitext(file.name)[0]
        output_filename = f"{input_filename}_Regression_Output_{timestamp}.xlsx"

        st.download_button(
            label="Download Regression Workbook",
            data=output.getvalue(),
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if stage_xml_exports:
            st.subheader('Stage XML Downloads')
            for entry in stage_xml_exports:
                safe_stage_name = re.sub(r'[^A-Za-z0-9._-]+', '_', entry['Stage'])
                st.download_button(
                    label=f"Download XML for {entry['Stage']}",
                    data=entry['XML'],
                    file_name=f"{input_filename}_{safe_stage_name}_{timestamp}.xml",
                    mime="application/xml"
                )

            with st.expander('XML Export Sheet Preview', expanded=False):
                xml_sheet_df = pd.DataFrame(stage_xml_exports)
                st.dataframe(xml_sheet_df, use_container_width=True)
        else:
            st.info('No tabular data was generated for XML export yet.')
