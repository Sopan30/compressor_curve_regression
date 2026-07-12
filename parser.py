"""Excel sheet parsing and block extraction for compressor data."""

from __future__ import annotations

import re

import pandas as pd

from compressor_curve_regression.constants import DIAMETER_TO_M, EFF_TO_PCT, FLOW_TO_M3HR, HEAD_TO_M, POWER_TO_KW
from compressor_curve_regression.units import UnitConverter


class SheetProcessor:
    """Detect operating conditions and performance triplet blocks from workbook sheets."""

    def __init__(self, converter: UnitConverter | None = None):
        self.converter = converter or UnitConverter()

    def clean_parameter_name(self, name: object) -> str:
        n = str(name).lower()
        if 'head' in n:
            return 'Head'
        if 'eff' in n:
            return 'Efficiency'
        if 'power' in n or 'bhp' in n or 'kw' in n:
            return 'Power'
        return str(name)

    def detect_triplet_blocks(self, raw_df: pd.DataFrame) -> list[dict]:
        blocks = []
        rows, cols = raw_df.shape
        for r in range(rows):
            for c in range(cols - 2):
                v1 = str(raw_df.iloc[r, c]).strip().lower()
                v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
                v3 = str(raw_df.iloc[r, c + 2]).strip()
                if v1 == 'speed' and 'flow' in v2:
                    p = self.clean_parameter_name(v3)
                    if p.lower() != 'nan':
                        speed_unit = flow_unit = value_unit = ''
                        if r + 1 < rows:
                            speed_unit = str(raw_df.iloc[r + 1, c]).strip()
                            flow_unit = str(raw_df.iloc[r + 1, c + 1]).strip()
                            value_unit = str(raw_df.iloc[r + 1, c + 2]).strip()
                        blocks.append({
                            'parameter': p,
                            'header_row': r,
                            'start_col': c,
                            'speed_unit': speed_unit,
                            'flow_unit': flow_unit,
                            'value_unit': value_unit,
                        })

        uniq = []
        seen = set()
        for block in blocks:
            k = block['start_col']
            if k not in seen:
                seen.add(k)
                uniq.append(block)
        return uniq

    def extract_block_data(self, raw_df: pd.DataFrame, block: dict) -> tuple[pd.DataFrame, bool, bool]:
        r = block['header_row']
        c = block['start_col']
        data = []
        for row in range(r + 1, len(raw_df)):
            try:
                sp = float(raw_df.iloc[row, c])
                fl = float(raw_df.iloc[row, c + 1])
                val = float(raw_df.iloc[row, c + 2])
                data.append([sp, fl, val])
            except Exception:
                pass
        df = pd.DataFrame(data, columns=['Speed', 'Flow', 'Value'])
        if df.empty:
            return df, False, False

        flow_converted = value_converted = False
        df['Flow'], flow_converted = self.converter.convert_flow_to_m3_hr(df['Flow'], block['flow_unit'])

        param = block['parameter']
        if param == 'Head':
            df['Value'], value_converted = self.converter.convert_head_to_m(df['Value'], block['value_unit'])
        elif param == 'Power':
            df['Value'], value_converted = self.converter.convert_power_to_kw(df['Value'], block['value_unit'])
        elif param == 'Efficiency':
            df['Value'], value_converted = self.converter.convert_efficiency_to_pct(df['Value'], block['value_unit'])
        return df, flow_converted, value_converted

    def detect_property_block(self, raw_df: pd.DataFrame) -> dict | None:
        rows, cols = raw_df.shape
        for r in range(rows):
            for c in range(cols - 2):
                v1 = str(raw_df.iloc[r, c]).strip().lower()
                v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
                v3 = str(raw_df.iloc[r, c + 2]).strip().lower()
                if v1 == 'parameter' and v2 == 'value' and v3 == 'units':
                    return {'header_row': r, 'start_col': c}
        return None

    def extract_property_block(self, raw_df: pd.DataFrame, block: dict | None) -> pd.DataFrame:
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
                    converted_val, success = self.converter.convert_diameter_to_m(float(v_val), u_str)
                    if success:
                        v_val = converted_val
                        u_str = 'm'
                except (ValueError, TypeError):
                    pass
            elif 'pressure' in p_name.lower():
                try:
                    if isinstance(v_val, str):
                        v_val = re.split(r'[,/]', v_val)[0].strip()
                    converted_val, success = self.converter.convert_pressure_to_kg_cm2a(float(v_val), u_str)
                    if success:
                        v_val = converted_val
                        u_str = 'kg/cm2a'
                except (ValueError, TypeError):
                    pass
            elif 'temperature' in p_name.lower():
                try:
                    if isinstance(v_val, str):
                        v_val = re.split(r'[,/]', v_val)[0].strip()
                    converted_val, success = self.converter.convert_temperature_to_c(float(v_val), u_str)
                    if success:
                        v_val = converted_val
                        u_str = 'deg C'
                except (ValueError, TypeError):
                    pass

            rows_out.append({'Parameter': p_name, 'Value': v_val, 'Units': u_str})
        return pd.DataFrame(rows_out, columns=['Parameter', 'Value', 'Units'])

    def gas_properties_from_df(self, prop_df: pd.DataFrame) -> dict | None:
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
            elif 'isentropic' in name or 'k' == name.strip():
                lookup['k'] = row['Value']
            elif 'diameter' in name:
                lookup['diameter'] = row['Value']

        try:
            return {
                'pressure_kg_cm2a': float(lookup['pressure']),
                'temperature_c': float(lookup['temperature']),
                'mw': float(lookup['mw']),
                'z': float(lookup['z']),
                'k': float(lookup.get('k', 1.4)),
                'diameter_m': float(lookup.get('diameter', 1.0)),
            }
        except (KeyError, TypeError, ValueError):
            return None
