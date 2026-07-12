"""Unit conversion helpers for compressor curve data."""

from __future__ import annotations

from compressor_curve_regression.constants import (
    DIAMETER_TO_M,
    EFF_TO_PCT,
    FLOW_TO_M3HR,
    G,
    HEAD_TO_M,
    P_ATM_KG_CM2,
    POWER_TO_KW,
    R_UNIVERSAL,
)


class UnitConverter:
    """Normalize and convert various engineering units."""

    def normalize_unit(self, unit_value: object) -> str:
        """Normalize user-entered unit strings into a consistent key."""
        s = str(unit_value).strip().lower()
        s = s.replace('³', '3').replace('²', '2')
        s = s.replace('^3', '3').replace('^2', '2')
        s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
        s = s.replace('cu', '')
        return s

    def convert_unit(self, value: float, unit_str: object, table: dict[str, float], label: str) -> tuple[float, bool]:
        key = self.normalize_unit(unit_str)
        if key in table:
            return value * table[key], True
        return value, False

    def convert_pressure_to_kg_cm2a(self, val: float, unit_str: object) -> tuple[float, bool]:
        u = self.normalize_unit(unit_str)
        multipliers = {
            'psi': 0.070306958, 'psia': 0.070306958, 'psig': 0.070306958,
            'bar': 1.01971621, 'bara': 1.01971621, 'barg': 1.01971621,
            'kpa': 0.010197162, 'kpaa': 0.010197162, 'kpag': 0.010197162,
            'mpa': 10.1971621, 'mpaa': 10.1971621, 'mpag': 10.1971621,
            'kg/cm2': 1.0, 'kg/cm2a': 1.0, 'kg/cm2g': 1.0,
            'kgf/cm2': 1.0, 'kgf/cm2a': 1.0, 'kgf/cm2g': 1.0,
            'atm': 1.033227, 'atmg': 1.033227,
            'pa': 0.00001019716, 'pag': 0.00001019716,
        }

        if u not in multipliers:
            return val, False

        kg_cm2_val = val * multipliers[u]
        if u.endswith('g'):
            return kg_cm2_val + P_ATM_KG_CM2, True
        return kg_cm2_val, True

    def convert_temperature_to_c(self, val: float, unit_str: object) -> tuple[float, bool]:
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

    def kg_cm2a_to_pa(self, kg_cm2a: float) -> float:
        return kg_cm2a * 98066.5

    def c_to_k(self, deg_c: float) -> float:
        return deg_c + 273.15

    def gas_density_kg_m3(self, pressure_kg_cm2a: float, temperature_c: float, mw: float, z: float) -> float:
        p_pa = self.kg_cm2a_to_pa(pressure_kg_cm2a)
        t_k = self.c_to_k(temperature_c)
        return (p_pa * mw) / (z * R_UNIVERSAL * t_k)

    def convert_flow_to_m3_hr(self, value: float, unit_str: object) -> tuple[float, bool]:
        return self.convert_unit(value, unit_str, FLOW_TO_M3HR, 'flow')

    def convert_head_to_m(self, value: float, unit_str: object) -> tuple[float, bool]:
        return self.convert_unit(value, unit_str, HEAD_TO_M, 'head')

    def convert_power_to_kw(self, value: float, unit_str: object) -> tuple[float, bool]:
        return self.convert_unit(value, unit_str, POWER_TO_KW, 'power')

    def convert_efficiency_to_pct(self, value: float, unit_str: object) -> tuple[float, bool]:
        return self.convert_unit(value, unit_str, EFF_TO_PCT, 'efficiency')

    def convert_diameter_to_m(self, value: float, unit_str: object) -> tuple[float, bool]:
        return self.convert_unit(value, unit_str, DIAMETER_TO_M, 'diameter')

    def get_target_unit(self, parameter_name: str) -> str:
        return {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(parameter_name, '')
