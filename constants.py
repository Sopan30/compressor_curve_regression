"""Shared constants and conversion tables for compressor curve processing."""

R_UNIVERSAL = 8314.462618   # J/(kmol.K)
G = 9.80665                 # m/s^2
P_ATM_KG_CM2 = 1.033227     # Standard Atmospheric Pressure in kg/cm2

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
    'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,
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
    'm': 1.0, 'meter': 1.0, 'meters': 1.0,
}
