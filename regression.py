"""Regression model helpers for compressor curves."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


class RegressionModelBuilder:
    """Build polynomial or spline models for compressor curve data."""

    def build_model(self, x, y, meth: str):
        if meth == 'Spline':
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]
            s = CubicSpline(x, y)
            r2 = r2_score(y, s(x))
            return {'type': 'spline', 'model': s, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

        deg = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3, '4th Order': 4, '5th Order': 5}[meth]
        poly = PolynomialFeatures(deg)
        X = poly.fit_transform(x.reshape(-1, 1))
        lr = LinearRegression().fit(X, y)
        r2 = r2_score(y, lr.predict(X))
        return {'type': 'poly', 'poly': poly, 'model': lr, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

    def predict_model(self, obj, flow):
        if obj['type'] == 'spline':
            return obj['model'](flow)
        return obj['model'].predict(obj['poly'].transform(flow.reshape(-1, 1)))

    def auto_best(self, x, y):
        best = None
        best_name = None
        best_r2 = -1e9
        for method in ['Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline']:
            try:
                mdl = self.build_model(x, y, method)
                if mdl['r2'] > best_r2:
                    best_r2 = mdl['r2']
                    best = mdl
                    best_name = method
            except Exception:
                continue
        return best_name, best
