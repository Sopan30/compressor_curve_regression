# Compressor Curve Regression Tool

A Streamlit-based web application for performing polynomial and spline regression analysis on compressor performance curves. This tool processes compressor data from Excel workbooks and generates fitted curves with R² metrics.

## Features

- **Multiple Regression Methods**
  - Linear regression
  - Polynomial regression (Quadratic, Cubic, 4th, 5th Order)
  - Cubic spline interpolation
  - Automatic best-fit selection based on R² score

- **Batch Processing**
  - Process multiple workbook sheets in one upload
  - Automatic column detection (Speed, Flow, Head, Efficiency, Power)
  - Generate fitted curves for multiple operating conditions

- **Interactive Visualizations**
  - Plotly charts for each parameter
  - Original data points vs. fitted curves
  - Multiple speed lines per chart

- **Excel Output**
  - Workbook overview sheet
  - R² summary statistics
  - Fitted curve data with original parameters
  - One output sheet per stage

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sopan30/compressor_curve_regression.git
cd compressor_curve_regression
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Input Data Format

Your Excel workbook should contain:
- **Required columns:**
  - Speed (RPM or equivalent)
  - Flow (volumetric flow rate)

- **Optional columns:**
  - Head (pressure head)
  - Efficiency
  - Power (BHP or kW)

The tool will automatically detect these columns regardless of naming convention.

### Workflow

1. **Upload Excel File**: Use the file uploader to select your `.xlsx` workbook
2. **Select Regression Method**: Choose from dropdown menu:
   - Auto Best Fit (recommended for initial analysis)
   - Specific polynomial/spline method
3. **Set Generated Points**: Adjust the number of interpolated points (10-200)
4. **View Results**: 
   - Interactive charts display for each parameter
   - R² values show regression quality
   - Download fitted data as Excel workbook

### Configuration

Use the sidebar to adjust:
- **Regression Method**: Default is "Auto Best Fit"
- **Generated Points**: Number of points to generate on the fitted curve (default: 15)

## Output Files

The generated Excel workbook includes:

| Sheet | Description |
|-------|-------------|
| **Workbook_Overview** | Summary of all processed stages and parameters |
| **Summary_R2** | R² scores for each regression showing goodness of fit |
| **[Stage Name]** | Fitted curve data (Speed, Flow, and fitted values) |

## How It Works

### Column Detection
The `detect_columns()` function searches for common naming patterns:
- Speed: "speed"
- Flow: "flow", "volumeflow", "inlet1_volumeflowrate"
- Head: "head"
- Efficiency: "eff", "efficiency"
- Power: "power", "bhp", "kw"

### Regression Methods

**Polynomial Fitting**: Uses scikit-learn's `LinearRegression` with `PolynomialFeatures` to fit polynomial models of specified degree.

**Spline Fitting**: Uses SciPy's `CubicSpline` for smooth interpolation through data points.

**Auto Best Fit**: Tests all methods and selects the one with highest R² score.

### Data Processing

1. For each workbook sheet (stage):
   - Detect relevant columns
   - Group data by speed
   - For each speed:
     - Apply selected regression method
     - Calculate R² score
     - Generate fitted curve with specified number of points
   - Merge results across all parameters
   - Write to output sheet

## Example

Input file structure:
```
Sheet: "Stage_1"
| Speed | Flow | Head | Efficiency | Power |
|-------|------|------|------------|-------|
| 3500  | 100  | 250  | 0.85       | 45    |
| 3500  | 150  | 240  | 0.87       | 68    |
| 3600  | 100  | 255  | 0.86       | 46    |
| ...   | ...  | ...  | ...        | ...   |
```

Output will include:
- Interactive visualization showing original points and fitted curve
- Fitted data with interpolated values
- R² metric indicating fit quality

## Requirements

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- scipy
- openpyxl (for Excel support)

## Technical Details

### Minimum Data Requirements
- At least 4 data points per speed curve for polynomial fitting
- Cubic spline will use available points

### Performance
- Typical processing time: <1 second per stage
- Auto best fit tests 6 methods per speed curve
- Output file size depends on generated point count and data volume

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Columns not detected | Check column naming against detection patterns; rename if needed |
| "Skipping stage" message | Stage missing Speed or Flow columns |
| High memory usage | Reduce "Generated Points" slider value |
| Spline fitting error | Ensure x values are sorted and unique (handled automatically) |

## License

[Add your license here, e.g., MIT, Apache 2.0]

## Author

Sopan30

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an GitHub issue in the repository.
