
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')

METHODS=['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline']
fit_method=st.sidebar.selectbox('Regression Method', METHODS)
generated_points=st.sidebar.slider('Generated Points',10,200,15)
uploaded_file=st.file_uploader('Upload Workbook',type=['xlsx'])

def clean_parameter_name(name):
    n=str(name).lower()
    if 'head' in n: return 'Head'
    if 'eff' in n: return 'Efficiency'
    if 'power' in n or 'bhp' in n or 'kw' in n: return 'Power'
    return str(name)

def detect_triplet_blocks(raw_df):
    blocks=[]
    rows,cols=raw_df.shape
    for r in range(rows):
        for c in range(cols-2):
            v1=str(raw_df.iloc[r,c]).strip().lower()
            v2=str(raw_df.iloc[r,c+1]).strip().lower()
            v3=str(raw_df.iloc[r,c+2]).strip()
            if v1=='speed' and ('flow' in v2):
                p=clean_parameter_name(v3)
                if p.lower()!='nan':
                    blocks.append({'parameter':p,'header_row':r,'start_col':c})
    unique=[]
    seen=set()
    for b in blocks:
        k=(b['parameter'],b['start_col'])
        if k not in seen:
            unique.append(b); seen.add(k)
    return unique

def extract_block_data(raw_df, block):
    r=block['header_row']
    c=block['start_col']
    data=[]
    for row in range(r+1,len(raw_df)):
        try:
            sp=float(raw_df.iloc[row,c])
            fl=float(raw_df.iloc[row,c+1])
            val=float(raw_df.iloc[row,c+2])
            data.append([sp,fl,val])
        except:
            pass
    return pd.DataFrame(data,columns=['Speed','Flow','Value'])

def poly_fit(x,y,d,n):
    poly=PolynomialFeatures(d)
    X=poly.fit_transform(x.reshape(-1,1))
    m=LinearRegression().fit(X,y)
    r2=r2_score(y,m.predict(X))
    xn=np.linspace(x.min(),x.max(),n)
    yn=m.predict(poly.transform(xn.reshape(-1,1)))
    return xn,yn,r2

def spline_fit(x,y,n):
    idx=np.argsort(x)
    x,y=x[idx],y[idx]
    s=CubicSpline(x,y)
    r2=r2_score(y,s(x))
    xn=np.linspace(x.min(),x.max(),n)
    return xn,s(xn),r2

def run_method(method,x,y,n):
    if method=='Linear': return poly_fit(x,y,1,n)
    if method=='Quadratic': return poly_fit(x,y,2,n)
    if method=='Cubic': return poly_fit(x,y,3,n)
    if method=='4th Order': return poly_fit(x,y,4,n)
    if method=='5th Order': return poly_fit(x,y,5,n)
    return spline_fit(x,y,n)

def auto_best(x,y,n):
    best=None; bestr=-1e9; bestm=''
    for m in METHODS[1:]:
        try:
            r=run_method(m,x,y,n)
            if r[2]>bestr:
                best=r; bestr=r[2]; bestm=m
        except:
            pass
    return bestm,best

if uploaded_file:
    xls=pd.ExcelFile(uploaded_file)
    output=BytesIO()
    r2_rows=[]
    overview=[]
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        for sheet in xls.sheet_names:
            raw=pd.read_excel(uploaded_file,sheet_name=sheet,header=None)
            st.header(sheet)
            blocks=detect_triplet_blocks(raw)
            if not blocks:
                st.warning(f'No Speed-Flow-Parameter blocks found in {sheet}')
                continue
            overview.append({'Stage':sheet,'Blocks':len(blocks),'Parameters':','.join([b['parameter'] for b in blocks])})
            tabs=st.tabs([b['parameter'] for b in blocks])
            for tab,block in zip(tabs,blocks):
                with tab:
                    df=extract_block_data(raw,block)
                    st.write(f"Rows detected: {len(df)}")
                    fig=go.Figure()
                    exports=[]
                    for speed in sorted(df.Speed.unique()):
                        sdf=df[df.Speed==speed]
                        if len(sdf)<4: continue
                        x=sdf.Flow.values
                        y=sdf.Value.values
                        if fit_method=='Auto Best Fit':
                            used,res=auto_best(x,y,generated_points)
                            if res is None: continue
                            xf,yf,r2=res
                        else:
                            xf,yf,r2=run_method(fit_method,x,y,generated_points)
                            used=fit_method
                        r2_rows.append([sheet,speed,block['parameter'],used,r2])
                        fig.add_trace(go.Scatter(x=x,y=y,mode='markers',name=f'{speed} Original'))
                        fig.add_trace(go.Scatter(x=xf,y=yf,mode='lines',name=f'{speed} Fit'))
                        exports.append(pd.DataFrame({'Speed':speed,'Flow':xf,block['parameter']:yf}))
                    st.plotly_chart(fig,use_container_width=True)
                    if exports:
                        exp=pd.concat(exports,ignore_index=True)
                        exp.to_excel(writer,sheet_name=f"{sheet}_{block['parameter']}"[:31],index=False)
        pd.DataFrame(r2_rows,columns=['Stage','Speed','Parameter','Method','R2']).to_excel(writer,sheet_name='Summary_R2',index=False)
        pd.DataFrame(overview).to_excel(writer,sheet_name='Workbook_Overview',index=False)
    st.download_button('Download Regression Workbook',output.getvalue(),'Regression_Output.xlsx')
