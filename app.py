
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

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')

method = st.sidebar.selectbox('Regression Method',
    ['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline'])
points = st.sidebar.slider('Generated Points',15,50,15)
file = st.file_uploader('Upload Workbook', type=['xlsx'])


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
            if v1=='speed' and 'flow' in v2:
                p=clean_parameter_name(v3)
                if p.lower()!='nan':
                    blocks.append({'parameter':p,'header_row':r,'start_col':c})
    uniq=[]; seen=set()
    for b in blocks:
        k=(b['parameter'],b['start_col'])
        if k not in seen:
            seen.add(k); uniq.append(b)
    return uniq


def extract_block_data(raw_df, block):
    r=block['header_row']; c=block['start_col']
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


def build_model(x,y,meth):
    if meth=='Spline':
        idx=np.argsort(x)
        x=x[idx]; y=y[idx]
        s=CubicSpline(x,y)
        r2=r2_score(y,s(x))
        return {'type':'spline','model':s,'xmin':x.min(),'xmax':x.max(),'r2':r2}

    deg={'Linear':1,'Quadratic':2,'Cubic':3,'4th Order':4,'5th Order':5}[meth]
    poly=PolynomialFeatures(deg)
    X=poly.fit_transform(x.reshape(-1,1))
    lr=LinearRegression().fit(X,y)
    r2=r2_score(y,lr.predict(X))
    return {'type':'poly','poly':poly,'model':lr,'xmin':x.min(),'xmax':x.max(),'r2':r2}


def predict_model(obj,flow):
    if obj['type']=='spline':
        return obj['model'](flow)
    return obj['model'].predict(obj['poly'].transform(flow.reshape(-1,1)))


def auto_best(x,y):
    best=None; best_name=None; best_r2=-1e9
    for m in ['Linear','Quadratic','Cubic','4th Order','5th Order','Spline']:
        try:
            mdl=build_model(x,y,m)
            if mdl['r2']>best_r2:
                best_r2=mdl['r2']; best=mdl; best_name=m
        except:
            pass
    return best_name,best


if file:
    xls=pd.ExcelFile(file)
    output=BytesIO()
    r2_rows=[]
    overview=[]

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        for stage in xls.sheet_names:
            st.header(stage)
            raw=pd.read_excel(file,sheet_name=stage,header=None)
            blocks=detect_triplet_blocks(raw)

            if not blocks:
                st.warning(f'No blocks found in {stage}')
                continue

            st.dataframe(pd.DataFrame(blocks))

            stage_models={}
            stage_parameters=[]

            tabs=st.tabs([b['parameter'] for b in blocks])

            for tab,block in zip(tabs,blocks):
                with tab:
                    param=block['parameter']
                    df=extract_block_data(raw,block)
                    stage_parameters.append(param)

                    fig=go.Figure()

                    if param not in stage_models:
                        stage_models[param]={}

                    for speed in sorted(df['Speed'].unique()):
                        sdf=df[df['Speed']==speed]
                        if len(sdf)<4:
                            continue

                        x=sdf['Flow'].values.astype(float)
                        y=sdf['Value'].values.astype(float)

                        if method=='Auto Best Fit':
                            used,mdl=auto_best(x,y)
                        else:
                            mdl=build_model(x,y,method)
                            used=method

                        stage_models[param][speed]=mdl

                        r2_rows.append([
                            stage,speed,param,used,round(mdl['r2'],6)
                        ])

                        flow_fit=np.linspace(x.min(),x.max(),points)
                        y_fit=predict_model(mdl,flow_fit)

                        fig.add_trace(go.Scatter(x=x,y=y,mode='markers',name=f'{speed} Original'))
                        fig.add_trace(go.Scatter(x=flow_fit,y=y_fit,mode='lines',name=f'{speed} Fit'))

                    st.plotly_chart(fig,use_container_width=True)

            # COMMON FLOW EXPORT PER SPEED
            speeds=set()
            for p in stage_models:
                speeds.update(stage_models[p].keys())

            export_rows=[]

            for speed in sorted(speeds):
                available=[]
                for p in stage_models:
                    if speed in stage_models[p]:
                        available.append(stage_models[p][speed])

                if len(available)<1:
                    continue

                common_min=max(m['xmin'] for m in available)
                common_max=min(m['xmax'] for m in available)

                if common_max<=common_min:
                    continue

                common_flow=np.linspace(common_min,common_max,points)

                temp={'Speed':[speed]*points,'Flow':common_flow}

                for p in stage_models:
                    if speed in stage_models[p]:
                        temp[p]=predict_model(stage_models[p][speed],common_flow)

                export_rows.append(pd.DataFrame(temp))

            if export_rows:
                final_df=pd.concat(export_rows,ignore_index=True)
                final_df.to_excel(writer,sheet_name=stage[:31],index=False)

            overview.append({
                'Stage':stage,
                'Parameters':','.join(stage_parameters),
                'Blocks Found':len(blocks)
            })

        pd.DataFrame(r2_rows,columns=['Stage','Speed','Parameter','Method','R2']).to_excel(writer,sheet_name='Summary_R2',index=False)
        pd.DataFrame(overview).to_excel(writer,sheet_name='Workbook_Overview',index=False)

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
