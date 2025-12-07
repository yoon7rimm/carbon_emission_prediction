# carbon_emission_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

# ---------------------------
# LANGUAGE DICTIONARY
# ---------------------------
TEXT = {
    "title": {"en": "Prediction of CO₂ Emission in Architectural Construction Using ML Models",
              "kr": "건축 시공 CO₂ 배출량 예측 (기계학습 모델)"},
    "sidebar_title": {"en": "Data & Settings", "kr": "데이터 및 설정"},
    "sidebar_desc": {"en": "Choose dataset source, language, and options", "kr": "데이터 소스, 언어 및 옵션 선택"},
    "lang_label": {"en": "Language", "kr": "언어"},
    "data_option_label": {"en": "Dataset source", "kr": "데이터 소스"},
    "use_sample": {"en": "Use sample_data.csv (project)", "kr": "sample_data.csv 사용 (프로젝트)"},
    "upload_file": {"en": "Upload my own CSV", "kr": "내 CSV 업로드"},
    "upload_prompt": {"en": "Upload CSV file", "kr": "CSV 파일 업로드"},
    "sample_missing": {"en": "sample_data.csv not found. Upload or place sample_data.csv next to script.",
                       "kr": "sample_data.csv가 없습니다. CSV 업로드 또는 스크립트 옆에 두세요."},
    "tabs": {"home": {"en": "🏠 Home", "kr": "🏠 홈"},
             "dataset": {"en": "📄 Dataset", "kr": "📄 데이터셋"},
             "train": {"en": "⚙️ Train Model", "kr": "⚙️ 모델 학습"},
             "predict": {"en": "🔮 Prediction", "kr": "🔮 예측"},
             "feature": {"en": "📈 Feature Importance", "kr": "📈 변수 중요도"},
             "shap3d": {"en": "🧠 SHAP & 3D", "kr": "🧠 SHAP & 3D"}},
    "no_data": {"en": "No dataset loaded. Use sidebar to load sample or upload CSV.",
                "kr": "데이터셋 없음. 사이드바에서 sample 사용 또는 CSV 업로드"},
    "recommended_features": {"en": "Recommended numeric features: cement_ton, steel_ton, sand_ton, concrete_m3, diesel_liter, electricity_kwh, equipment_hours, project_area_m2, duration_months. Target: CO2_ton",
                             "kr": "권장 숫자형 변수: cement_ton, steel_ton, sand_ton, concrete_m3, diesel_liter, electricity_kwh, equipment_hours, project_area_m2, duration_months. 타깃: CO2_ton"},
    "train_button": {"en": "Train RandomForest Model", "kr": "랜덤포레스트 학습"},
    "predict_single": {"en": "Predict Single", "kr": "단일 예측"},
    "predict_bulk": {"en": "Run Bulk Prediction on Loaded Data", "kr": "전체 데이터 예측"},
    "download_single": {"en": "⬇️ Download single prediction (CSV)", "kr": "⬇️ 단일 예측 다운로드"},
    "download_bulk": {"en": "⬇️ Download bulk predictions (CSV)", "kr": "⬇️ 전체 예측 다운로드"},
    "need_three_numeric": {"en": "Need at least 3 numeric features for 3D scatter.", "kr": "3D 산점도 위해 숫자형 변수 3개 필요"},
    "install_shap": {"en": "Install SHAP to enable explainability (pip install shap).", "kr": "SHAP 설치 필요 (pip install shap)"},
    "model_trained": {"en": "Model trained and saved to session.", "kr": "모델 학습 완료, 세션에 저장됨"}
}

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title=TEXT["title"]["en"], page_icon="🌱", layout="wide")
st.markdown("""
<style>
    .main { background-color: #f7fbfc; }
    .stSidebar { background-color: #0f1724; color: white; }
    .stTabs [role="tab"] { font-size: 15px; padding:10px 12px; font-weight:600; }
    .stTabs [role="tab"][aria-selected="true"] { background:#0ea5a4; color:white !important; border-radius:8px; }
    h1,h2,h3 { color: #0f1724; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title(TEXT["sidebar_title"]["en"])
st.sidebar.write(TEXT["sidebar_desc"]["en"])
st.sidebar.markdown("---")

lang_choice = st.sidebar.selectbox(TEXT["lang_label"]["en"], ("English","한국어"))
L = "en" if lang_choice=="English" else "kr"

st.sidebar.markdown("---")
st.sidebar.write(f"**{TEXT['data_option_label'][L]}**")
data_option = st.sidebar.radio("", (TEXT["use_sample"][L], TEXT["upload_file"][L]))

df = None
uploaded_file = None

if data_option==TEXT["use_sample"][L]:
    sample_path = "sample_data.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.sidebar.success(f"Loaded {sample_path}")
    else:
        st.sidebar.error(TEXT["sample_missing"][L])
else:
    uploaded_file = st.sidebar.file_uploader(TEXT["upload_prompt"][L], type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Uploaded file loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded file: {e}")

st.sidebar.markdown("---")
st.sidebar.info(TEXT["recommended_features"][L])

# ---------------------------
# TABS
# ---------------------------
tab_home, tab_data, tab_train, tab_predict, tab_featimp, tab_shap3d = st.tabs(
    [TEXT["tabs"]["home"][L], TEXT["tabs"]["dataset"][L], TEXT["tabs"]["train"][L],
     TEXT["tabs"]["predict"][L], TEXT["tabs"]["feature"][L], TEXT["tabs"]["shap3d"][L]]
)

# ---------------------------
# HOME TAB
# ---------------------------
with tab_home:
    st.title(TEXT["title"][L])
    st.markdown(TEXT["recommended_features"][L])
    if df is not None:
        st.subheader("Dataset preview")
        st.dataframe(df.head(5), use_container_width=True)
    else:
        st.info("No dataset loaded yet. Use the sidebar to load sample_data.csv or upload your own CSV.")

# ---------------------------
# DATA TAB
# ---------------------------
with tab_data:
    st.header(TEXT["tabs"]["dataset"][L])
    if df is None:
        st.warning(TEXT["no_data"][L])
    else:
        st.subheader("Preview (first 10 rows)" if L=="en" else "미리보기 (첫 10행)")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Summary statistics" if L=="en" else "기본 통계")
        st.dataframe(df.describe().T, use_container_width=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"{('Detected numeric columns:' if L=='en' else '감지된 숫자형 열:')} {numeric_cols}")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                 title=("Correlation matrix" if L=="en" else "상관 행렬"))
            st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------
# TRAIN TAB
# ---------------------------
with tab_train:
    st.header(TEXT["tabs"]["train"][L])
    if df is None:
        st.warning(TEXT["no_data"][L])
    else:
        cols = df.columns.tolist()
        target_default = "CO2_ton" if "CO2_ton" in cols else cols[-1]
        target_col = st.selectbox("Select target column (y)" if L=="en" else "타깃 컬럼 선택 (y)",
                                  options=cols, index=cols.index(target_default))
        possible_features = [c for c in cols if c!=target_col]
        numeric_features = df[possible_features].select_dtypes(include=[np.number]).columns.tolist()
        st.markdown(("**Select features (X)** — numeric columns are recommended" if L=="en" else "**특성 선택 (X)** — 숫자형 권장"))
        selected_features = st.multiselect(("Features to use" if L=="en" else "사용할 특성 선택"),
                                           options=numeric_features, default=numeric_features[:9])
        if selected_features:
            test_size = st.slider(("Test set fraction" if L=="en" else "테스트 비율"),0.05,0.5,0.2,0.05)
            trees = st.slider(("Random Forest trees" if L=="en" else "랜덤포레스트 트리 수"),50,1000,300,50)
            random_state = st.number_input(("Random seed" if L=="en" else "랜덤 시드"), value=42, step=1)
            if st.button(TEXT["train_button"][L]):
                X = df[selected_features].copy()
                y = df[target_col].copy()
                data_for_model = pd.concat([X, y], axis=1).dropna()
                X = data_for_model[selected_features]
                y = data_for_model[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))
                model = RandomForestRegressor(n_estimators=int(trees), random_state=int(random_state), n_jobs=-1)
                with st.spinner(("Training model..." if L=="en" else "모델 학습 중...")):
                    model.fit(X_train, y_train)
                st.session_state["model"]=model
                st.session_state["features"]=selected_features
                st.session_state["target"]=target_col
                y_pred=model.predict(X_test)
                rmse=np.sqrt(mean_squared_error(y_test,y_pred))
                r2=r2_score(y_test,y_pred)
                st.success(TEXT["model_trained"][L])
                col1,col2=st.columns(2)
                col1.metric("RMSE",f"{rmse:.4f}")
                col2.metric("R²",f"{r2:.4f}")
                fig_scatter=px.scatter(x=y_test,y=y_pred,
                                       labels={"x":"Actual" if L=="en" else "실제","y":"Predicted" if L=="en" else "예측"},
                                       title="Actual vs Predicted" if L=="en" else "실제 vs 예측",template="plotly_white")
                st.plotly_chart(fig_scatter,use_container_width=True)

# ---------------------------
# PREDICTION TAB
# ---------------------------
with tab_predict:
    st.header("🔮 Prediction")
    if "model" not in st.session_state:
        st.warning("Train a model first in Train tab.")
    else:
        model=st.session_state["model"]
        features=st.session_state["features"]
        target_col=st.session_state["target"]

        # Single prediction
        st.subheader("Single prediction (manual input)" if L == "en" else "단일 예측 (수동 입력)")

        if df is None:
            st.warning("Load a dataset first to enable manual predictions / 데이터셋을 먼저 불러오세요.")
        else:
            input_vals = {}
            cols_per_row = 3
            cols_ui = st.columns(cols_per_row)

            for i, f in enumerate(features):
                if df is not None and f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                    default = float(df[f].mean())
                else:
                    default = 0.0
                input_vals[f] = cols_ui[i % cols_per_row].number_input(f"{f}", value=default, format="%.4f")

            if st.button(TEXT["predict_single"][L]):
                input_df = pd.DataFrame([input_vals])
                pred = model.predict(input_df)[0]
                st.success(f"🌱 Predicted {target_col}: **{pred:.4f}**")

                out_df = input_df.copy()
                out_df[f"Predicted_{target_col}"] = pred
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(TEXT["download_single"][L], csv_bytes, file_name="single_prediction.csv",
                                   mime="text/csv")

            st.markdown("---")

        # Bulk prediction
        st.subheader("Bulk prediction (full dataset)" if L=="en" else "전체 데이터 예측")
        if st.button(TEXT["predict_bulk"][L]):
            if df is None:
                st.error(TEXT["no_data"][L])
            elif "model" not in st.session_state:
                st.warning("Train a model first / 모델을 먼저 학습하세요.")
            else:
                missing_cols=[c for c in features if c not in df.columns]
                if missing_cols:
                    st.error(f"Missing feature columns: {missing_cols}")
                else:
                    df_copy=df.copy().dropna(subset=features)
                    if df_copy.empty:
                        st.warning("No valid rows in dataset for prediction after dropping NA values.")
                    else:
                        preds=model.predict(df_copy[features])
                        df_copy[f"Predicted_{target_col}"]=preds
                        st.dataframe(df_copy.head(50),use_container_width=True)
                        buf=BytesIO()
                        df_copy.to_csv(buf,index=False)
                        buf.seek(0)
                        st.download_button(TEXT["download_bulk"][L],buf,file_name="bulk_predictions.csv",mime="text/csv")

# ---------------------------
# FEATURE IMPORTANCE TAB (3D colored bar)
# ---------------------------
with tab_featimp:
    st.header("📈 Feature Importance")
    if "model" not in st.session_state:
        st.warning("Train a model first.")
    else:
        model=st.session_state["model"]
        features=st.session_state["features"]
        fi=model.feature_importances_
        fi_df=pd.DataFrame({"feature":features,"importance":fi}).sort_values("importance",ascending=True)
        fig_fi_3d=go.Figure(data=[go.Bar(x=fi_df["importance"],y=fi_df["feature"],orientation='h',
                                         marker=dict(color=fi_df["importance"],colorscale='Viridis',showscale=True))])
        fig_fi_3d.update_layout(title="Feature Importance (3D Color Bar)",
                                xaxis_title="Importance",yaxis_title="Feature",
                                template="plotly_white",height=500)
        st.plotly_chart(fig_fi_3d,use_container_width=True)

# ---------------------------
# SHAP & 3D Tab
# ---------------------------
with tab_shap3d:
    st.header("🧠 SHAP & 3D")
    if "model" not in st.session_state:
        st.warning("Train a model first.")
    else:
        model=st.session_state["model"]
        features=st.session_state["features"]
        if df is not None:
            numeric_cols=df[features].select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols=[]

        # 3D scatter
        if len(numeric_cols)>=3:
            st.subheader("3D Feature Scatter")
            x_col=st.selectbox("X axis",numeric_cols,index=0)
            y_col=st.selectbox("Y axis",numeric_cols,index=1)
            z_col=st.selectbox("Z axis",numeric_cols,index=2)
            fig3d=px.scatter_3d(df.dropna(subset=[x_col,y_col,z_col]),x=x_col,y=y_col,z=z_col,
                                color=features[0] if features else None,opacity=0.8,size_max=6,
                                title=f"3D: {x_col}/{y_col}/{z_col}",template="plotly_white")
            st.plotly_chart(fig3d,use_container_width=True)
        else:
            st.info(TEXT["need_three_numeric"][L])

        # SHAP
        st.subheader("SHAP Explainability")
        if shap_available and df is not None:
            try:
                sample_df=df[features].dropna().sample(min(200,len(df)),random_state=42)
                explainer=shap.TreeExplainer(model)
                shap_values=explainer.shap_values(sample_df)
                st.write("SHAP summary (dot)")
                plt.figure(figsize=(5,3))
                shap.summary_plot(shap_values,sample_df,show=False)
                st.pyplot(plt.gcf())
                plt.close()
                st.write("SHAP summary (bar)")
                plt.figure(figsize=(5,3))
                shap.summary_plot(shap_values,sample_df,plot_type="bar",show=False)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"SHAP plotting failed: {e}")
        else:
            st.info(TEXT["install_shap"][L])

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("App: Prediction of CO₂ Emission in Architectural Construction Using ML Models — built with Streamlit")

# Run in terminal (copy and paste)
# pip install streamlit pandas numpy scikit-learn plotly matplotlib shap
# streamlit run carbon_emission_prediction.py
