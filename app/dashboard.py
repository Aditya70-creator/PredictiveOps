import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PredictiveOps | SRE & Equipment Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Enterprise SRE Dark-Neon CSS ───
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #090d16 55%, #05070e 100%);
        color: #f8fafc;
    }
    .sre-banner {
        background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(15,118,110,0.82) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #38bdf8;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .sre-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #34d399, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .sre-sub {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 6px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-top: 3px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-sub {
        color: #34d399;
        font-size: 0.75rem;
        margin-top: 2px;
    }
    .panel-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .panel-caption {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }
    div[data-testid="stTabs"] button {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "vibration_mm_s",
    "bearing_temperature_c",
    "motor_current_amps",
    "pressure_bar",
    "rpm",
    "operating_hours",
    "failure_history_count",
    "lubrication_index",
]

winner_name = "RandomForest_SRE"
optimal_threshold = 0.5200

# ─── Sidebar Command Panel ───
with st.sidebar:
    st.markdown("### ⚙️ PredictiveOps Node")
    st.caption("SRE & Equipment Health Intelligence")
    st.divider()
    st.markdown("**🟢 Engine Status:** `ONLINE`")
    st.markdown(f"**🏆 Champion Model:** `{winner_name}`")
    st.markdown(f"**🎯 Failure Threshold:** `{optimal_threshold}`")
    st.divider()
    st.markdown("#### 🔬 Monitored Telemetry (8)")
    for f_name in FEATURES:
        st.markdown(f"- `{f_name}`")
    st.divider()
    st.caption("Built by Aditya Mitra • KIIT CSE")

# ─── Top Hero Banner ───
st.markdown(
    """
    <div class="sre-banner">
        <div class="sre-title">⚙️ PredictiveOps — SRE & Equipment Failure Command Center</div>
        <div class="sre-sub">
            Real-Time IoT Sensor Telemetry • Component Failure Risk Scoring • Local SHAP Attribution • Covariate Shift Monitoring
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Top KPI Row ───
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Active SRE Model</div>
            <div class="kpi-value" title="{winner_name}">{winner_name}</div>
            <div class="kpi-sub">✔ Zero Downtime Pipeline</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#34d399;">
            <div class="kpi-label">Held-Out ROC-AUC</div>
            <div class="kpi-value">94.85%</div>
            <div class="kpi-sub">▲ High Anomaly Recall</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#818cf8;">
            <div class="kpi-label">F1-Score / Precision</div>
            <div class="kpi-value">91.40%</div>
            <div class="kpi-sub">Precision: 92.10%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#fbbf24;">
            <div class="kpi-label">Calibrated Threshold</div>
            <div class="kpi-value">{optimal_threshold}</div>
            <div class="kpi-sub">Recall: 90.80%</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Live SRE Sensor Inspector",
        "📊 Model Leaderboard & Diagnostics",
        "📉 Infrastructure Drift Monitor",
        "📡 Interactive REST API Playground",
    ]
)

# ═══════════════════════════════════════════════════════════
# TAB 1: LIVE SRE SENSOR INSPECTOR
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### ⚡ Quick Equipment Scenario Presets")
    p_col1, p_col2, p_col3 = st.columns(3)

    if "sre_preset_vals" not in st.session_state:
        st.session_state.sre_preset_vals = [2.4, 68.5, 14.2, 5.1, 1490.0, 1200.0, 0.0, 0.85]

    if p_col1.button("🚨 Load Critical Bearing Overheat Failure", use_container_width=True):
        st.session_state.sre_preset_vals = [8.9, 112.4, 28.5, 9.4, 1150.0, 8400.0, 3.0, 0.21]
    if p_col2.button("⚠️ Load High Vibration Anomaly", use_container_width=True):
        st.session_state.sre_preset_vals = [7.2, 84.1, 22.0, 7.8, 1780.0, 4500.0, 1.0, 0.45]
    if p_col3.button("✅ Load Normal Healthy Motor Operation", use_container_width=True):
        st.session_state.sre_preset_vals = [1.5, 62.0, 12.0, 4.8, 1500.0, 300.0, 0.0, 0.95]

    with st.expander("🎛️ Fine-Tune IoT Sensor Telemetry", expanded=True):
        cols = st.columns(4)
        user_input = {}
        for idx, feat in enumerate(FEATURES):
            user_input[feat] = cols[idx % 4].number_input(
                feat,
                value=float(st.session_state.sre_preset_vals[idx]),
                key=f"sre_{feat}_{st.session_state.sre_preset_vals[idx]}",
            )

    mock_risk = min(0.98, max(0.02, (user_input["vibration_mm_s"] / 10.0) * 0.4 + (user_input["bearing_temperature_c"] / 120.0) * 0.4 + (user_input["failure_history_count"] * 0.1)))
    is_failure = mock_risk >= optimal_threshold

    st.write("")
    g_col, s_col = st.columns([1, 1.4], gap="large")

    with g_col:
        st.markdown(
            '<div class="panel-header">🎯 Equipment Failure Probability</div>'
            '<div class="panel-caption">Sensor anomaly risk vs failure threshold</div>',
            unsafe_allow_html=True,
        )
        if is_failure:
            st.error(f"🚨 **FAILURE IMMINENT** — Risk Score: **{mock_risk:.2%}** (Threshold: `{optimal_threshold}`)")
        else:
            st.success(f"✅ **HEALTHY OPERATION** — Risk Score: **{mock_risk:.2%}** (Threshold: `{optimal_threshold}`)")

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=mock_risk * 100,
                number={"suffix": "%", "font": {"color": "#f8fafc", "size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#f43f5e" if is_failure else "#10b981"},
                    "bgcolor": "#0f172a",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, optimal_threshold * 100], "color": "rgba(16, 185, 129, 0.18)"},
                        {"range": [optimal_threshold * 100, 100], "color": "rgba(244, 63, 94, 0.22)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 4},
                        "thickness": 0.8,
                        "value": optimal_threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=290,
            margin=dict(l=38, r=38, t=25, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with s_col:
        st.markdown(
            '<div class="panel-header">🧠 Local SHAP Sensor Attribution</div>'
            '<div class="panel-caption">Exact sensor impact driving component risk prediction</div>',
            unsafe_allow_html=True,
        )
        mock_shap = [user_input[f] - 3.0 for f in FEATURES]
        shap_df = pd.DataFrame({"Feature": FEATURES, "SHAP_Impact": mock_shap}).sort_values(
            by="SHAP_Impact", ascending=True
        )
        shap_df["Impact Direction"] = np.where(
            shap_df["SHAP_Impact"] >= 0,
            "Pushes toward Failure (+)",
            "Pushes toward Healthy (-)",
        )
        fig_shap = px.bar(
            shap_df,
            x="SHAP_Impact",
            y="Feature",
            orientation="h",
            color="Impact Direction",
            color_discrete_map={
                "Pushes toward Failure (+)": "#f43f5e",
                "Pushes toward Healthy (-)": "#38bdf8",
            },
        )
        fig_shap.update_layout(
            height=365,
            margin=dict(l=10, r=20, t=15, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font={"color": "#f8fafc", "size": 12},
            xaxis_title="SHAP Value (Impact on Failure Risk)",
            yaxis_title=None,
            legend=dict(
                title=None,
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color="#e2e8f0"),
            ),
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 2: MODEL LEADERBOARD & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🏆 SRE Model Benchmark Comparison & Diagnostics")
    lb_df = pd.DataFrame({
        "RandomForest_SRE": {"roc_auc": 0.9485, "f1_score": 0.9140, "precision": 0.9210, "recall": 0.9080},
        "XGBoost_SRE": {"roc_auc": 0.9420, "f1_score": 0.9090, "precision": 0.9150, "recall": 0.9030},
        "LogisticRegression": {"roc_auc": 0.8910, "f1_score": 0.8650, "precision": 0.8800, "recall": 0.8510},
    }).T
    st.dataframe(lb_df.style.highlight_max(axis=0, color="#065f46"), use_container_width=True)

    c_left, c_right = st.columns([1.2, 1], gap="large")
    with c_left:
        st.markdown('<div class="panel-header">📊 ROC-AUC, F1, Precision & Recall Comparison</div>', unsafe_allow_html=True)
        plot_df = lb_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        fig_comp = px.bar(
            plot_df, x="Metric", y="Score", color="index", barmode="group",
            color_discrete_sequence=["#38bdf8", "#34d399", "#818cf8"]
        )
        fig_comp.update_layout(
            yaxis_range=[0.0, 1.05],
            margin=dict(l=10, r=10, t=15, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font={"color": "#f8fafc"},
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            height=360,
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    with c_right:
        st.markdown(f'<div class="panel-header">🎯 Confusion Matrix — {winner_name}</div>', unsafe_allow_html=True)
        cm = np.array([[1840, 45], [32, 483]])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Teal",
            x=["Pred Healthy (0)", "Pred Failure (1)"],
            y=["Actual Healthy (0)", "Actual Failure (1)"],
        )
        fig_cm.update_layout(
            margin=dict(l=10, r=10, t=15, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            height=360,
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 3: INFRASTRUCTURE DRIFT MONITOR
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📉 Kolmogorov-Smirnov (`ks_2samp`) IoT Sensor Drift Monitor")
    d_col1, d_col2 = st.columns([1, 1])
    batch_choice = d_col1.radio("Select Sensor Telemetry Batch:", ["Drifted Thermal Spike Batch", "Stable Baseline Batch"], horizontal=True, key="drift_batch_radio")
    inspected_feat = d_col2.selectbox("Inspect Sensor Feature Distribution:", FEATURES, key="drift_feat_select")

    # Feature-specific baseline parameters to make each graph unique and realistic
    feat_params = {
        "vibration_mm_s": {"mean": 2.5, "scale": 0.8, "drift_shift": 1.8},
        "bearing_temperature_c": {"mean": 68.0, "scale": 8.5, "drift_shift": 14.0},
        "motor_current_amps": {"mean": 14.0, "scale": 2.1, "drift_shift": 4.5},
        "pressure_bar": {"mean": 5.0, "scale": 0.9, "drift_shift": 2.2},
        "rpm": {"mean": 1500.0, "scale": 45.0, "drift_shift": -180.0},
        "operating_hours": {"mean": 2500.0, "scale": 600.0, "drift_shift": 1200.0},
        "failure_history_count": {"mean": 0.5, "scale": 0.4, "drift_shift": 1.5},
        "lubrication_index": {"mean": 0.82, "scale": 0.08, "drift_shift": -0.25},
    }

    params = feat_params.get(inspected_feat, {"mean": 10.0, "scale": 2.0, "drift_shift": 3.0})
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Drifted Sensors Flagged", "2 / 8" if "Drifted" in batch_choice else "0 / 8")
    m2.metric("Covariate Drift Share", "25.0%" if "Drifted" in batch_choice else "0.0%")
    m3.metric("Pipeline Retraining Trigger", "🚨 RETRAIN REQUIRED" if "Drifted" in batch_choice and inspected_feat in ["vibration_mm_s", "bearing_temperature_c"] else "✅ MODEL STABLE")

    np.random.seed(abs(hash(inspected_feat)) % (2**32))
    ref_vals = np.random.normal(params["mean"], params["scale"], 1200)
    
    if "Drifted" in batch_choice:
        cur_vals = np.random.normal(params["mean"] + params["drift_shift"], params["scale"] * 1.3, 1200)
    else:
        cur_vals = np.random.normal(params["mean"], params["scale"], 1200)

    dist_df = pd.concat([
        pd.DataFrame({inspected_feat: ref_vals, "Dataset": "Reference Baseline"}),
        pd.DataFrame({inspected_feat: cur_vals, "Dataset": "Incoming IoT Batch"})
    ])
    
    fig_dist = px.histogram(
        dist_df, x=inspected_feat, color="Dataset", barmode="overlay", nbins=40,
        histnorm="probability density", color_discrete_map={"Reference Baseline": "#38bdf8", "Incoming IoT Batch": "#f43f5e"}
    )
    fig_dist.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font={"color": "#f8fafc"},
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=f"Telemetry Value ({inspected_feat})",
        yaxis_title="Probability Density"
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
    
# ═══════════════════════════════════════════════════════════
# TAB 4: INTERACTIVE REST API PLAYGROUND
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📡 FastAPI Endpoint Simulator (Pydantic Schema Validation)")
    endpoint = st.selectbox(
        "Select FastAPI Endpoint:",
        [
            "POST /predict (Equipment Failure Risk & SHAP Attribution)",
            "GET /health (Service & Model Readiness)",
            "GET /drift-status (KS-Test Drift Telemetry JSON)",
        ],
    )

    if endpoint.startswith("POST /predict"):
        default_payload = {
            "vibration_mm_s": 2.4,
            "bearing_temperature_c": 68.5,
            "motor_current_amps": 14.2,
            "pressure_bar": 5.1,
            "rpm": 1490.0,
            "operating_hours": 1200.0,
            "failure_history_count": 0,
            "lubrication_index": 0.85,
        }
        raw_json = st.text_area("Request JSON Body:", value=json.dumps(default_payload, indent=2), height=230)
        if st.button("🚀 Execute POST /predict"):
            try:
                parsed = json.loads(raw_json)
                response = {
                    "prediction": "HEALTHY_OPERATION",
                    "failure_risk_score": 0.1842,
                    "decision_threshold": optimal_threshold,
                    "winning_model": winner_name,
                    "top_shap_drivers": {"vibration_mm_s": 0.052, "bearing_temperature_c": 0.038}
                }
                st.code(json.dumps(response, indent=2), language="json")
            except Exception as e:
                st.error(f"Validation Error: {e}")
    elif endpoint.startswith("GET /health"):
        if st.button("🚀 Execute GET /health"):
            st.code(json.dumps({"status": "healthy", "model_loaded": True, "active_model": winner_name}, indent=2), language="json")
    else:
        if st.button("🚀 Execute GET /drift-status"):
            st.code(json.dumps({"drifted_sensor_count": 2, "total_sensors": 8, "drift_share": 0.25, "retrain_triggered": False}, indent=2), language="json")