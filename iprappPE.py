"""
IPR Analyzer — Inflow Performance Relationship curve builder
Methods: Simple IPR (constant J), Vogel's Method, Fetkovich's Method
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="IPR Analyzer", page_icon="🛢️", layout="wide")

# =====================================================================================
#  DEMO DATA  (taken from the reference IPR workbook so results are reproducible)
# =====================================================================================
DEMO_IPR = dict(
    label="Default example (reference workbook, Sheet 1)",
    Pr=1300.0, Pwf=900.0, Qo=110.0,
    table=pd.DataFrame({"Pwf": [1300, 1170, 1040, 910, 780, 650, 520, 390, 260, 130, 0 ], "Qo": [0, 35.75,71.5, 107.25, 143, 178.75, 214.5, 250.25, 286, 321.75, 357.5]}),
)

DEMO_VOGEL_SAT = dict(
    label="Saturated example (Pr ≤ Pb)",
    Pr=2500.0, Pb=2600.0, Pwf=2000.0, Qo=350.0,
    table=pd.DataFrame({"Pwf": [2500, 2250, 2000, 1750, 1500, 1250, 1000, 750, 500, 250, 0], "Qo": [0, 183.536, 350, 499.39, 631.70, 746.95, 845.12, 926.21, 990.24, 1037.19, 1067.07]}),
    Prf=2200.0,
)
DEMO_VOGEL_UNDERSAT = dict(
    label="Undersaturated example (Pr > Pb)",
    Pr=3000.0, Pb=2130.0, Pwf=2500.0, Qo=250.0, 
    table=pd.DataFrame({"Pwf": [3000, 2826, 2652, 2478, 2304, 2130, 1917, 1704, 1491, 1278, 1065, 852, 639, 426, 213, 0 ], "Qo": [0, 87, 174, 261, 348, 435, 536.77, 629.07, 711.9, 785.27, 849.17, 903.6, 948.57, 984.07, 1010.1, 1026.67]}),
    Prf=2600.0,
)

DEMO_FETKOVICH_SAT = dict(
    label="Saturated example (Pr ≤ Pb)",
    Pr=3600.0,
    table=pd.DataFrame({"Pwf": [3170, 2890, 2440, 2150], "Qo": [263, 383, 497, 640]}),
    Prf=2000.0,
)
DEMO_FETKOVICH_UNDERSAT = dict(
    label="Undersaturated example (Pr > Pb)",
    Pr=3600.0,
    table=pd.DataFrame({"Pwf": [3000, 2700, 2400, 2100, 1800, 1500, 1350, 1200, 1050, 900, 750, 600, 450, 300, 150, 0], "Qo": [0.0, 127.48, 254.97, 382.45, 509.93, 637.42, 697.97, 752.15, 799.96, 841.39, 876.45, 905.13, 927.38, 943.38, 952.94, 956.13]}),
    Prf=2000.0,
)

# Real published field case (Well B, Keokuk Pool, Seminole County, Oklahoma, Aug 1935),
# used as a second Fetkovich demo — includes an actual repeat test 8 months later so the
# future-IPR prediction can be checked against real data.
FIELD_CASE_3 = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935) — Multi-rate test data",
    Pr=1714.0,
    table=pd.DataFrame({
        "Pwf": [1714, 1583, 1443, 1272, 1196, 982],
        "Qo": [0, 280, 508, 780, 1125, 1335],
    }),
    Prf=1605.0,
    future_actual_table=pd.DataFrame({
        "Pwf": [1605, 1381, 1231, 1120],
        "Qo": [0, 420, 720, 850],
    }),
)

# The published "New Correlation" full IPR curves for the same well (present and future),
# given as complete Pwf/Qo curves rather than raw multi-rate test points.
FIELD_CASE_3_PRESENT_CURVE = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935) — present IPR curve (full curve)",
    Pr=1714.0,
    table=pd.DataFrame({
        "Pwf": [1714, 1583, 1443, 1272, 1196, 982, 800, 600, 400, 200, 0],
        "Qo": [0, 311.3553721, 624.0096558, 977.7314268, 1125, 1506.812135,
               1793.363642, 2067.796199, 2299.863093, 2489.564324, 2636.899892],
    }),
    Prf=1400.0,
)
FIELD_CASE_3_FUTURE_CURVE = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935) — future IPR curve (full curve)",
    Pr=1605.0,
    table=pd.DataFrame({
        "Pwf": [1605, 1381, 1231, 1120, 982, 800, 600, 400, 200, 0],
        "Qo": [0, 434.7769194, 701.8747149, 887.1085623, 1102.669899, 1361.996156,
               1614.220934, 1832.153431, 2015.793646, 2165.141578],
    }),
    Prf=1300.0,
)

# Same Well B / Keokuk Pool field data adapted for Simple IPR (constant J) — uses one
# representative stabilized test point (the median of the real multi-rate test) rather
# than the full multi-rate table, since Simple IPR only needs a single (Pwf, Qo) pair.
FIELD_CASE_3_SIMPLE = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935)",
    Pr=1714.0, Pwf=1272.0, Qo=780.0,
    table=pd.DataFrame({
        "Pwf": [1714, 1583, 1443, 1272, 1196, 982],
        "Qo": [0, 280, 508, 780, 1125, 1335],
    }),
)

# Same Well B / Keokuk Pool field data adapted for Vogel's method. The original 1935
# report doesn't state a bubble-point pressure, so two Pb assumptions are offered as
# separate demos — one that keeps the well saturated (Pb above Pr) and one that makes
# it undersaturated (Pb below Pr, and below the 1272 psi test point, so the undersaturated
# demo actually exercises the combined linear+Vogel branch of the equation). Pb is a live
# input in the Vogel section either way, so adjust it there if a different assumption is
# preferred.
FIELD_CASE_3_VOGEL_SAT = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935) (assumed Pb = 1900 psi, saturated)",
    Pr=1714.0, Pb=1900.0, Pwf=1272.0, Qo=780.0, Prf=1605.0,
    table=pd.DataFrame({
        "Pwf": [1714, 1583, 1443, 1272, 1196, 982],
        "Qo": [0, 280, 508, 780, 1125, 1335],
    }),
)
FIELD_CASE_3_VOGEL_UNDERSAT = dict(
    label="Field Case 3 — Well B, Keokuk pool, Oklahoma (1935) (assumed Pb = 1500 psi, undersaturated)",
    Pr=1714.0, Pb=1500.0, Pwf=1272.0, Qo=780.0, Prf=1605.0,
    table=pd.DataFrame({
        "Pwf": [1714, 1583, 1443, 1272, 1196, 982],
        "Qo": [0, 280, 508, 780, 1125, 1335],
    }),
)



METHOD_LABELS = {
    "1": "1. Simple IPR (constant J)",
    "2": "2. Vogel's Method",
    "3": "3. Fetkovich's Method",
}

# One flat, ordered catalog of every demo dataset — shown as a single list so the user can
# pick exactly which dataset they want, regardless of method. Picking one also determines
# which method section renders below.
DEMO_CATALOG = [
    # -------------------------------------------------------------------------
    # Simple IPR datasets
    # -------------------------------------------------------------------------
    (
        "1",
        "Simple IPR-demo",
        DEMO_IPR
    ),
    (
        "1",
        "Field Case 3 — Well B, Keokuk Pool (1935) — Simple IPR",
        FIELD_CASE_3_SIMPLE
    ),

    # -------------------------------------------------------------------------
    # Vogel datasets
    # -------------------------------------------------------------------------
    (
        "2",
        "Vogel — saturated example (Pr ≤ Pb)",
        DEMO_VOGEL_SAT
    ),
    (
            "2",
            "Vogel — undersaturated example (Pr > Pb)",
            DEMO_VOGEL_UNDERSAT
    ),
    (
        "2",
        "Field Case 3 — Well B, Keokuk Pool (1935) — Vogel (saturated)",
        FIELD_CASE_3_VOGEL_SAT
    ),
    (
        "2",
        "Field Case 3 — Well B, Keokuk Pool (1935) — Vogel (undersaturated)",
        FIELD_CASE_3_VOGEL_UNDERSAT
    ),

    # -------------------------------------------------------------------------
    # Fetkovich datasets
    # -------------------------------------------------------------------------
    (
        "3",
        "Field Case 3 — Well B, Keokuk Pool (1935) — multi-rate test data",
        FIELD_CASE_3
    ),
    (
        "3",
        "Fetkovich — saturated synthetic example (reference workbook)",
        DEMO_FETKOVICH_SAT
    ),
    (
        "3",
        "Fetkovich — undersaturated synthetic example (reference workbook)",
        DEMO_FETKOVICH_UNDERSAT
    ),
    (
        "3",
        "Field Case 3 — Well B, Keokuk Pool (1935) — present IPR curve",
        FIELD_CASE_3_PRESENT_CURVE
    ),

    (
        "3",
        "Field Case 3 — Well B, Keokuk Pool (1935) — future IPR curve",
        FIELD_CASE_3_FUTURE_CURVE
    ),
]
DEMO_LABELS = [label for _, label, _ in DEMO_CATALOG]
DEMO_LOOKUP = {label: (method_key, d) for method_key, label, d in DEMO_CATALOG}
DEMO_LABELS_BY_METHOD = {
    mk: [label for m, label, _ in DEMO_CATALOG if m == mk] for mk in ("1", "2", "3")
}

# =====================================================================================
#  CALCULATION HELPERS  (formulas verified against the reference workbook)
# =====================================================================================

def simple_ipr(Pr, Pwf_s, Qo_s):
    J = Qo_s / (Pr - Pwf_s)
    AOF = J * Pr
    pwf_curve = np.linspace(Pr, 0, 25)
    qo_curve = J * (Pr - pwf_curve)
    return J, AOF, pwf_curve, qo_curve


def vogel_saturated(Pr, Pwf_s, Qo_s, Prf=None):
    J_const = Qo_s / (Pr - Pwf_s)
    Qomax = Qo_s / (1 - 0.2 * (Pwf_s / Pr) - 0.8 * (Pwf_s / Pr) ** 2)

    pwf_curve = np.linspace(Pr, 0, 25)
    qo_vogel = Qomax * (1 - 0.2 * (pwf_curve / Pr) - 0.8 * (pwf_curve / Pr) ** 2)
    qo_linear = J_const * (Pr - pwf_curve)

    out = dict(J=J_const, Qomax=Qomax, pwf=pwf_curve, qo_vogel=qo_vogel, qo_linear=qo_linear)

    if Prf is not None:
        Qomax_f = Qomax * (Prf / Pr) * (0.2 + 0.8 * (Prf / Pr))
        pwf_f = np.linspace(Prf, 0, 25)
        qo_f = Qomax_f * (1 - 0.2 * (pwf_f / Prf) - 0.8 * (pwf_f / Prf) ** 2)
        out.update(Qomax_f=Qomax_f, pwf_f=pwf_f, qo_f=qo_f)

    return out


def vogel_undersaturated(Pr, Pb, Pwf_s, Qo_s, Prf=None):
    if Pwf_s >= Pb:
        case = 1
        J = Qo_s / (Pr - Pwf_s)
    else:
        case = 2
        J = Qo_s / ((Pr - Pb) + (Pb / 1.8) * (1 - 0.2 * (Pwf_s / Pb) - 0.8 * (Pwf_s / Pb) ** 2))

    Qob = J * (Pr - Pb)
    Qomax_vogel_part = J * Pb / 1.8
    Qomax_total = Qob + Qomax_vogel_part

    def curve(J_, Pr_, Pb_, Qob_, Qmaxpart_, n=25):
        pwf_c = np.linspace(Pr_, 0, n)
        qo_c = np.where(
            pwf_c >= Pb_,
            J_ * (Pr_ - pwf_c),
            Qob_ + Qmaxpart_ * (1 - 0.2 * (pwf_c / Pb_) - 0.8 * (pwf_c / Pb_) ** 2),
        )
        return pwf_c, qo_c

    pwf_curve, qo_curve = curve(J, Pr, Pb, Qob, Qomax_vogel_part)

    out = dict(case=case, J=J, Qob=Qob, Qomax_vogel=Qomax_vogel_part, Qomax_total=Qomax_total,
               pwf=pwf_curve, qo=qo_curve)

    if Prf is not None:
        Jf = J * (Prf / Pr) ** 2
        Qob_f = Jf * (Prf - Pb)
        Qomax_vogel_f = Jf * Pb / 1.8
        pwf_f, qo_f = curve(Jf, Prf, Pb, Qob_f, Qomax_vogel_f)
        out.update(Jf=Jf, Qob_f=Qob_f, Qomax_vogel_f=Qomax_vogel_f,
                   Qomax_total_f=Qob_f + Qomax_vogel_f, pwf_f=pwf_f, qo_f=qo_f)

    return out


def fetkovich(Pr, pwf_arr, qo_arr, Prf=None):
    pwf_arr = np.asarray(pwf_arr, dtype=float)
    qo_arr = np.asarray(qo_arr, dtype=float)

    dP2 = Pr ** 2 - pwf_arr ** 2
    logQ = np.log10(qo_arr)
    logD = np.log10(dP2)

    slope, intercept = np.polyfit(logQ, logD, 1)
    n = 1 / slope
    C = 10 ** (-intercept / slope)

    pwf_curve = np.linspace(Pr, 0, 25)
    qo_curve = C * (Pr ** 2 - pwf_curve ** 2) ** n

    table = pd.DataFrame({"Pwf": pwf_arr, "Qo": qo_arr, "Pr^2 - Pwf^2": dP2})
    loglog = pd.DataFrame({"log(Qo)": logQ, "log(Pr^2 - Pwf^2)": logD})

    out = dict(slope=slope, intercept=intercept, n=n, C=C,
               table=table, loglog=loglog, pwf=pwf_curve, qo=qo_curve)

    if Prf is not None:
        Cf = C * (Prf / Pr)
        pwf_f = np.linspace(Prf, 0, 25)
        qo_f = Cf * (Prf ** 2 - pwf_f ** 2) ** n
        out.update(Cf=Cf, pwf_f=pwf_f, qo_f=qo_f)

    return out


# =====================================================================================
#  DATA LOADING HELPERS
# =====================================================================================

def try_load_uploaded(file):
    """Load a user file and try to auto-detect Pwf / Qo columns."""
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df


def guess_column(df, keywords):
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in keywords):
            return col
    return None


def render_formulas(method_key):
    """Render the formula reference for the given method (used in the sidebar)."""
    if method_key == "1":
        st.markdown("**Constant productivity index (straight-line IPR):**")
        st.latex(r"J = \dfrac{Q_o}{P_r - P_{wf}}")
        st.latex(r"\text{AOF} = J \times P_r")
        st.latex(r"Q_o(P_{wf}) = J \,(P_r - P_{wf})")

    elif method_key == "2":
        st.markdown("**Saturated (Pr ≤ Pb)** — Vogel (1968):")
        st.latex(r"(Q_o)_{max} = \dfrac{Q_o}{1 - 0.2\left(\frac{P_{wf}}{P_r}\right) - 0.8\left(\frac{P_{wf}}{P_r}\right)^2}")
        st.latex(r"Q_o(P_{wf}) = (Q_o)_{max}\left[1 - 0.2\left(\frac{P_{wf}}{P_r}\right) - 0.8\left(\frac{P_{wf}}{P_r}\right)^2\right]")
        st.markdown("Future IPR (Standing, 1970):")
        st.latex(r"(Q_o)_{max,f} = (Q_o)_{max,p}\!\left(\dfrac{P_{r,f}}{P_{r,p}}\right)\!\left[0.2 + 0.8\!\left(\dfrac{P_{r,f}}{P_{r,p}}\right)\right]")
        st.markdown("---")
        st.markdown("**Undersaturated (Pr > Pb)** — combined linear + Vogel:")
        st.latex(r"\text{if } P_{wf,test}\!\ge\! P_b:\ J = \dfrac{Q_o}{P_r - P_{wf,test}}")
        st.latex(r"\text{if } P_{wf,test}\!<\! P_b:\ J = \dfrac{Q_o}{(P_r - P_b) + \frac{P_b}{1.8}\left[1 - 0.2\frac{P_{wf,test}}{P_b} - 0.8\left(\frac{P_{wf,test}}{P_b}\right)^2\right]}")
        st.latex(r"Q_{ob} = J(P_r - P_b) \qquad (Q_o)_{max} = Q_{ob} + \dfrac{JP_b}{1.8}")
        st.latex(r"Q_o(P_{wf}) = \begin{cases} J(P_r - P_{wf}) & P_{wf}\ge P_b \\ Q_{ob} + \frac{JP_b}{1.8}\left[1 - 0.2\frac{P_{wf}}{P_b} - 0.8\left(\frac{P_{wf}}{P_b}\right)^2\right] & P_{wf} < P_b \end{cases}")
        st.markdown("Future IPR:")
        st.latex(r"J_f = J_p\left(\dfrac{P_{r,f}}{P_{r,p}}\right)^2")

    else:
        st.markdown("**Fetkovich (1973) empirical deliverability equation:**")
        st.latex(r"Q_o = C\,(P_r^2 - P_{wf}^2)^n")
        st.markdown("Fitted via log-log linear regression of the test data:")
        st.latex(r"\log Q_o = n \log(P_r^2 - P_{wf}^2) + \log C")
        st.markdown("Future IPR — n assumed unchanged, C scaled with pressure:")
        st.latex(r"C_f = C_p\left(\dfrac{P_{r,f}}{P_{r,p}}\right)")


# =====================================================================================
#  SIDEBAR — data source, method, demo dataset, formulas
# =====================================================================================

with st.sidebar:
    st.title("🛢️ IPR Analyzer")

    st.header("1 · Data source")
    data_source = st.radio(
        "How would you like to provide data?",
        ["Use demo data", "Upload my own data"],
    )

    user_df = None
    if data_source == "Upload my own data":
        up = st.file_uploader(
            "Upload CSV or Excel (columns like 'Pwf' and 'Qo'). Mainly used for the "
            "Fetkovich multi-rate table — for Simple IPR / Vogel you can just type in "
            "the single test point in the main panel.",
            type=["csv", "xlsx", "xls"],
        )
        if up is not None:
            try:
                user_df = try_load_uploaded(up)
                st.success(f"Loaded '{up.name}' — {user_df.shape[0]} rows.")
            except Exception as e:
                st.error(f"Could not read that file: {e}")
    else:
        st.caption("Using built-in demo data. Switch to 'Upload my own data' any time.")

    st.header("2 · Method")
    method = st.radio(
        "Select IPR method",
        ["1. IPR", "2. Vogel's", "3. Fetkovich's"],
    )
    method_key = method[0]

    demo_dict = None
    if data_source == "Use demo data":
        st.header("3 · Demo dataset")
        options = DEMO_LABELS_BY_METHOD[method_key]
        if len(options) > 1:
            demo_label = st.selectbox("Choose a demo dataset for this method", options)
        else:
            demo_label = options[0]
            st.caption(f"Only one demo dataset for this method: **{demo_label}**")
        _, demo_dict = DEMO_LOOKUP[demo_label]
    else:
        st.caption("Not used — you're uploading your own data.")

    st.header("📐 Formulas")
    with st.expander("View formulas for the selected method", expanded=False):
        render_formulas(method_key)

    # -------------------------------------------------------------------------
    # 4 · Reservoir / Well Parameters
    # -------------------------------------------------------------------------
    st.header("4 · Reservoir / Well Parameters")

    h = st.number_input(
        "Net pay thickness, h (ft)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    rw = st.number_input(
        "Wellbore radius, rw (ft)",
        min_value=0.0,
        value=0.3,
        step=0.1,
    )

    re = st.number_input(
        "Drainage radius, re (ft)",
        min_value=0.0,
        value=660.0,
        step=10.0,
    )

    uo = st.number_input(
        "Oil viscosity, μo (cp)",
        min_value=0.0,
        value=2.4,
        step=0.1,
    )

    Bo = st.number_input(
        "Oil formation volume factor, Bo (bbl/STB)",
        min_value=0.0,
        value=1.4,
        step=0.1,
    )

    S = st.number_input(
        "Skin factor, S",
        value=-0.5,
        step=0.1,
    )

    K = st.number_input(
        "Permeability, k (md)",
        min_value=0.0,
        value=65.0,
        step=1.0,
    )

    # -------------------------------------------------------------------------
    # 📐 Formulas
    # -------------------------------------------------------------------------
    st.header("📐 Formulas")

    with st.expander(
        "View formulas for the selected method",
        expanded=False
    ):
        render_formulas(method_key)

# =====================================================================================
#  MAIN PANEL
# =====================================================================================

st.caption("Build Inflow Performance Relationship curves using the Simple (constant J), "
           "Vogel's, and Fetkovich's method. Use the sidebar to set the data source, method, "
           "demo dataset (or upload), and to review the formulas.")

st.header("Check the data")
if user_df is not None:
    st.dataframe(user_df, use_container_width=True)
elif data_source == "Upload my own data":
    st.write("No file uploaded yet — use the sidebar to upload a file, or switch to demo data.")
else:
    if method_key == "3":
        if "note" in demo_dict:
            st.caption(demo_dict["note"])
        st.dataframe(demo_dict["table"], use_container_width=True)
        if "future_actual_table" in demo_dict:
            st.caption("This scenario also has a real repeat test taken later — see the "
                       "future-prediction section below to overlay it.")
            with st.expander("Preview the future repeat-test data"):
                st.dataframe(demo_dict["future_actual_table"], use_container_width=True)
    else:
        scalar_rows = {k: v for k, v in demo_dict.items()
                       if k not in ("label", "table", "future_actual_table")}
        st.dataframe(pd.DataFrame([scalar_rows]), use_container_width=True)
        if "table" in demo_dict:
            with st.expander("Preview the full reference curve for this dataset"):
                st.dataframe(demo_dict["table"], use_container_width=True)

st.divider()

# =====================================================================================
#  METHOD 1 — SIMPLE IPR
# =====================================================================================
if method_key == "1":
    st.header("Simple IPR — Constant Productivity Index")
    st.write("This method assumes a straight-line IPR (valid for undersaturated oil "
             "reservoirs above the bubble point). Provide the average reservoir pressure "
             "and one stabilized flow test point.")

    d = demo_dict if demo_dict is not None else dict(Pr=2000.0, Pwf=1500.0, Qo=200.0)

    c1, c2, c3 = st.columns(3)
    Pr = c1.number_input("Average reservoir pressure, Pr (psi)", value=float(d["Pr"]), min_value=0.0)
    Pwf_s = c2.number_input("Stabilized Pwf (psi)", value=float(d["Pwf"]), min_value=0.0, max_value=Pr)
    Qo_s = c3.number_input("Stabilized Qo (STB/day)", value=float(d["Qo"]), min_value=0.0)

    if Pwf_s >= Pr:
        st.warning("Pwf must be less than Pr.")
    else:
        J, AOF, pwf_c, qo_c = simple_ipr(Pr, Pwf_s, Qo_s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qo_c, y=pwf_c, mode="lines", name="IPR curve"))
        fig.add_trace(go.Scatter(x=[Qo_s], y=[Pwf_s], mode="markers", name="Test point",
                                  marker=dict(size=10, color="red")))
        fig.update_layout(title="IPR Curve (Simple / Constant J)",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(rangemode="tozero"), xaxis=dict(rangemode="tozero"),
                           height=520, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results")
        summary = pd.DataFrame({
            "Parameter": ["Reservoir pressure, Pr", "Pwf (stabilized)",
                          "AOF (Absolute Open Flow)", "Productivity Index, J"],
            "Value": [f"{Pr:,.2f} psi", f"{Pwf_s:,.2f} psi",
                      f"{AOF:,.2f} STB/day", f"{J:,.4f} STB/day/psi"],
        })
        st.table(summary)

# =====================================================================================
#  METHOD 2 — VOGEL
# =====================================================================================
elif method_key == "2":
    st.header("Vogel's Method")
    st.write("Provide the average reservoir pressure and bubble-point pressure first — "
             "the app will tell you whether the reservoir is **saturated** (Pr ≤ Pb) or "
             "**undersaturated** (Pr > Pb) and apply the right form of Vogel's equation.")

    d = demo_dict if demo_dict is not None else dict(Pr=3000.0, Pb=2500.0, Pwf=2600.0, Qo=300.0, Prf=2700.0)

    c1, c2 = st.columns(2)
    Pr = c1.number_input("Average reservoir pressure, Pr (psi)", value=float(d["Pr"]), min_value=0.0)
    Pb = c2.number_input("Bubble-point pressure, Pb (psi)", value=float(d["Pb"]), min_value=0.0)

    if Pr <= Pb:
        st.success(f"Since Pr ({Pr:,.0f}) ≤ Pb ({Pb:,.0f}) → **Saturated reservoir**. "
                   "Vogel's equation is applied directly.")
        case = "saturated"
    else:
        st.success(f"Since Pr ({Pr:,.0f}) > Pb ({Pb:,.0f}) → **Undersaturated reservoir**. "
                   "A combined linear + Vogel IPR is used.")
        case = "undersaturated"

    c3, c4 = st.columns(2)
    Pwf_s = c3.number_input("Stabilized Pwf test point (psi) — adjust if needed",
                             value=float(d["Pwf"]), min_value=0.0, max_value=Pr)
    Qo_s = c4.number_input("Stabilized Qo test point (STB/day)", value=float(d["Qo"]), min_value=0.0)

    do_future = st.checkbox("Also predict a future IPR curve at a different reservoir pressure", value=True)
    Prf = None
    if do_future:
        Prf = st.number_input("Future average reservoir pressure, Pr,future (psi)",
                               value=float(d["Prf"]), min_value=0.0)

    if case == "saturated":
        res = vogel_saturated(Pr, Pwf_s, Qo_s, Prf if do_future else None)

        if do_future:
            st.markdown("**Future IPR curve data table** — the exact (Pwf, Qo) points used to draw "
                        "the dashed future curve below, so you can check specific values against the plot.")
            future_curve_df = pd.DataFrame({"Pwf": res["pwf_f"], "Qo": res["qo_f"]})
            st.dataframe(future_curve_df.style.format({"Pwf": "{:.1f}", "Qo": "{:.2f}"}),
                        use_container_width=True)
            st.download_button("Download future IPR curve (CSV)",
                                future_curve_df.to_csv(index=False).encode(),
                                "vogel_saturated_future_ipr.csv", key="vogel_sat_future_dl")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["qo_vogel"], y=res["pwf"], mode="lines", name="Vogel IPR"))
        fig.add_trace(go.Scatter(x=res["qo_linear"], y=res["pwf"], mode="lines",
                                  name="Linear IPR (constant J)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=[Qo_s], y=[Pwf_s], mode="markers", name="Test point",
                                  marker=dict(size=10, color="red")))
        if do_future:
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future Vogel IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
        fig.update_layout(title="Vogel IPR — Saturated Reservoir",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(rangemode="tozero"), xaxis=dict(rangemode="tozero"),
                           height=520, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results")
        summary = pd.DataFrame({
            "Parameter": ["Productivity index, J (linear, for reference)", "Qob (rate at Pb)",
                          "(Qo)max — Vogel part", "(Qo)max — total"],
            "Value": [f"{res['J']:.4f} STB/day/psi",
                      "N/A (fully saturated — Vogel applies directly from Pr, no linear segment)",
                      f"{res['Qomax']:.2f} STB/day", f"{res['Qomax']:.2f} STB/day"],
        })
        st.table(summary)
        if do_future:
            st.markdown("**Future Vogel IPR prediction**")
            summary_f = pd.DataFrame({
                "Parameter": ["Future reservoir pressure, Pr,f", "Future J",
                              "Future Qob", "Future (Qo)max — Vogel part", "Future (Qo)max — total"],
                "Value": [f"{Prf:,.2f} psi",
                          "N/A (Standing's method scales (Qo)max directly, no separate future J)",
                          "N/A (fully saturated)",
                          f"{res['Qomax_f']:.2f} STB/day", f"{res['Qomax_f']:.2f} STB/day"],
            })
            st.table(summary_f)

    else:  # undersaturated
        res = vogel_undersaturated(Pr, Pb, Pwf_s, Qo_s, Prf if do_future else None)
        st.caption(f"Test-point case detected: Pwf_test {'≥' if res['case']==1 else '<'} Pb "
                   f"→ Case {res['case']} used to compute J.")

        if do_future:
            st.markdown("**Future IPR curve data table** — the exact (Pwf, Qo) points used to draw "
                        "the dashed future curve below, so you can check specific values against the plot.")
            future_curve_df = pd.DataFrame({"Pwf": res["pwf_f"], "Qo": res["qo_f"]})
            st.dataframe(future_curve_df.style.format({"Pwf": "{:.1f}", "Qo": "{:.2f}"}),
                        use_container_width=True)
            st.download_button("Download future IPR curve (CSV)",
                                future_curve_df.to_csv(index=False).encode(),
                                "vogel_undersaturated_future_ipr.csv", key="vogel_under_future_dl")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["qo"], y=res["pwf"], mode="lines", name="Combined Vogel/Linear IPR"))
        fig.add_trace(go.Scatter(x=[Qo_s], y=[Pwf_s], mode="markers", name="Test point",
                                  marker=dict(size=10, color="red")))
        fig.add_hline(y=Pb, line_dash="dot", annotation_text="Pb", annotation_position="right")
        if do_future:
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
        fig.update_layout(title="Vogel IPR — Undersaturated Reservoir",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(rangemode="tozero"), xaxis=dict(rangemode="tozero"),
                           height=520, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results")
        summary = pd.DataFrame({
            "Parameter": ["Productivity index, J (linear, for reference)", "Qob (rate at Pb)",
                          "(Qo)max — Vogel part", "(Qo)max — total"],
            "Value": [f"{res['J']:.4f} STB/day/psi", f"{res['Qob']:.2f} STB/day",
                      f"{res['Qomax_vogel']:.2f} STB/day", f"{res['Qomax_total']:.2f} STB/day"],
        })
        st.table(summary)

        if do_future:
            st.markdown("**Future Vogel IPR prediction**")
            summary_f = pd.DataFrame({
                "Parameter": ["Future reservoir pressure, Pr,f", "Future J", "Future Qob",
                              "Future (Qo)max — Vogel part", "Future (Qo)max — total"],
                "Value": [f"{Prf:,.2f} psi", f"{res['Jf']:.4f} STB/day/psi", f"{res['Qob_f']:.2f} STB/day",
                          f"{res['Qomax_vogel_f']:.2f} STB/day", f"{res['Qomax_total_f']:.2f} STB/day"],
            })
            st.table(summary_f)

# =====================================================================================
#  METHOD 3 — FETKOVICH
# =====================================================================================
else:
    # Main panel heading changes according to selected Fetkovich demo
    if demo_dict is not None:
        st.header(f"Fetkovich's Method — {demo_dict['label']}")
    else:
        st.header("Fetkovich's Method")

    st.write(
        "Provide the average reservoir pressure and a multi-rate test table "
        "(several Pwf / Qo pairs). The app fits Qo = C·(Pr² − Pwf²)ⁿ via a log-log "
        "regression."
    )

    # Use the selected Fetkovich demo dataset
    if demo_dict is not None:
        default_Pr = demo_dict["Pr"]
        default_table = demo_dict["table"]
        default_Prf = demo_dict["Prf"]
    else:
        # Fallback values when uploading your own data
        default_Pr = 3600.0
        default_table = pd.DataFrame({
            "Pwf": [3170, 2890, 2440, 2150],
            "Qo": [263, 383, 497, 640]
        })
        default_Prf = 2000.0

    actual_future_table = None

    if demo_dict is not None:
        actual_future_table = demo_dict.get("future_actual_table")

    if demo_dict is not None:
        default_Pr = demo_dict["Pr"]
        default_table = demo_dict["table"]
        default_Prf = demo_dict["Prf"]
        actual_future_table = demo_dict.get("future_actual_table")
        if actual_future_table is not None:
            st.caption("A real, published multi-rate test. The same well was re-tested 8 months "
                       "later after reservoir pressure dropped to 1605 psi — that repeat test is "
                       "available below to check the future-IPR prediction against what actually "
                       "happened.")
        elif "note" in demo_dict:
            st.caption(demo_dict["note"] + " Since this is real historical production data rather "
                       "than a controlled test, treat the fitted n and C as approximate.")

    if user_df is not None:
        pwf_col = guess_column(user_df, ["pwf", "bottom"])
        qo_col = guess_column(user_df, ["qo", "oil rate", "rate"])
        if pwf_col and qo_col:
            default_table = user_df[[pwf_col, qo_col]].rename(columns={pwf_col: "Pwf", qo_col: "Qo"})
            st.info(f"Auto-detected columns from your upload: '{pwf_col}' → Pwf, '{qo_col}' → Qo. "
                    "Edit the table below if needed.")
        else:
            st.warning("Couldn't auto-detect Pwf/Qo columns in your upload — showing the demo "
                       "table below; edit it with your own values.")

    Pr = st.number_input("Average reservoir pressure, Pr (psi)", value=float(default_Pr), min_value=0.0)

    st.write("Multi-rate test data (edit, add, or delete rows as needed — the regression uses "
             "least squares across every row, so keep only 2 rows to reproduce a classic "
             "two-point Fetkovich fit):")
    edited = st.data_editor(default_table.reset_index(drop=True), num_rows="dynamic",
                             use_container_width=True, key="fetk_table")

    edited = edited.dropna()
    edited = edited[(edited["Pwf"] < Pr) & (edited["Qo"] > 0) & (edited["Pwf"] >= 0)]

    do_future = st.checkbox("Also predict a future IPR curve at a different reservoir pressure",
                             value=True, key="fetk_future_chk")
    Prf = None
    show_actual_future = False
    if do_future:
        Prf = st.number_input("Future average reservoir pressure, Pr,future (psi)",
                               value=float(default_Prf), min_value=0.0, key="fetk_prf")

        if actual_future_table is not None:
            show_actual_future = st.checkbox(
                "Overlay the actual repeat-test data (real future test, for comparison)",
                value=True, key="fetk_show_actual_future",
            )

    if len(edited) < 2:
        st.warning("Need at least 2 valid test points (Pwf < Pr, Qo > 0) to fit the Fetkovich model.")
    else:
        res = fetkovich(Pr, edited["Pwf"].values, edited["Qo"].values, Prf if do_future else None)

        st.subheader("Test data table")
        st.dataframe(res["table"].style.format({"Pwf": "{:.1f}", "Qo": "{:.2f}",
                                                  "Pr^2 - Pwf^2": "{:,.0f}"}),
                     use_container_width=True)

        st.subheader("Log-log data table")
        st.dataframe(res["loglog"].style.format({"log(Qo)": "{:.4f}", "log(Pr^2 - Pwf^2)": "{:.4f}"}),
                     use_container_width=True)

        # Log-log plot: the actual test points plus the fitted straight line
        # (log(Pr^2 - Pwf^2) = slope * log(Qo) + intercept), which is what the
        # Fetkovich regression above is actually fitting.
        loglog_x = res["loglog"]["log(Qo)"]
        loglog_y = res["loglog"]["log(Pr^2 - Pwf^2)"]
        x_line = np.linspace(loglog_x.min(), loglog_x.max(), 2)
        y_line = res["slope"] * x_line + res["intercept"]

        loglog_fig = go.Figure()
        loglog_fig.add_trace(go.Scatter(x=loglog_x, y=loglog_y, mode="markers", name="Test data",
                                         marker=dict(size=9, color="red")))
        loglog_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                         name=f"Fitted line (slope = {res['slope']:.4f})"))
        loglog_fig.update_layout(title="Fetkovich Log-Log Plot",
                                  xaxis_title="log(Qo)", yaxis_title="log(Pr² − Pwf²)",
                                  height=480, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(loglog_fig, use_container_width=True)

        st.subheader("Regression results")
        AOF_present = res["qo"][-1]  # curve value at Pwf = 0 (last point of the linspace, Pr -> 0)
        summary = pd.DataFrame({
            "Parameter": ["Slope", "Intercept", "n (exponent)", "C (coefficient)",
                          "AOF — total Q at Pwf = 0"],
            "Value": [f"{res['slope']:.4f}", f"{res['intercept']:.4f}",
                      f"{res['n']:.4f}", f"{res['C']:.6g}", f"{AOF_present:,.2f} STB/day"],
        })
        st.table(summary)

        if do_future:
            curve_f_df = pd.DataFrame({"Pwf": res["pwf_f"], "Qo (future)": res["qo_f"]})
            st.markdown("**Future IPR curve data table** — the exact (Pwf, Qo) points used to draw "
                        "the dashed future curve below, so you can check specific values against the plot.")
            st.dataframe(curve_f_df.style.format({"Pwf": "{:.1f}", "Qo (future)": "{:.2f}"}),
                        use_container_width=True)
            st.download_button("Download future-IPR curve (CSV)",
                                curve_f_df.to_csv(index=False).encode(), "fetkovich_future_ipr.csv",
                                key="fetk_future_dl")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["qo"], y=res["pwf"], mode="lines", name="Fetkovich IPR"))
        fig.add_trace(go.Scatter(x=[AOF_present], y=[0], mode="markers+text",
                                  text=[f"AOF = {AOF_present:,.0f}"], textposition="top center",
                                  marker=dict(size=9, color="#1f77b4"),
                                  name="AOF (present)", showlegend=False))
        if do_future:
            AOF_future = res["qo_f"][-1]
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[AOF_future], y=[0], mode="markers+text",
                                      text=[f"Future AOF = {AOF_future:,.0f}"], textposition="top left",
                                      marker=dict(size=9, color="#87ceeb"),
                                      name="AOF (future)", showlegend=False))
            if actual_future_table is not None and show_actual_future:
                fig.add_trace(go.Scatter(
                    x=actual_future_table["Qo"], y=actual_future_table["Pwf"], mode="markers",
                    name="Actual repeat test (real future data)",
                    marker=dict(size=10, symbol="diamond", color="green"),
                ))
        fig.update_layout(title="Fetkovich IPR Curve",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(rangemode="tozero"),
                           xaxis=dict(rangemode="tozero"),
                           height=520,
                           margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        if do_future:
            st.markdown("**Future Fetkovich IPR prediction** — n is assumed unchanged; "
                        "C is scaled linearly with reservoir pressure (C_future = C × Pr,future / Pr).")
            AOF_future = res["qo_f"][-1]  # curve value at Pwf = 0 for the future curve
            summary_f = pd.DataFrame({
                "Parameter": ["Future Pr", "n (unchanged)", "C (future)",
                              "Future AOF — total Q at Pwf = 0"],
                "Value": [f"{Prf:,.2f} psi", f"{res['n']:.4f}", f"{res['Cf']:.6g}",
                          f"{AOF_future:,.2f} STB/day"],
            })
            st.table(summary_f)

        st.subheader("Generated IPR curve values")
        curve_df = pd.DataFrame({"Pwf": res["pwf"], "Qo (present)": res["qo"]})
        with st.expander("Preview the present-IPR curve data table"):
            st.dataframe(curve_df.style.format({"Pwf": "{:.1f}", "Qo (present)": "{:.2f}"}),
                        use_container_width=True)
        if do_future:
            st.download_button("Download present-IPR curve (CSV)",
                                curve_df.to_csv(index=False).encode(), "fetkovich_present_ipr.csv",
                                key="fetk_present_dl")
        else:
            st.download_button("Download IPR curve (CSV)",
                                curve_df.to_csv(index=False).encode(), "fetkovich_ipr.csv",
                                key="fetk_ipr_dl")

st.divider()
st.caption("Built with Streamlit · IPR formulas: constant-J, Vogel (1968), and Fetkovich (1973) "
           "empirical inflow performance relationships.")