"""
IPR Analyzer — Inflow Performance Relationship curve builder
Methods: Simple IPR (constant J), Vogel's Method, Fetkovich's Method, Wiggins' Method, and a
Compare tab that scores the last three against one real saturated well (Well B, Keokuk Pool).
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
    Prf=1500.0,
)
DEMO_VOGEL_UNDERSAT = dict(
    label="Undersaturated example (Pr > Pb)",
    Pr=3000.0, Pb=2130.0, Pwf=2500.0, Qo=250.0, 
    table=pd.DataFrame({"Pwf": [3000, 2826, 2652, 2478, 2304, 2130, 1917, 1704, 1491, 1278, 1065, 852, 639, 426, 213, 0 ], "Qo": [0, 87, 174, 261, 348, 435, 536.77, 629.07, 711.9, 785.27, 849.17, 903.6, 948.57, 984.07, 1010.1, 1026.67]}),
    Prf=2000.0,
)

DEMO_FETKOVICH_SAT = dict(
    label="Saturated example (Pr ≤ Pb)",
    Pr=3600.0,
    table=pd.DataFrame({"Pwf": [3170, 2890, 2440, 2150], "Qo": [263, 383, 497, 640]}),
    Prf=2600.0,
)
DEMO_FETKOVICH_UNDERSAT = dict(
    label="Undersaturated example (Pr > Pb)",
    Pr=3600.0,
    table=pd.DataFrame({"Pwf": [3000, 2700, 2400, 2100, 1800, 1500, 1350, 1200, 1050, 900, 750, 600, 450, 300, 150, 0], "Qo": [346.78, 470.49, 576.42, 667.08, 743.92, 807.85, 835.18, 859.50, 880.85, 899.29, 914.84, 927.52, 937.37, 944.39, 948.60, 950.00]}),
    Prf=2600.0,
)

# =====================================================================================
#  FIELD CASE 3 — Well B, Keokuk Pool, Seminole County, Oklahoma (Aug 1935)
#  A real SATURATED well (Pr <= Pb). It is the common "exam paper" used to test Vogel,
#  Fetkovich and Wiggins against the same measured data — it does not belong to any one
#  method. The well was re-tested 8 months later (Pr fell from 1714 to 1605 psi), so each
#  method's future-IPR prediction can be checked against a real repeat test.
# =====================================================================================
WELL_B_NAME = "Field Case 3 — Saturated reservoir, Well B, Keokuk Pool, Oklahoma (1935)"
WELL_B_PR = 1714.0      # present average reservoir pressure (psi)
WELL_B_PB = 3420.0      # bubble-point pressure (psi) -> Pr <= Pb, so saturated
WELL_B_PRF = 1605.0     # real reservoir pressure at the 8-month repeat test (psi)
WELL_B_TEST_PWF = 1196.0   # single test point used to calibrate Vogel / Wiggins
WELL_B_TEST_QO = 1125.0

WELL_B_PRESENT_TABLE = pd.DataFrame({
    "Pwf": [1714, 1583, 1443, 1272, 1196, 982],
    "Qo": [0, 280, 508, 780, 1125, 1335],
})
WELL_B_FUTURE_TABLE = pd.DataFrame({
    "Pwf": [1605, 1381, 1231, 1120],
    "Qo": [0, 420, 720, 850],
})

# Vogel — Well B (saturated). Future (Qo)max uses Eickmeier's cube law, as in the workbook.
FIELD_CASE_3_VOGEL = dict(
    label=WELL_B_NAME,
    Pr=WELL_B_PR, Pb=WELL_B_PB, Pwf=WELL_B_TEST_PWF, Qo=WELL_B_TEST_QO, Prf=WELL_B_PRF,
    historical_prf=True,     # 1605 psi is a REAL recorded pressure, so it stays fixed
    future_method="eickmeier",
    table=WELL_B_PRESENT_TABLE,
    future_actual_table=WELL_B_FUTURE_TABLE,
)

# Fetkovich — Well B. Uses the full multi-rate test table (every row is used in the fit).
FIELD_CASE_3 = dict(
    label=WELL_B_NAME,
    Pr=WELL_B_PR,
    table=WELL_B_PRESENT_TABLE,
    Prf=WELL_B_PRF,
    historical_prf=True,
    future_actual_table=WELL_B_FUTURE_TABLE,
)

# Wiggins — Well B. Wiggins' own coefficients: C1 = 0.52 (present); future shape C1 = 0.15.
# (Qo)max = 2878.8 STB/day is the value given in the reference workbook for this well.
FIELD_CASE_3_WIGGINS = dict(
    label=WELL_B_NAME,
    Pr=WELL_B_PR, Pb=WELL_B_PB, Qomax=2878.8, C1=0.52,
    Prf=WELL_B_PRF, C1_f=0.15, future_method="wiggins",
    Pwf=WELL_B_TEST_PWF, Qo=WELL_B_TEST_QO,    # only used if you switch to "back-calculate"
    historical_prf=True,
    table=WELL_B_PRESENT_TABLE,
    future_actual_table=WELL_B_FUTURE_TABLE,
)

# Compare tab — Wiggins vs Vogel (first approximation) vs Fetkovich, all scored on Well B.
FIELD_CASE_3_COMPARE = dict(
    label=WELL_B_NAME,
    Pr=WELL_B_PR, Pb=WELL_B_PB,
    calib_pwf=WELL_B_TEST_PWF, calib_qo=WELL_B_TEST_QO,   # single point used for Vogel only
    C1_present=0.52, Qomax=2878.8, C1_future=0.15, future_method="wiggins",
    table=WELL_B_PRESENT_TABLE,
    Prf=WELL_B_PRF,
    future_actual_table=WELL_B_FUTURE_TABLE,
)



METHOD_LABELS = {
    "1": "1. Simple IPR (constant J)",
    "2": "2. Vogel's Method",
    "3": "3. Fetkovich's Method",
    "4": "4. Wiggins' Method",
    "5": "5. Compare (Well B)",
}

# One flat, ordered catalog of every demo dataset. Picking one also determines which method
# section renders below. Well B (Field Case 3) appears under Vogel, Fetkovich and Wiggins only.
DEMO_CATALOG = [
    # Simple IPR
    ("1", "Simple IPR-demo", DEMO_IPR),

    # Vogel
    ("2", "Vogel — saturated example (Pr ≤ Pb)", DEMO_VOGEL_SAT),
    ("2", "Vogel — undersaturated example (Pr > Pb)", DEMO_VOGEL_UNDERSAT),
    ("2", "Field Case 3 — Saturated, Well B, Keokuk Pool (1935)", FIELD_CASE_3_VOGEL),

    # Fetkovich
    ("3", "Fetkovich — saturated synthetic example (reference workbook)", DEMO_FETKOVICH_SAT),
    ("3", "Fetkovich — undersaturated synthetic example (reference workbook)", DEMO_FETKOVICH_UNDERSAT),
    ("3", "Field Case 3 — Saturated, Well B, Keokuk Pool (1935) — multi-rate test", FIELD_CASE_3),

    # Wiggins
    ("4", "Field Case 3 — Saturated, Well B, Keokuk Pool (1935)", FIELD_CASE_3_WIGGINS),
]
DEMO_LABELS = [label for _, label, _ in DEMO_CATALOG]
DEMO_LOOKUP = {(method_key, label): d for method_key, label, d in DEMO_CATALOG}  # keyed by method too: Well B appears under several methods
DEMO_LABELS_BY_METHOD = {
    mk: [label for m, label, _ in DEMO_CATALOG if m == mk] for mk in ("1", "2", "3", "4")
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


def vogel_saturated(Pr, Pwf_s, Qo_s, Prf=None, future_method="standing"):
    J_const = Qo_s / (Pr - Pwf_s)
    Qomax = Qo_s / (1 - 0.2 * (Pwf_s / Pr) - 0.8 * (Pwf_s / Pr) ** 2)

    pwf_curve = np.linspace(Pr, 0, 25)
    qo_vogel = Qomax * (1 - 0.2 * (pwf_curve / Pr) - 0.8 * (pwf_curve / Pr) ** 2)
    qo_linear = J_const * (Pr - pwf_curve)

    out = dict(J=J_const, Qomax=Qomax, pwf=pwf_curve, qo_vogel=qo_vogel, qo_linear=qo_linear)

    if Prf is not None:
        ratio = Prf / Pr
        Qomax_f_standing = Qomax * ratio * (0.2 + 0.8 * ratio)   # Standing / Vogel's own future eq.
        Qomax_f_eickmeier = Qomax * ratio ** 3                   # Eickmeier cube law
        Qomax_f = Qomax_f_eickmeier if future_method == "eickmeier" else Qomax_f_standing
        pwf_f = np.linspace(Prf, 0, 25)
        qo_f = Qomax_f * (1 - 0.2 * (pwf_f / Prf) - 0.8 * (pwf_f / Prf) ** 2)
        out.update(Qomax_f=Qomax_f, Qomax_f_standing=Qomax_f_standing,
                   Qomax_f_eickmeier=Qomax_f_eickmeier, pwf_f=pwf_f, qo_f=qo_f)

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


def vogel_type_ipr(Pr, Pwf_test, Qo_test, C1, Prf=None, C1_f=None):
    """Generalized Vogel-type IPR: Qo = Qomax * [1 - C1*(Pwf/Pr) - (1-C1)*(Pwf/Pr)^2].
    Vogel's method is the special case C1 = 0.2 (fixed); here C1 is a free coefficient,
    back-calculated / tuned against one calibration test point so the curve can be fit
    tighter to a specific well. Future (Qo)max uses Eickmeier's equation:
    Qomax,f = Qomax,p * (Pr,f / Pr,p)^3 — a cube-of-pressure-ratio scaling, distinct
    from Standing's future-Vogel equation used elsewhere in this app.
    """
    Qomax = Qo_test / (1 - C1 * (Pwf_test / Pr) - (1 - C1) * (Pwf_test / Pr) ** 2)
    pwf_curve = np.linspace(Pr, 0, 25)
    qo_curve = Qomax * (1 - C1 * (pwf_curve / Pr) - (1 - C1) * (pwf_curve / Pr) ** 2)
    out = dict(Qomax=Qomax, pwf=pwf_curve, qo=qo_curve)

    if Prf is not None:
        Qomax_f = Qomax * (Prf / Pr) ** 3  # Eickmeier's equation
        pwf_f = np.linspace(Prf, 0, 25)
        qo_f = Qomax_f * (1 - C1_f * (pwf_f / Prf) - (1 - C1_f) * (pwf_f / Prf) ** 2)
        out.update(Qomax_f=Qomax_f, pwf_f=pwf_f, qo_f=qo_f)

    return out


def eval_wiggins(pwf_arr, Pr, Qomax, C1):
    """Evaluate the Wiggins / Vogel equation at specific Pwf points (for
    comparing predicted vs. actual test-data rates, rather than drawing a curve)."""
    pwf_arr = np.asarray(pwf_arr, dtype=float)
    return Qomax * (1 - C1 * (pwf_arr / Pr) - (1 - C1) * (pwf_arr / Pr) ** 2)


def qomax_from_test(Pr, Pwf_test, Qo_test, C1):
    """Back-calculate (Qo)max from one stabilized test point (Pwf_test < Pr)."""
    x = Pwf_test / Pr
    return Qo_test / (1 - C1 * x - (1 - C1) * x ** 2)


def wiggins_ipr(Pr, C1, Qomax, Prf=None, C1_f=None, future_method="wiggins"):
    """Wiggins' IPR for a saturated oil reservoir:
        Qo = Qomax * [1 - C1*(Pwf/Pr) - (1-C1)*(Pwf/Pr)^2]      (Wiggins: C1 = 0.52)
    (Qo)max is SUPPLIED — the curve is not forced through any test point.
    Future (Qo)max, either:
        'wiggins'   : Qomax_p * [0.15*(Prf/Pr) + 0.84*(Prf/Pr)^2]
        'eickmeier' : Qomax_p * (Prf/Pr)^3
    The future curve then uses the future coefficient C1_f."""
    pwf_curve = np.linspace(Pr, 0, 25)
    out = dict(Qomax=Qomax, pwf=pwf_curve, qo=eval_wiggins(pwf_curve, Pr, Qomax, C1))
    if Prf is not None:
        r = Prf / Pr
        Qomax_f_wiggins = Qomax * (0.15 * r + 0.84 * r ** 2)
        Qomax_f_eickmeier = Qomax * r ** 3
        Qomax_f = Qomax_f_eickmeier if future_method == "eickmeier" else Qomax_f_wiggins
        pwf_f = np.linspace(Prf, 0, 25)
        out.update(Qomax_f=Qomax_f, Qomax_f_wiggins=Qomax_f_wiggins,
                   Qomax_f_eickmeier=Qomax_f_eickmeier,
                   pwf_f=pwf_f, qo_f=eval_wiggins(pwf_f, Prf, Qomax_f, C1_f))
    return out


def pct_error(pred, actual):
    """Percentage error vs. actual test data: |pred - actual| / actual * 100.
    Rows where actual = 0 (e.g. the Pwf = Pr point) are left as NaN — undefined %
    error, excluded from the average rather than treated as a zero error."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(actual > 0, np.abs(pred - actual) / actual * 100, np.nan)
    return err


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
        st.markdown("Or Eickmeier's cube law (selectable in the Vogel section; used for Well B):")
        st.latex(r"(Q_o)_{max,f} = (Q_o)_{max,p}\left(\dfrac{P_{r,f}}{P_{r,p}}\right)^3")
        st.markdown("---")
        st.markdown("**Undersaturated (Pr > Pb)** — combined linear + Vogel:")
        st.latex(r"\text{if } P_{wf,test}\!\ge\! P_b:\ J = \dfrac{Q_o}{P_r - P_{wf,test}}")
        st.latex(r"\text{if } P_{wf,test}\!<\! P_b:\ J = \dfrac{Q_o}{(P_r - P_b) + \frac{P_b}{1.8}\left[1 - 0.2\frac{P_{wf,test}}{P_b} - 0.8\left(\frac{P_{wf,test}}{P_b}\right)^2\right]}")
        st.latex(r"Q_{ob} = J(P_r - P_b) \qquad (Q_o)_{max} = Q_{ob} + \dfrac{JP_b}{1.8}")
        st.latex(r"Q_o(P_{wf}) = \begin{cases} J(P_r - P_{wf}) & P_{wf}\ge P_b \\ Q_{ob} + \frac{JP_b}{1.8}\left[1 - 0.2\frac{P_{wf}}{P_b} - 0.8\left(\frac{P_{wf}}{P_b}\right)^2\right] & P_{wf} < P_b \end{cases}")
        st.markdown("Future IPR:")
        st.latex(r"J_f = J_p\left(\dfrac{P_{r,f}}{P_{r,p}}\right)^2")

    elif method_key == "3":
        st.markdown("**Fetkovich (1973) empirical deliverability equation:**")
        st.latex(r"Q_o = C\,(P_r^2 - P_{wf}^2)^n")
        st.markdown("Fitted via log-log linear regression of the test data:")
        st.latex(r"\log Q_o = n \log(P_r^2 - P_{wf}^2) + \log C")
        st.markdown("Future IPR — n assumed unchanged, C scaled with pressure:")
        st.latex(r"C_f = C_p\left(\dfrac{P_{r,f}}{P_{r,p}}\right)")

    elif method_key in ("4", "5"):
        st.markdown("**Wiggins' correlation (saturated oil reservoir):**")
        st.latex(r"Q_o = (Q_o)_{max}\left[1 - 0.52\left(\dfrac{P_{wf}}{P_r}\right) - 0.48\left(\dfrac{P_{wf}}{P_r}\right)^2\right]")
        st.markdown("Written with a free coefficient C1 (Wiggins' own value is C1 = 0.52; Vogel is C1 = 0.2):")
        st.latex(r"Q_o(P_{wf}) = (Q_o)_{max}\left[1 - C_1\left(\frac{P_{wf}}{P_r}\right) - (1-C_1)\left(\frac{P_{wf}}{P_r}\right)^2\right]")
        st.markdown("Future (Qo)max — Wiggins:")
        st.latex(r"(Q_o)_{max,f} = (Q_o)_{max,p}\left[0.15\left(\dfrac{P_{r,f}}{P_{r,p}}\right) + 0.84\left(\dfrac{P_{r,f}}{P_{r,p}}\right)^2\right]")
        st.markdown("or Eickmeier's cube law (selectable):")
        st.latex(r"(Q_o)_{max,f} = (Q_o)_{max,p}\left(\dfrac{P_{r,f}}{P_{r,p}}\right)^3")
        st.markdown("Percentage error vs. actual test data:")
        st.latex(r"\%\,\text{error} = \dfrac{|Q_{o,pred} - Q_{o,actual}|}{Q_{o,actual}} \times 100")


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
        ["1. IPR", "2. Vogel's", "3. Fetkovich's", "4. Wiggins'", "5. Compare (Well B)"],
    )
    method_key = method[0]

    demo_dict = None
    demo_label = None
    if method_key == "5":
        st.header("3 · Demo dataset")
        st.caption("Compare mode always uses the real saturated Well B (Keokuk Pool) dataset — "
                   "not configurable here. Choose Present or Future in the main panel.")
    elif data_source == "Use demo data":
        st.header("3 · Demo dataset")
        options = DEMO_LABELS_BY_METHOD[method_key]
        if len(options) > 1:
            demo_label = st.selectbox("Choose a demo dataset for this method", options)
        else:
            demo_label = options[0]
            st.caption(f"Only one demo dataset for this method: **{demo_label}**")
        demo_dict = DEMO_LOOKUP[(method_key, demo_label)]
    else:
        st.caption("Not used — you're uploading your own data.")

    # Used to scope widget keys (like the future-pressure input) to the currently
    # selected dataset, so switching datasets resets those widgets to the new
    # dataset's own defaults instead of Streamlit reusing a stale value from
    # whichever dataset was previously selected.
    demo_key_suffix = demo_label if demo_label is not None else "custom"

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
           "Vogel's, Fetkovich's and Wiggins' methods, then compare the last three on one real well. "
           "Use the sidebar to set the data source, method, demo dataset (or upload), and to "
           "review the formulas.")

with st.expander("ℹ️ About the Well B (Keokuk Pool) demo datasets"):
    st.markdown(
        "**Field Case 3 — Well B, Keokuk Pool, Seminole County, Oklahoma (August 1935)** is a real, "
        "published **saturated** well (Pr = 1,714 psi, Pb = 3,420 psi). It is used as the common test "
        "case for Vogel's, Fetkovich's and Wiggins' methods, so each method is scored against the "
        "same measured data. The well was genuinely **re-tested 8 months later** after reservoir "
        "pressure dropped to 1,605 psi, so each method's future-IPR prediction can be checked against "
        "what actually happened in the field. The Compare tab puts the three methods side by side."
    )

if method_key != "5":
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
    Pr = c1.number_input("Average reservoir pressure, Pr (psi)", value=float(d["Pr"]), min_value=0.0,
                          key=f"simple_pr_{demo_key_suffix}")
    Pwf_s = c2.number_input("Stabilized Pwf (psi)", value=float(d["Pwf"]), min_value=0.0, max_value=Pr,
                             key=f"simple_pwf_{demo_key_suffix}")
    Qo_s = c3.number_input("Stabilized Qo (STB/day)", value=float(d["Qo"]), min_value=0.0,
                            key=f"simple_qo_{demo_key_suffix}")

    if Pwf_s >= Pr:
        st.warning("Pwf must be less than Pr.")
    else:
        J, AOF, pwf_c, qo_c = simple_ipr(Pr, Pwf_s, Qo_s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qo_c, y=pwf_c, mode="lines", name="IPR curve"))
        fig.add_trace(go.Scatter(x=[Qo_s], y=[Pwf_s], mode="markers", name="Test point",
                                  marker=dict(size=10, color="red")))
        fig.add_trace(go.Scatter(x=[AOF], y=[0], mode="markers+text",
                                  text=[f"AOF = {AOF:,.0f}"], textposition="top center",
                                  marker=dict(size=9, color="#1f77b4"), showlegend=False))
        fig.update_layout(title="IPR Curve (Simple / Constant J)",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Pr * 0.06, Pr * 1.08]),
                           xaxis=dict(rangemode="tozero"),
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
    Pr = c1.number_input("Average reservoir pressure, Pr (psi)", value=float(d["Pr"]), min_value=0.0,
                          key=f"vogel_pr_{demo_key_suffix}")
    Pb = c2.number_input("Bubble-point pressure, Pb (psi)", value=float(d["Pb"]), min_value=0.0,
                          key=f"vogel_pb_{demo_key_suffix}")

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
                             value=float(d["Pwf"]), min_value=0.0, max_value=Pr,
                             key=f"vogel_pwf_{demo_key_suffix}")
    Qo_s = c4.number_input("Stabilized Qo test point (STB/day)", value=float(d["Qo"]), min_value=0.0,
                            key=f"vogel_qo_{demo_key_suffix}")

    do_future = st.checkbox("Also predict a future IPR curve at a different reservoir pressure", value=True)
    Prf = None
    if do_future:
        # The two Well B Vogel demos share the same real 8-month repeat-test pressure as the
        # Fetkovich Well B case, so that pressure stays fixed. Every other dataset (including
        # custom/no-demo input) gets a default that tracks whatever Pr is currently set to.
        is_historical_prf = demo_dict is not None and demo_dict.get("historical_prf", False)
        if is_historical_prf:
            default_Prf = d["Prf"]
            prf_key = f"vogel_prf_{demo_key_suffix}"
        else:
            default_Prf = max(100.0, Pr - 1000.0)
            prf_key = f"vogel_prf_{demo_key_suffix}_{Pr:.0f}"

        Prf = st.number_input("Future average reservoir pressure, Pr,future (psi)",
                               value=float(default_Prf), min_value=0.0,
                               key=prf_key)

    future_method = "standing"
    if do_future and case == "saturated":
        fm_options = ["eickmeier", "standing"]
        fm_names = {"eickmeier": "Eickmeier — (Qo)max,f = (Qo)max,p × (Pr,f / Pr,p)³",
                    "standing": "Standing — (Qo)max,f = (Qo)max,p × (Pr,f / Pr,p) × [0.2 + 0.8 (Pr,f / Pr,p)]"}
        future_method = st.radio("How should (Qo)max change with the future pressure?", fm_options,
                                  index=fm_options.index(d.get("future_method", "standing")),
                                  format_func=lambda k: fm_names[k],
                                  key=f"vogel_fm_{demo_key_suffix}")

    if case == "saturated":
        res = vogel_saturated(Pr, Pwf_s, Qo_s, Prf if do_future else None, future_method)

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
        fig.add_trace(go.Scatter(x=[res["Qomax"]], y=[0], mode="markers+text",
                                  text=[f"(Qo)max = {res['Qomax']:,.0f}"], textposition="top center",
                                  marker=dict(size=9, color="#1f77b4"), showlegend=False))
        linear_AOF = res["J"] * Pr
        fig.add_trace(go.Scatter(x=[linear_AOF], y=[0], mode="markers+text",
                                  text=[f"Linear AOF = {linear_AOF:,.0f}"], textposition="bottom center",
                                  marker=dict(size=8, color="gray", symbol="diamond"), showlegend=False))
        if do_future:
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future Vogel IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[res["Qomax_f"]], y=[0], mode="markers+text",
                                      text=[f"Future (Qo)max = {res['Qomax_f']:,.0f}"], textposition="top left",
                                      marker=dict(size=9, color="#87ceeb"), showlegend=False))
        if "table" in d and "future_actual_table" in d:   # real well: overlay the measured data
            fig.add_trace(go.Scatter(x=d["table"]["Qo"], y=d["table"]["Pwf"], mode="markers",
                                      name="Actual test data",
                                      marker=dict(size=9, color="green", symbol="diamond")))
            if do_future:
                fig.add_trace(go.Scatter(x=d["future_actual_table"]["Qo"], y=d["future_actual_table"]["Pwf"],
                                          mode="markers", name="Actual repeat test",
                                          marker=dict(size=10, color="darkgreen", symbol="diamond-open")))
        fig.update_layout(title="Vogel IPR — Saturated Reservoir",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Pr * 0.08, Pr * 1.08]),
                           xaxis=dict(rangemode="tozero"),
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
                          f"N/A ({future_method.capitalize()} scaling is applied to (Qo)max directly, no separate future J)",
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
        fig.add_trace(go.Scatter(x=[res["Qomax_total"]], y=[0], mode="markers+text",
                                  text=[f"(Qo)max total = {res['Qomax_total']:,.0f}"], textposition="top center",
                                  marker=dict(size=9, color="#1f77b4"), showlegend=False))
        if do_future:
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[res["Qomax_total_f"]], y=[0], mode="markers+text",
                                      text=[f"Future (Qo)max total = {res['Qomax_total_f']:,.0f}"],
                                      textposition="top left",
                                      marker=dict(size=9, color="#87ceeb"), showlegend=False))
        fig.update_layout(title="Vogel IPR — Undersaturated Reservoir",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Pr * 0.08, Pr * 1.08]),
                           xaxis=dict(rangemode="tozero"),
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
elif method_key == "3":
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
        actual_future_table = demo_dict.get("future_actual_table")
        if actual_future_table is not None:
            st.caption("A real, published multi-rate test. The same well was re-tested 8 months "
                       "later after reservoir pressure dropped to 1605 psi — that repeat test is "
                       "available below to check the future-IPR prediction against what actually "
                       "happened.")
        elif "note" in demo_dict:
            st.caption(demo_dict["note"] + " Since this is real historical production data rather "
                       "than a controlled test, treat the fitted n and C as approximate.")
    else:
        # Fallback values when uploading your own data
        default_Pr = 3600.0
        default_table = pd.DataFrame({
            "Pwf": [3170, 2890, 2440, 2150],
            "Qo": [263, 383, 497, 640]
        })
        actual_future_table = None

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

    Pr = st.number_input("Average reservoir pressure, Pr (psi)", value=float(default_Pr), min_value=0.0,
                          key=f"fetk_pr_{demo_key_suffix}")

    st.write("Multi-rate test data (edit, add, or delete rows as needed — the regression uses "
             "least squares across every row, so keep only 2 rows to reproduce a classic "
             "two-point Fetkovich fit):")
    edited = st.data_editor(default_table.reset_index(drop=True), num_rows="dynamic",
                             use_container_width=True, key=f"fetk_table_{demo_key_suffix}")

    edited = edited.dropna()
    edited = edited[(edited["Pwf"] < Pr) & (edited["Qo"] > 0) & (edited["Pwf"] >= 0)]

    do_future = st.checkbox("Also predict a future IPR curve at a different reservoir pressure",
                             value=True, key="fetk_future_chk")
    Prf = None
    show_actual_future = False
    if do_future:
        # The real Well B multi-rate dataset carries a genuine historical future pressure
        # (the actual repeat-test condition) — keep that fixed. Every other dataset (including
        # custom uploads) gets a default that tracks whatever Pr is currently set to, so
        # editing Pr updates the suggested future pressure instead of leaving behind a
        # stale, unrelated number.
        is_historical_prf = demo_dict is not None and demo_dict.get("historical_prf", False)
        if is_historical_prf:
            default_Prf = demo_dict["Prf"]
            prf_key = f"fetk_prf_{demo_key_suffix}"
        else:
            default_Prf = max(100.0, Pr - 1000.0)
            prf_key = f"fetk_prf_{demo_key_suffix}_{Pr:.0f}"

        Prf = st.number_input("Future average reservoir pressure, Pr,future (psi)",
                               value=float(default_Prf), min_value=0.0,
                               key=prf_key)

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
        fig.add_trace(go.Scatter(x=res["table"]["Qo"], y=res["table"]["Pwf"], mode="markers",
                                  name="Test data used in fit",
                                  marker=dict(size=9, color="green", symbol="diamond")))
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
                           yaxis=dict(range=[-Pr * 0.08, Pr * 1.08]),
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

# =====================================================================================
#  METHOD 4 — WIGGINS
# =====================================================================================
elif method_key == "4":
    st.header("Wiggins' Method")
    st.write("Wiggins' correlation for a **saturated** oil reservoir: "
             "Qo = (Qo)max · [1 − 0.52·(Pwf/Pr) − 0.48·(Pwf/Pr)²]. The app writes it with a coefficient "
             "C1 (Wiggins' own value is C1 = 0.52). Unlike Vogel, the curve is not forced through a "
             "test point — (Qo)max is supplied, so the curve can sit above or below the measured data. "
             "That gap is exactly what the Compare tab scores.")

    d = demo_dict if demo_dict is not None else dict(
        Pr=2000.0, Pb=2500.0, Pwf=1500.0, Qo=500.0, C1=0.52, Prf=1700.0, C1_f=0.15)

    c1, c2, c3 = st.columns(3)
    Pr = c1.number_input("Average reservoir pressure, Pr (psi)", value=float(d["Pr"]), min_value=0.0,
                          key=f"wig_pr_{demo_key_suffix}")
    Pb = c2.number_input("Bubble-point pressure, Pb (psi)", value=float(d["Pb"]), min_value=0.0,
                          key=f"wig_pb_{demo_key_suffix}")
    C1 = c3.number_input("Coefficient, C1 (Wiggins = 0.52)", value=float(d["C1"]), min_value=0.0,
                          max_value=1.0, step=0.01, key=f"wig_c1_{demo_key_suffix}")

    if Pr <= Pb:
        st.success(f"Since Pr ({Pr:,.0f}) ≤ Pb ({Pb:,.0f}) → **Saturated reservoir** — Wiggins' "
                   "correlation applies directly.")
    else:
        st.warning(f"Pr ({Pr:,.0f}) > Pb ({Pb:,.0f}): the reservoir is undersaturated, which this "
                   "saturated-reservoir correlation does not model. Treat results with caution.")

    mode_opts = ["Enter (Qo)max directly", "Back-calculate (Qo)max from one test point"]
    mode = st.radio("Where does (Qo)max come from?", mode_opts,
                     index=0 if "Qomax" in d else 1, horizontal=True,
                     key=f"wig_mode_{demo_key_suffix}")
    if mode == mode_opts[0]:
        Qomax_in = st.number_input("(Qo)max (STB/day)", value=float(d.get("Qomax", 1000.0)),
                                    min_value=0.0, key=f"wig_qomax_{demo_key_suffix}")
        Qomax = Qomax_in
        Pwf_s = Qo_s = None
    else:
        c4, c5 = st.columns(2)
        Pwf_s = c4.number_input("Stabilized Pwf test point (psi)", value=float(d.get("Pwf", 0.5 * Pr)),
                                 min_value=0.0, max_value=Pr, key=f"wig_pwf_{demo_key_suffix}")
        Qo_s = c5.number_input("Stabilized Qo test point (STB/day)", value=float(d.get("Qo", 500.0)),
                                min_value=0.0, key=f"wig_qo_{demo_key_suffix}")
        Qomax = qomax_from_test(Pr, Pwf_s, Qo_s, C1) if Pwf_s < Pr else None

    do_future = st.checkbox("Also predict a future IPR curve at a different reservoir pressure",
                             value=True, key="wig_future_chk")
    Prf, C1_f, fut_method = None, None, "wiggins"
    if do_future:
        is_historical_prf = demo_dict is not None and demo_dict.get("historical_prf", False)
        if is_historical_prf:
            default_Prf = d["Prf"]
            prf_key = f"wig_prf_{demo_key_suffix}"
        else:
            default_Prf = max(100.0, Pr - 1000.0)
            prf_key = f"wig_prf_{demo_key_suffix}_{Pr:.0f}"
        f1, f2 = st.columns(2)
        Prf = f1.number_input("Future average reservoir pressure, Pr,future (psi)",
                               value=float(default_Prf), min_value=0.0, key=prf_key)
        C1_f = f2.number_input("Future coefficient, C1,future (shapes the future curve)",
                                value=float(d.get("C1_f", C1)), min_value=0.0, max_value=1.0,
                                step=0.01, key=f"wig_c1f_{demo_key_suffix}")
        fm_opts = ["wiggins", "eickmeier"]
        fm_names = {"wiggins": "Wiggins — (Qo)max,f = (Qo)max,p × [0.15 (Pr,f/Pr,p) + 0.84 (Pr,f/Pr,p)²]",
                    "eickmeier": "Eickmeier — (Qo)max,f = (Qo)max,p × (Pr,f/Pr,p)³"}
        fut_method = st.radio("How should (Qo)max change with the future pressure?", fm_opts,
                               index=fm_opts.index(d.get("future_method", "wiggins")),
                               format_func=lambda k: fm_names[k], key=f"wig_fm_{demo_key_suffix}")

    if Qomax is None:
        st.warning("Pwf must be less than Pr to back-calculate (Qo)max.")
    else:
        res = wiggins_ipr(Pr, C1, Qomax, Prf if do_future else None, C1_f, fut_method)
        has_actual = "table" in d and "future_actual_table" in d   # real well with measured data

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["qo"], y=res["pwf"], mode="lines", name="Wiggins IPR"))
        fig.add_trace(go.Scatter(x=[Qomax], y=[0], mode="markers+text",
                                  text=[f"(Qo)max = {Qomax:,.0f}"], textposition="top center",
                                  marker=dict(size=9, color="#1f77b4"), showlegend=False))
        if Pwf_s is not None:
            fig.add_trace(go.Scatter(x=[Qo_s], y=[Pwf_s], mode="markers", name="Calibration test point",
                                      marker=dict(size=11, color="red")))
        if has_actual:
            fig.add_trace(go.Scatter(x=d["table"]["Qo"], y=d["table"]["Pwf"], mode="markers",
                                      name="Actual test data",
                                      marker=dict(size=9, color="green", symbol="diamond")))
        if do_future:
            fig.add_trace(go.Scatter(x=res["qo_f"], y=res["pwf_f"], mode="lines",
                                      name=f"Future Wiggins IPR (Pr={Prf:,.0f})", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[res["Qomax_f"]], y=[0], mode="markers+text",
                                      text=[f"Future (Qo)max = {res['Qomax_f']:,.0f}"],
                                      textposition="top left",
                                      marker=dict(size=9, color="#87ceeb"), showlegend=False))
            if has_actual:
                fig.add_trace(go.Scatter(x=d["future_actual_table"]["Qo"], y=d["future_actual_table"]["Pwf"],
                                          mode="markers", name="Actual repeat test",
                                          marker=dict(size=10, color="darkgreen", symbol="diamond-open")))
        fig.update_layout(title="Wiggins IPR — Saturated Reservoir",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Pr * 0.06, Pr * 1.08]),
                           xaxis=dict(rangemode="tozero"),
                           height=520, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results")
        st.table(pd.DataFrame({
            "Parameter": ["Coefficient, C1", "(Qo)max (present)"],
            "Value": [f"{C1:.4f}", f"{Qomax:,.2f} STB/day"],
        }))

        curve_df = pd.DataFrame({"Pwf": res["pwf"], "Qo (present)": res["qo"]})
        with st.expander("Present IPR curve data table"):
            st.dataframe(curve_df.style.format({"Pwf": "{:.1f}", "Qo (present)": "{:.2f}"}),
                         use_container_width=True)
        st.download_button("Download present IPR curve (CSV)", curve_df.to_csv(index=False).encode(),
                            "wiggins_present_ipr.csv", key="wig_present_dl")

        if do_future:
            st.markdown("**Future Wiggins IPR prediction**")
            st.table(pd.DataFrame({
                "Parameter": ["Future reservoir pressure, Pr,f", "Future coefficient, C1,f",
                              "Future (Qo)max — Wiggins equation", "Future (Qo)max — Eickmeier cube law",
                              "Future (Qo)max — used"],
                "Value": [f"{Prf:,.2f} psi", f"{C1_f:.4f}",
                          f"{res['Qomax_f_wiggins']:,.2f} STB/day", f"{res['Qomax_f_eickmeier']:,.2f} STB/day",
                          f"{res['Qomax_f']:,.2f} STB/day ({fut_method.capitalize()})"],
            }))
            curve_f_df = pd.DataFrame({"Pwf": res["pwf_f"], "Qo (future)": res["qo_f"]})
            with st.expander("Future IPR curve data table"):
                st.dataframe(curve_f_df.style.format({"Pwf": "{:.1f}", "Qo (future)": "{:.2f}"}),
                             use_container_width=True)
            st.download_button("Download future IPR curve (CSV)", curve_f_df.to_csv(index=False).encode(),
                                "wiggins_future_ipr.csv", key="wig_future_dl")

        if has_actual:
            st.subheader("Predicted vs. actual (this well's measured data)")
            pwf_a = d["table"]["Pwf"].values.astype(float)
            act_a = d["table"]["Qo"].values.astype(float)
            pred_a = eval_wiggins(pwf_a, Pr, Qomax, C1)
            err_a = pct_error(pred_a, act_a)
            st.dataframe(pd.DataFrame({"Pwf (psi)": pwf_a, "Actual Qo": act_a,
                                        "Wiggins Qo": pred_a, "% error": err_a})
                         .style.format({"Pwf (psi)": "{:.0f}", "Actual Qo": "{:.2f}",
                                         "Wiggins Qo": "{:.2f}", "% error": "{:.2f}"}),
                         use_container_width=True)
            st.caption(f"Average absolute error (present): {np.nanmean(err_a):.2f}%")
            if do_future:
                pwf_b = d["future_actual_table"]["Pwf"].values.astype(float)
                act_b = d["future_actual_table"]["Qo"].values.astype(float)
                pred_b = eval_wiggins(pwf_b, Prf, res["Qomax_f"], C1_f)
                err_b = pct_error(pred_b, act_b)
                st.dataframe(pd.DataFrame({"Pwf (psi)": pwf_b, "Actual Qo (repeat test)": act_b,
                                            "Wiggins Qo (future)": pred_b, "% error": err_b})
                             .style.format({"Pwf (psi)": "{:.0f}", "Actual Qo (repeat test)": "{:.2f}",
                                             "Wiggins Qo (future)": "{:.2f}", "% error": "{:.2f}"}),
                             use_container_width=True)
                st.caption(f"Average absolute error (future, vs. repeat test): {np.nanmean(err_b):.2f}%")

# =====================================================================================
#  METHOD 5 — COMPARE (Well B: Wiggins vs Vogel's first approximation vs Fetkovich's)
# =====================================================================================
else:
    cc = FIELD_CASE_3_COMPARE
    st.header(f"Compare the three methods — {cc['label']}")
    st.write(
        "Runs **Wiggins'**, **Vogel's** (first approximation) and **Fetkovich's** methods side by "
        "side on the real saturated Well B test data, and scores each one's predicted rate against "
        "what was actually measured. Vogel is calibrated on one test point "
        f"(Pwf = {cc['calib_pwf']:,.0f} psi, Qo = {cc['calib_qo']:,.0f} STB/day); Wiggins uses its "
        "own coefficients and the (Qo)max given for this well; Fetkovich is fitted on every "
        "multi-rate test point."
    )

    view = st.radio("View", ["Present", "Future"], horizontal=True, key="cmp_view")
    Pr = cc["Pr"]
    table = cc["table"]
    fit_tbl = table[(table["Pwf"] < Pr) & (table["Qo"] > 0)]   # Qo = 0 at Pwf = Pr can't be log-fit
    fit_pwf = fit_tbl["Pwf"].values.astype(float)
    fit_qo = fit_tbl["Qo"].values.astype(float)

    def avg_error_table(err_df):
        return pd.DataFrame({
            "Method": ["Wiggins", "Vogel's", "Fetkovich's"],
            "Average absolute error (%)": [np.nanmean(err_df[c]) for c in ["Wiggins", "Vogel's", "Fetkovich's"]],
        })

    if view == "Present":
        w1, w2 = st.columns(2)
        C1 = w1.number_input("Wiggins coefficient, C1  (Wiggins = 0.52)",
                              value=float(cc["C1_present"]), min_value=0.0, max_value=1.0, step=0.01,
                              key="cmp_c1_present")
        Qomax_w = w2.number_input("Wiggins (Qo)max (STB/day)", value=float(cc["Qomax"]),
                                   min_value=0.0, key="cmp_qomax")

        wig = wiggins_ipr(Pr, C1, Qomax_w)
        vogel = vogel_type_ipr(Pr, cc["calib_pwf"], cc["calib_qo"], 0.2)
        fet = fetkovich(Pr, fit_pwf, fit_qo)

        pwf_pts = table["Pwf"].values.astype(float)
        actual = table["Qo"].values.astype(float)
        pred_wig = eval_wiggins(pwf_pts, Pr, Qomax_w, C1)
        pred_vog = eval_wiggins(pwf_pts, Pr, vogel["Qomax"], 0.2)
        pred_fet = fet["C"] * (Pr ** 2 - pwf_pts ** 2) ** fet["n"]

        st.subheader("Test data table")
        comp_df = pd.DataFrame({
            "Pwf (psi)": pwf_pts, "Oil Rate — actual (bbl/day)": actual,
            "Wiggins — Qo": pred_wig, "Vogel's — Qo": pred_vog, "Fetkovich's — Qo": pred_fet,
        })
        st.dataframe(comp_df.style.format({c: "{:.2f}" for c in comp_df.columns[1:]}),
                     use_container_width=True)

        st.subheader("Percentage error (vs. actual)")
        err_df = pd.DataFrame({
            "Wiggins": pct_error(pred_wig, actual),
            "Vogel's": pct_error(pred_vog, actual),
            "Fetkovich's": pct_error(pred_fet, actual),
        })
        st.dataframe(err_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("Average absolute errors")
        st.table(avg_error_table(err_df).style.format({"Average absolute error (%)": "{:.2f}"}))

        st.subheader("Fitted parameters")
        st.table(pd.DataFrame({
            "Method": ["Wiggins", "Vogel's", "Fetkovich's"],
            "Parameters": [f"C1 = {C1:.2f}, (Qo)max = {Qomax_w:,.2f} STB/day (given)",
                           f"C1 = 0.20, (Qo)max = {vogel['Qomax']:,.2f} STB/day (from the test point)",
                           f"n = {fet['n']:.4f}, C = {fet['C']:.4g}  (least squares on {len(fit_pwf)} test points)"],
        }))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wig["qo"], y=wig["pwf"], mode="lines", name="Wiggins"))
        fig.add_trace(go.Scatter(x=vogel["qo"], y=vogel["pwf"], mode="lines", name="Vogel's"))
        fig.add_trace(go.Scatter(x=fet["qo"], y=fet["pwf"], mode="lines", name="Fetkovich's"))
        fig.add_trace(go.Scatter(x=actual, y=pwf_pts, mode="markers", name="Actual",
                                  marker=dict(size=10, color="green", symbol="diamond")))
        fig.update_layout(title="IPR curve — method comparison (present)",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Pr * 0.06, Pr * 1.08]), xaxis=dict(rangemode="tozero"),
                           height=560, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download comparison table (CSV)", comp_df.to_csv(index=False).encode(),
                            "well_b_compare_present.csv", key="cmp_present_dl")

    else:  # Future
        Prf = cc["Prf"]
        st.caption(f"Future prediction at Pr,future = {Prf:,.0f} psi — the real 8-month "
                   "repeat-test pressure — checked against the actual repeat test. "
                   "Vogel scales (Qo)max with Eickmeier's cube law; Fetkovich keeps n and scales C "
                   "with pressure.")

        w1, w2, w3 = st.columns(3)
        C1 = w1.number_input("Present Wiggins coefficient, C1", value=float(cc["C1_present"]),
                              min_value=0.0, max_value=1.0, step=0.01, key="cmp_c1_present_f")
        Qomax_w = w2.number_input("Wiggins (Qo)max (STB/day)", value=float(cc["Qomax"]),
                                   min_value=0.0, key="cmp_qomax_f")
        C1_f = w3.number_input("Future Wiggins coefficient, C1,future", value=float(cc["C1_future"]),
                                min_value=0.0, max_value=1.0, step=0.01, key="cmp_c1_future")
        fm_opts = ["wiggins", "eickmeier"]
        fm_names = {"wiggins": "Wiggins — (Qo)max,f = (Qo)max,p × [0.15 (Pr,f/Pr,p) + 0.84 (Pr,f/Pr,p)²]",
                    "eickmeier": "Eickmeier — (Qo)max,f = (Qo)max,p × (Pr,f/Pr,p)³"}
        fut_method = st.radio("Wiggins future (Qo)max scaling", fm_opts,
                               index=fm_opts.index(cc["future_method"]),
                               format_func=lambda k: fm_names[k], key="cmp_wig_fm")

        wig = wiggins_ipr(Pr, C1, Qomax_w, Prf, C1_f, fut_method)
        vogel = vogel_type_ipr(Pr, cc["calib_pwf"], cc["calib_qo"], 0.2, Prf, 0.2)
        fet = fetkovich(Pr, fit_pwf, fit_qo, Prf)

        fut = cc["future_actual_table"]
        pwf_pts = fut["Pwf"].values.astype(float)
        actual = fut["Qo"].values.astype(float)
        pred_wig = eval_wiggins(pwf_pts, Prf, wig["Qomax_f"], C1_f)
        pred_vog = eval_wiggins(pwf_pts, Prf, vogel["Qomax_f"], 0.2)
        pred_fet = fet["Cf"] * (Prf ** 2 - pwf_pts ** 2) ** fet["n"]

        st.subheader("Future test data table")
        comp_df = pd.DataFrame({
            "Pwf (psi)": pwf_pts, "Oil Rate — actual (bbl/day)": actual,
            "Wiggins — Qo": pred_wig, "Vogel's — Qo": pred_vog, "Fetkovich's — Qo": pred_fet,
        })
        st.dataframe(comp_df.style.format({c: "{:.2f}" for c in comp_df.columns[1:]}),
                     use_container_width=True)

        st.subheader("Percentage error (vs. actual repeat test)")
        err_df = pd.DataFrame({
            "Wiggins": pct_error(pred_wig, actual),
            "Vogel's": pct_error(pred_vog, actual),
            "Fetkovich's": pct_error(pred_fet, actual),
        })
        st.dataframe(err_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("Average absolute errors")
        st.table(avg_error_table(err_df).style.format({"Average absolute error (%)": "{:.2f}"}))

        st.subheader("Future parameters")
        st.table(pd.DataFrame({
            "Method": ["Wiggins", "Vogel's", "Fetkovich's"],
            "Parameters": [f"C1,f = {C1_f:.2f}, (Qo)max,f = {wig['Qomax_f']:,.2f} STB/day ({fut_method.capitalize()})",
                           f"C1 = 0.20, (Qo)max,f = {vogel['Qomax_f']:,.2f} STB/day (Eickmeier)",
                           f"n = {fet['n']:.4f}, C,f = {fet['Cf']:.4g}"],
        }))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wig["qo_f"], y=wig["pwf_f"], mode="lines", name="Wiggins"))
        fig.add_trace(go.Scatter(x=vogel["qo_f"], y=vogel["pwf_f"], mode="lines", name="Vogel's"))
        fig.add_trace(go.Scatter(x=fet["qo_f"], y=fet["pwf_f"], mode="lines", name="Fetkovich's"))
        fig.add_trace(go.Scatter(x=actual, y=pwf_pts, mode="markers", name="Actual (repeat test)",
                                  marker=dict(size=10, color="green", symbol="diamond")))
        fig.update_layout(title="Future IPR curve — method comparison",
                           xaxis_title="Qo (STB/day)", yaxis_title="Pwf (psi)",
                           yaxis=dict(range=[-Prf * 0.06, Prf * 1.1]), xaxis=dict(rangemode="tozero"),
                           height=560, margin=dict(t=60, b=60, l=60, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download comparison table (CSV)", comp_df.to_csv(index=False).encode(),
                            "well_b_compare_future.csv", key="cmp_future_dl")

st.divider()
st.caption("Built with Streamlit · IPR formulas: constant-J, Vogel (1968), Fetkovich (1973), "
           "and a generalized Vogel-type Wiggins correlation with Eickmeier future scaling.")