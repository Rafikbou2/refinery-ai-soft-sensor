"""Refinery AI — Predictive Quality & Process Optimization (process engineer UI).

    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from softsensor import SoftSensor, generate_history
from softsensor.config import FEATURES, MANIPULATED, SPEC_BY_NAME, TAG_BY_NAME, TARGETS
from softsensor.i18n import LANGUAGES, RTL, fmt_time, ltr, spec_label, tag_label, tr
from softsensor.io import (EXCEL_EXT, FileFormatError, ReadReport, excel_sheets, make_template,
                           read_operating_file)
from softsensor.optimizer import recommend
from softsensor.risk import build_alerts, compute_kpis, hours_to_breach, risk_level, sensor_health

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "soft_sensor.joblib"
DATA_PATH = ROOT / "data" / "history.csv"

# Chart roles (validated categorical + status palette).
C_PRED, C_LAB, C_TRUE = "#2a78d6", "#eb6834", "#898781"
STATUS = {"low": ("#0ca30c", "✅"), "medium": ("#fab219", "⚠️"), "high": ("#d03b3b", "⛔")}

st.set_page_config(page_title="Refinery AI — Soft Sensor", page_icon="🛢️", layout="wide")

# ------------------------------------------------------------------ language
lang = st.sidebar.radio("🌐 اللغة / Language", list(LANGUAGES), format_func=LANGUAGES.get,
                        horizontal=True, key="lang")


def T(key: str, **kw) -> str:
    return tr(lang, key, **kw)


def level_text(level: str) -> str:
    return T(f"risk_{level}")


def L(value) -> str:
    """A number/unit string kept in reading order inside Arabic text."""
    return ltr(value, lang)


def with_unit(label: str, unit: str) -> str:
    return f"{label} ({L(unit)})"


def rtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tables render left-to-right; put the first column on the right for Arabic."""
    return df[df.columns[::-1]] if lang in RTL else df


st.markdown("""<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }
[data-testid="stSidebarUserContent"] { padding-top: 1rem; }
</style>""", unsafe_allow_html=True)
if lang in RTL:
    st.markdown("""<style>
    [data-testid="stMain"], [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    [data-testid="stPlotlyChart"], [data-testid="stSlider"], [data-testid="stDataFrame"],
    [data-testid="stCode"] { direction: ltr; }
    </style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Training soft sensor…")
def load_model() -> SoftSensor:
    if MODEL_PATH.exists():
        return SoftSensor.load(MODEL_PATH)
    hist = load_history()
    sensor = SoftSensor().fit(hist)
    sensor.save(MODEL_PATH)
    return sensor


@st.cache_data
def load_history() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    hist = generate_history(days=180)
    DATA_PATH.parent.mkdir(exist_ok=True)
    hist.to_csv(DATA_PATH)
    return hist


@st.cache_data
def read_upload(content: bytes, name: str, sheet) -> tuple[pd.DataFrame, ReadReport]:
    return read_operating_file(content, name, sheet)


@st.cache_data
def template_bytes() -> bytes:
    return make_template(load_history()[FEATURES + [f"{t}_lab" for t in TARGETS]].tail(48))


@st.cache_data
def predict_cached(_sensor: SoftSensor, df: pd.DataFrame) -> pd.DataFrame:
    return _sensor.predict(df)


sensor = load_model()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛢️ Refinery AI")
st.sidebar.caption(T("brand_caption"))
source = st.sidebar.radio(T("data_source"), ["upload", "demo"],
                          format_func=lambda k: T(f"src_{k}"))
st.sidebar.download_button(T("template_btn"), template_bytes(), "refinery_ai_template.xlsx",
                           help=T("template_help"))
if source == "upload":
    up = st.sidebar.file_uploader(T("uploader"), type=["xlsx", "xlsm", "xls", "csv"])
    if up is None:
        st.title(T("app_title"))
        st.info(T("upload_intro"))
        st.stop()
    content = up.getvalue()
    sheet = 0
    if up.name.lower().endswith(EXCEL_EXT):
        sheets = excel_sheets(content)
        if len(sheets) > 1:
            sheet = st.sidebar.selectbox(T("sheet"), sheets)
    try:
        data_all, report = read_upload(content, up.name, sheet)
    except FileFormatError as e:
        st.error(T("read_error", name=up.name, err=e.message(lang)))
        st.stop()
    msg = T("file_ok", name=up.name, n=f"{len(data_all):,}",
            start=fmt_time(lang, data_all.index[0]), end=fmt_time(lang, data_all.index[-1]))
    if not report.has_lab:
        msg += T("no_lab")
    st.sidebar.success(msg)
    if report.dropped_rows or report.filled_values or report.ignored:
        with st.sidebar.expander(T("file_details")):
            if report.dropped_rows:
                st.write(T("rows_skipped", n=report.dropped_rows))
            if report.filled_values:
                st.write(T("gaps_filled", n=report.filled_values))
            if report.ignored:
                st.write(T("ignored_cols", cols=", ".join(report.ignored)))
            st.write(T("recognised_cols"), report.mapping)
else:
    data_all = load_history()

pred_all = predict_cached(sensor, data_all)

first, last = data_all.index[0].to_pydatetime(), data_all.index[-1].to_pydatetime()
replay_start = max(first, (data_all.index[-1] - pd.Timedelta(days=14)).to_pydatetime())
if replay_start < last:
    now = pd.Timestamp(st.sidebar.slider(
        T("replay"), min_value=replay_start, max_value=last, value=last,
        step=pd.Timedelta(minutes=10).to_pytimedelta(), format="DD MMM HH:mm", key="replay_now"))
else:
    now = pd.Timestamp(last)
window_h = st.sidebar.select_slider(
    T("window"), [12, 24, 48, 72, 168], value=48,
    format_func=lambda h: T("hours", n=h) if h < 48 else T("days", n=h // 24))
show_truth = st.sidebar.checkbox(T("show_truth"), value=False,
                                 disabled="t95_true" not in data_all.columns,
                                 help=T("show_truth_help"))

data = data_all[(data_all.index > now - pd.Timedelta(hours=window_h)) & (data_all.index <= now)]
pred = pred_all.loc[data.index]
cur, cur_pred = data.iloc[-1], pred.iloc[-1]
alerts = build_alerts(data_all[data_all.index <= now], pred_all[pred_all.index <= now],
                      {t: m["rmse"] for t, m in sensor.metrics.items()})
kpi = compute_kpis(data, pred)

# ------------------------------------------------------------------ header
worst = max(float(cur_pred[f"{t}_p_off"]) for t in TARGETS)
worst_level = risk_level(worst)
color, icon = STATUS[worst_level]
h1, h2 = st.columns([3, 1])
h1.title(T("app_title"))
h1.caption(T("snapshot", t=fmt_time(lang, now) + f" {now:%Y}", n=len(alerts),
             a=sensor.trained_on["start"][:10], b=sensor.trained_on["end"][:10]))
h2.markdown(
    f"<div style='border:2px solid {color};border-radius:10px;padding:10px 14px;margin-top:18px'>"
    f"<div style='font-size:.8rem;opacity:.75'>{T('offspec_risk')}</div>"
    f"<div style='font-size:1.4rem;font-weight:600'>{icon} {level_text(worst_level)} · "
    f"{L(f'{worst:.0%}')}</div></div>", unsafe_allow_html=True)

tab_live, tab_risk, tab_opt, tab_model = st.tabs(
    [T("tab_live"), T("tab_risk"), T("tab_opt"), T("tab_model")])


def layout(fig: go.Figure, title: str, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), hovermode="x unified",
                      title=dict(text=title, font=dict(size=15),
                                 x=0.99 if lang in RTL else 0, xanchor="right" if lang in RTL else "left"),
                      showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=0 if lang in RTL else 1,
                                  xanchor="left" if lang in RTL else "right"))
    fig.update_xaxes(showgrid=False)
    return fig


def quality_chart(target: str, height: int = 300) -> go.Figure:
    spec = SPEC_BY_NAME[target]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.index, y=pred[f"{target}_hi"], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=pred.index, y=pred[f"{target}_lo"], line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.18)",
                             name=T("interval90"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred[f"{target}_pred"], name=T("soft_sensor"),
                             line=dict(color=C_PRED, width=2),
                             hovertemplate="%{y:.1f} " + spec.unit))
    if show_truth:
        fig.add_trace(go.Scatter(x=data.index, y=data[f"{target}_true"], name=T("true_sim"),
                                 line=dict(color=C_TRUE, width=1.5, dash="dot"),
                                 hovertemplate="%{y:.1f} " + spec.unit))
    lab = data[f"{target}_lab"].dropna()
    fig.add_trace(go.Scatter(x=lab.index, y=lab, mode="markers", name=T("lab"),
                             marker=dict(color=C_LAB, size=9, line=dict(width=2, color="white")),
                             hovertemplate=T("lab") + " %{y:.1f} " + spec.unit))
    fig.add_hline(y=spec.limit, line=dict(color=STATUS["high"][0], width=1.5, dash="dash"),
                  annotation_text=T("spec_max" if spec.direction == "max" else "spec_min",
                                    v=f"{spec.limit:g}", u=spec.unit),
                  annotation_position="top left" if spec.direction == "max" else "bottom left")
    return layout(fig, f"{spec_label(lang, target)} ({spec.unit})", height)


# ------------------------------------------------------------------ live tab
with tab_live:
    one_h = pred[pred.index <= now - pd.Timedelta(hours=1)]
    c = st.columns(6)
    for col, t in zip(c[:2], TARGETS):
        s = SPEC_BY_NAME[t]
        delta = cur_pred[f"{t}_pred"] - one_h[f"{t}_pred"].iloc[-1] if len(one_h) else None
        col.metric(T("now", label=spec_label(lang, t, short=True)),
                   L(f"{cur_pred[f'{t}_pred']:.1f} {s.unit}"),
                   None if delta is None else T("delta_1h", d=f"{delta:+.1f}"),
                   delta_color="inverse" if s.direction == "max" else "normal",
                   help=T("interval_help", lo=f"{cur_pred[f'{t}_lo']:.1f}",
                          hi=f"{cur_pred[f'{t}_hi']:.1f}"))
    c[2].metric(T("on_spec"), L(f"{kpi['on_spec_pct']:.1f} %"))
    c[3].metric(T("diesel_yield"), L(f"{kpi['diesel_yield_pct']:.1f} vol%"),
                T("m3_day", v=f"{kpi['diesel_m3_day']:,.0f}"), delta_color="off")
    c[4].metric(T("giveaway"), L(f"${kpi['giveaway_usd_day']:,.0f}"),
                T("avg_margin", v=f"{kpi['t95_margin_c']:.1f}"), delta_color="off",
                help=T("giveaway_help"))
    c[5].metric(T("furnace_energy"), L(f"{kpi['energy_kwh_per_m3']:.1f}"),
                L(f"{kpi['furnace_mw']:.1f} MW"), delta_color="off")

    left, right = st.columns(2)
    for col, t in zip((left, right), TARGETS):
        col.plotly_chart(quality_chart(t), width="stretch")

    st.subheader(T("process_tags"))
    tags = st.multiselect(T("tags"), FEATURES, default=["diesel_draw_rate", "diesel_draw_temp",
                                                         "strip_steam", "crude_api"],
                          format_func=lambda n: tag_label(lang, n))
    cols = st.columns(2)
    for i, tag in enumerate(tags):
        unit = TAG_BY_NAME[tag].unit
        fig = go.Figure(go.Scatter(x=data.index, y=data[tag], line=dict(color=C_PRED, width=2),
                                   hovertemplate="%{y:.2f} " + unit, name=tag_label(lang, tag)))
        cols[i % 2].plotly_chart(layout(fig, f"{tag_label(lang, tag)} ({unit})", 200, legend=False),
                                 width="stretch")

# ------------------------------------------------------------------ risk tab
with tab_risk:
    st.subheader(T("active_alerts"))
    if not alerts:
        st.success(T("no_alerts"))
    for a in alerts:
        box = st.error if a.severity == "critical" else st.warning
        box(f"**{T('sev_' + a.severity)} · {T('cat_' + a.category)}** — {a.message(lang)}"
            f"\n\n➡️ {a.action(lang)}")

    st.subheader(T("risk_outlook"))
    r = st.columns(len(TARGETS))
    for col, t in zip(r, TARGETS):
        s = SPEC_BY_NAME[t]
        p = float(cur_pred[f"{t}_p_off"])
        lv = risk_level(p)
        clr, ic = STATUS[lv]
        h = hours_to_breach(t, pred_all[pred_all.index <= now])
        side = "right" if lang in RTL else "left"
        col.markdown(
            f"<div style='border-{side}:5px solid {clr};padding:6px 12px'>"
            f"<b>{spec_label(lang, t)}</b><br>"
            f"{T('risk_line', icon=ic, level=level_text(lv), p=f'{p:.0%}')}<br>"
            f"{T('margin_to_spec', v=f'{abs(s.limit - cur_pred[f'{t}_pred']):.1f} {s.unit}')}"
            f"{T('outside') if p > 0.5 else ''}<br>"
            f"{T('trend_stable') if h is None else T('trend_breach', h=f'{h:.1f}')}</div>",
            unsafe_allow_html=True)

    fig = go.Figure()
    for t, clr in zip(TARGETS, (C_PRED, C_LAB)):
        fig.add_trace(go.Scatter(x=pred.index, y=pred[f"{t}_p_off"] * 100,
                                 name=spec_label(lang, t), line=dict(color=clr, width=2),
                                 hovertemplate="%{y:.0f} %"))
    fig.add_hrect(y0=40, y1=100, fillcolor=STATUS["high"][0], opacity=0.07, line_width=0,
                  annotation_text=T("zone_high"), annotation_position="top left")
    fig.add_hrect(y0=10, y1=40, fillcolor=STATUS["medium"][0], opacity=0.07, line_width=0,
                  annotation_text=T("zone_medium"), annotation_position="top left")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(layout(fig, T("p_off_chart"), 300), width="stretch")

    a1, a2 = st.columns([3, 2])
    fig = go.Figure(go.Scatter(x=pred.index, y=-pred["anomaly_score"], name=T("anomaly_series"),
                               line=dict(color=C_PRED, width=2), hovertemplate="%{y:.3f}"))
    fig.add_hline(y=-sensor.anomaly_threshold, line=dict(color=STATUS["high"][0], dash="dash"),
                  annotation_text=T("unusual_op"), annotation_position="top left")
    a1.plotly_chart(layout(fig, T("anomaly_chart"), 260, legend=False), width="stretch")
    a2.markdown(T("health_title"))
    health = sensor_health(data_all[(data_all.index <= now)
                                    & (data_all.index > now - pd.Timedelta(hours=24))])
    health["label"] = [tag_label(lang, n) for n in health["tag"]]
    health["status"] = health["status"].map(lambda s: T(f"st_{s}"))
    a2.dataframe(rtl_columns(health[["label", "status", "frozen_points", "spikes"]].rename(columns={
        "label": T("col_tag"), "status": T("col_status"), "frozen_points": T("col_frozen"),
        "spikes": T("col_spikes")})), hide_index=True, width="stretch")

# ------------------------------------------------------------------ optimize tab
with tab_opt:
    st.subheader(T("rec_title"))
    st.caption(T("rec_caption"))
    o1, o2 = st.columns([1, 3])
    max_p = o1.slider(T("max_p"), 1, 20, 5, format="%d %%") / 100
    if o1.button(T("compute"), type="primary"):
        st.session_state["rec"] = (now, recommend(sensor, cur[FEATURES], max_p_off=max_p))
    rec_entry = st.session_state.get("rec")
    if rec_entry and rec_entry[0] == now:
        rec = rec_entry[1]
        if rec["status"] != "ok":
            o2.warning(T("no_safe"))
        table = pd.DataFrame({
            T("col_variable"): [with_unit(tag_label(lang, t), TAG_BY_NAME[t].unit) for t in MANIPULATED],
            T("col_current"): rec["current"].round(2).to_numpy(),
            T("col_recommended"): rec["setpoints"].round(2).to_numpy(),
            T("col_move"): (rec["setpoints"] - rec["current"]).round(2).to_numpy(),
        })
        o2.dataframe(rtl_columns(table), hide_index=True, width="stretch")
        m = o2.columns(len(TARGETS) + 1)
        for col, t in zip(m, TARGETS):
            s = SPEC_BY_NAME[t]
            col.metric(T("after", label=spec_label(lang, t, short=True)),
                       L(f"{rec['prediction'][f'{t}_pred']:.1f} {s.unit}"),
                       T("risk_after", p=f"{rec['prediction'][f'{t}_p_off']:.0%}",
                         q=f"{rec['current_prediction'][f'{t}_p_off']:.0%}"), delta_color="off")
        m[-1].metric(T("margin_gain"), T("usd_day", v=f"{rec['gain_usd_day']:,.0f}"))
    else:
        o2.info(T("press_compute"))

    st.divider()
    st.subheader(T("whatif_title"))
    w = st.columns(3)
    scenario = cur[FEATURES].copy()
    for i, tag in enumerate(FEATURES):
        t = TAG_BY_NAME[tag]
        val = float(np.clip(cur[tag], t.low, t.high))
        scenario[tag] = w[i % 3].slider(with_unit(tag_label(lang, tag), t.unit), float(t.low),
                                        float(t.high), val, step=(t.high - t.low) / 200,
                                        key=f"wi_{tag}_{now}")
    sp = sensor.predict(pd.DataFrame([scenario])).iloc[0]
    res = st.columns(len(TARGETS) + 1)
    for col, t in zip(res, TARGETS):
        s = SPEC_BY_NAME[t]
        lv = risk_level(float(sp[f"{t}_p_off"]))
        col.metric(spec_label(lang, t), L(f"{sp[f'{t}_pred']:.1f} {s.unit}"),
                   T("vs_now", d=f"{sp[f'{t}_pred'] - cur_pred[f'{t}_pred']:+.1f}"),
                   delta_color="inverse" if s.direction == "max" else "normal")
        col.markdown(T("risk_short", icon=STATUS[lv][1], level=level_text(lv),
                       p=f"{sp[f'{t}_p_off']:.0%}"))
    if sp["extrapolating"]:
        res[-1].warning(T("outside_range", tags=", ".join(
            tag_label(lang, n) for n in sp["outside_tags"].split(", "))))

# ------------------------------------------------------------------ model tab
with tab_model:
    st.subheader(T("perf_title"))
    mt = pd.DataFrame(sensor.metrics).T
    mt.index = [spec_label(lang, t) for t in mt.index]
    names = {"rmse": T("m_rmse"), "mae": T("m_mae"), "r2": T("m_r2"), "coverage_90": T("m_cov"),
             "linear_only_rmse": T("m_lin"), "n_train": T("m_ntrain"), "n_test": T("m_ntest")}
    st.dataframe(mt.rename(columns=names).style.format({
        names["rmse"]: "{:.2f}", names["mae"]: "{:.2f}", names["r2"]: "{:.3f}",
        names["coverage_90"]: "{:.0%}", names["linear_only_rmse"]: "{:.2f}",
        names["n_train"]: "{:.0f}", names["n_test"]: "{:.0f}"}), width="stretch")
    st.caption(T("model_caption"))
    cols = st.columns(len(TARGETS))
    for col, t in zip(cols, TARGETS):
        s, h = SPEC_BY_NAME[t], sensor.holdout_results[t]
        lim = [min(h["lab"].min(), h["pred"].min()) - 1, max(h["lab"].max(), h["pred"].max()) + 1]
        fig = go.Figure([
            go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=C_TRUE, dash="dot", width=1.5),
                       hoverinfo="skip"),
            go.Scatter(x=h["lab"], y=h["pred"], mode="markers",
                       marker=dict(color=C_PRED, size=8, opacity=0.7,
                                   line=dict(width=1, color="white")),
                       hovertemplate=T("holdout_hover") + "<extra></extra>")])
        fig = layout(fig, T("parity_title", label=spec_label(lang, t, short=True), u=s.unit), 340,
                     legend=False)
        fig.update_layout(hovermode="closest", xaxis_title=T("lab"), yaxis_title=T("soft_sensor"))
        fig.update_xaxes(showgrid=True)
        col.plotly_chart(fig, width="stretch")
        imp = sensor.importance[t].clip(lower=0).head(6)[::-1]
        fig = go.Figure(go.Bar(x=imp.to_numpy(), y=[tag_label(lang, n) for n in imp.index],
                               orientation="h", marker=dict(color=C_PRED, cornerradius=4),
                               hovertemplate="%{x:.3f}<extra></extra>"))
        fig = layout(fig, T("drivers"), 260, legend=False)
        fig.update_layout(hovermode="closest")
        col.plotly_chart(fig, width="stretch")
