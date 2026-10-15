import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import sir_core as sc

st.set_page_config(page_title="SIR Model Simulator", page_icon="🦠", layout="wide")
st.title("🦠 การจำลองและวิเคราะห์การแพร่ระบาดด้วยแบบจำลอง SIR")
st.caption("Math Software Project – กลุ่ม 9 | แก้ ODE เชิงตัวเลข + ข้อมูลจริงไข้หวัดใหญ่โรงเรียนประจำอังกฤษ ปี 1978")

COL = {"S": "#1f77b4", "I": "#d62728", "R": "#2ca02c"}  # ใช้ภาษาอังกฤษในกราฟ กันปัญหาฟอนต์ไทยบน server

tab_theory, tab_sim, tab_scn, tab_real, tab_sum = st.tabs(
    ["1) ทฤษฎี", "2) จำลองสถานการณ์", "3) วิเคราะห์สถานการณ์ (β, γ)", "4) ข้อมูลจริง + ประมาณค่า", "5) สรุปผล"]
)

# ------------------------------------------------------------------ 1) ทฤษฎี
with tab_theory:
    st.subheader("แบบจำลอง SIR")
    st.markdown(
        "แบ่งประชากร N คนเป็น 3 กลุ่ม: **S** เสี่ยงติดเชื้อ, **I** ติดเชื้อและแพร่โรคได้, "
        "**R** หายแล้วและมีภูมิคุ้มกัน (S + I + R = N)"
    )
    st.latex(r"\frac{dS}{dt} = -\beta \frac{S I}{N}")
    st.latex(r"\frac{dI}{dt} = \beta \frac{S I}{N} - \gamma I")
    st.latex(r"\frac{dR}{dt} = \gamma I")
    st.markdown(
        """
- **β** = อัตราการติดเชื้อ (การสัมผัสต่อวัน × โอกาสติดต่อครั้งละ)
- **γ** = อัตราการหายป่วย → ระยะเวลาแพร่เชื้อเฉลี่ย = 1/γ วัน
- **R₀ = β/γ** = จำนวนคนที่ผู้ติดเชื้อ 1 คนแพร่ให้ต่อ ในประชากรที่ยังไม่มีใครมีภูมิ  
  ถ้า R₀ > 1 เกิดการระบาด, ถ้า R₀ < 1 โรคจะค่อย ๆ หายไป
- จุดพีคเกิดเมื่อ S = N/R₀ (ตอนนั้น dI/dt = 0)
- สมการนี้ไม่มีผลเฉลยรูปปิด จึงแก้เชิงตัวเลขด้วย Runge–Kutta (`scipy.integrate.solve_ivp`)
"""
    )

# ------------------------------------------------------------------ 2) จำลอง
with tab_sim:
    st.subheader("ทดลองปรับพารามิเตอร์แล้วดูกราฟทันที")
    c1, c2 = st.columns([1, 3])
    with c1:
        N = st.number_input("ประชากรทั้งหมด N", 100, 100_000_000, 10_000, step=100)
        I0 = st.number_input("ผู้ติดเชื้อเริ่มต้น I(0)", 1, int(N) - 1, 10)
        beta = st.slider("β อัตราการติดเชื้อ (ต่อวัน)", 0.05, 2.0, 0.40, 0.01)
        gamma = st.slider("γ อัตราการหายป่วย (ต่อวัน)", 0.02, 1.0, 0.10, 0.01)
        days = st.slider("จำนวนวันที่จำลอง", 30, 365, 160, 10)
        distancing = st.slider("มาตรการเว้นระยะห่าง (ลด β ลง %)", 0, 90, 0, 5)
    beta_eff = beta * (1 - distancing / 100)
    base = sc.summary(beta, gamma, N, I0, days)
    cur = sc.summary(beta_eff, gamma, N, I0, days)
    with c2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for k in "SIR":
            ax.plot(cur["t"], cur[k], color=COL[k], lw=2.2, label=k)
        if distancing > 0:
            ax.plot(base["t"], base["I"], color=COL["I"], ls="--", lw=1.5, alpha=0.6, label="I (no distancing)")
        ax.axvline(cur["peak_day"], color="gray", ls=":", lw=1)
        ax.set_xlabel("Day"); ax.set_ylabel("Population"); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R₀ = β/γ", f"{cur['R0']:.2f}", None if distancing == 0 else f"{cur['R0']-base['R0']:.2f}")
        m2.metric("จำนวนผู้ติดเชื้อสูงสุด", f"{cur['peak_I']:,.0f}",
                  None if distancing == 0 else f"{cur['peak_I']-base['peak_I']:,.0f}")
        m3.metric("วันที่เกิดพีค", f"{cur['peak_day']:.0f}",
                  None if distancing == 0 else f"{cur['peak_day']-base['peak_day']:+.0f} วัน")
        m4.metric("สัดส่วนที่เคยติดเชื้อ", f"{cur['attack_rate']*100:.1f}%")
        if cur["R0"] <= 1:
            st.info("R₀ ≤ 1: ไม่เกิดการระบาดใหญ่")

# ------------------------------------------------------------------ 3) Scenario
with tab_scn:
    st.subheader("เปรียบเทียบหลายสถานการณ์")
    s1, s2 = st.columns(2)
    Ns = s1.number_input("N", 100, 100_000_000, 10_000, step=100, key="Ns")
    I0s = s2.number_input("I(0)", 1, int(Ns) - 1, 10, key="I0s")
    days_s = st.slider("จำนวนวัน", 30, 365, 160, 10, key="days_s")
    left, right = st.columns(2)

    with left:
        st.markdown("**(ก) เปลี่ยน β ให้ γ คงที่**")
        g_fix = st.slider("γ คงที่", 0.02, 1.0, 0.10, 0.01, key="g_fix")
        betas = st.multiselect("ค่า β", [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], default=[0.2, 0.3, 0.4, 0.6])
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = []
        for b in sorted(betas):
            r = sc.summary(b, g_fix, Ns, I0s, days_s)
            ax.plot(r["t"], r["I"], lw=2, label=f"β={b} (R0={r['R0']:.1f})")
            rows.append({"β": b, "R₀": round(r["R0"], 2), "พีค I": round(r["peak_I"]),
                         "วันพีค": round(r["peak_day"]), "ติดเชื้อรวม %": round(r["attack_rate"] * 100, 1)})
        ax.set_xlabel("Day"); ax.set_ylabel("Infectious I(t)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with right:
        st.markdown("**(ข) เปลี่ยน γ ให้ β คงที่**")
        b_fix = st.slider("β คงที่", 0.05, 2.0, 0.40, 0.01, key="b_fix")
        gammas = st.multiselect("ค่า γ", [0.05, 0.07, 0.1, 0.14, 0.2, 0.33, 0.5], default=[0.07, 0.1, 0.14, 0.2])
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = []
        for g in sorted(gammas):
            r = sc.summary(b_fix, g, Ns, I0s, days_s)
            ax.plot(r["t"], r["I"], lw=2, label=f"γ={g} (R0={r['R0']:.1f})")
            rows.append({"γ": g, "R₀": round(r["R0"], 2), "พีค I": round(r["peak_I"]),
                         "วันพีค": round(r["peak_day"]), "ติดเชื้อรวม %": round(r["attack_rate"] * 100, 1)})
        ax.set_xlabel("Day"); ax.set_ylabel("Infectious I(t)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("**(ค) เว้นระยะห่างทางสังคม: ลด β กี่ % ถึงจะลดพีคได้เท่าไร**")
    b0 = st.slider("β ก่อนมีมาตรการ", 0.1, 1.5, 0.40, 0.01, key="b0")
    g0 = st.slider("γ", 0.02, 0.5, 0.10, 0.01, key="g0")
    cuts = [0, 10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = []
    for c in cuts:
        r = sc.summary(b0 * (1 - c / 100), g0, Ns, I0s, days_s)
        ax.plot(r["t"], r["I"], lw=2, label=f"reduce beta {c}%")
        rows.append({"ลด β (%)": c, "R₀": round(r["R0"], 2), "พีค I": round(r["peak_I"]),
                     "วันพีค": round(r["peak_day"]), "ติดเชื้อรวม %": round(r["attack_rate"] * 100, 1)})
    ax.set_xlabel("Day"); ax.set_ylabel("Infectious I(t)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ 4) ข้อมูลจริง
with tab_real:
    st.subheader("กรณีศึกษา: ไข้หวัดใหญ่ในโรงเรียนประจำชาย อังกฤษ ปี 1978")
    st.markdown(
        """
เหตุการณ์จริงที่ใช้บ่อยในการสอนแบบจำลองโรคติดต่อ: นักเรียน **763 คน** อยู่ในโรงเรียนประจำ (ประชากรปิด)
มี **512 คน** ป่วยจนต้องนอนพัก ข้อมูลคือ **จำนวนนักเรียนที่นอนพักในแต่ละวัน 14 วัน** (22 ม.ค. – 4 ก.พ. 1978)
เราใช้เป็นตัวแทนของ **I(t)** (ผู้ติดเชื้อที่ยังแพร่เชื้อ) และใช้ **512 คน** เป็นจำนวนผู้ติดเชื้อสะสมสุดท้าย (≈ R ตอนจบ)
"""
    )
    up = st.file_uploader("(ทางเลือก) อัปโหลด CSV ของคุณเอง ต้องมีคอลัมน์ day, infected (เช่น export จาก pomp::bsflu)", type="csv")
    N_real = sc.BSFLU_N
    total_cases = sc.BSFLU_TOTAL_CASES
    day_obs, I_obs = sc.BSFLU_DAY, sc.BSFLU_B
    if up is not None:
        df_up = pd.read_csv(up)
        if {"day", "infected"} <= set(df_up.columns):
            day_obs, I_obs = df_up["day"].to_numpy(float), df_up["infected"].to_numpy(float)
            N_real = int(st.number_input("N ของข้อมูลที่อัปโหลด", 10, 100_000_000, N_real))
            total_cases = int(st.number_input("จำนวนติดเชื้อสะสมทั้งหมด (0 = ไม่ทราบ)", 0, N_real, 0))
        else:
            st.error("ไม่พบคอลัมน์ day และ infected ใช้ข้อมูลตัวอย่างแทน")
    st.caption("ตรวจสอบข้อมูลกับแหล่งอ้างอิง (BMJ 1978 / pomp::bsflu) ก่อนนำไปใส่ในรายงาน")

    st.markdown("#### วิธีประมาณค่า: Nonlinear Least Squares")
    st.latex(r"\min_{\beta,\gamma,I_0}\ \sum_{t}\big(I_{\text{model}}(t;\beta,\gamma,I_0) - I_{\text{obs}}(t)\big)^2")
    use_final = st.checkbox("เพิ่มเงื่อนไขให้ผู้ติดเชื้อสะสมช่วงท้ายใกล้ 512 คน (ข้อมูล 'คนที่หายแล้ว')",
                            value=False, disabled=total_cases == 0)
    w = st.slider("น้ำหนักเงื่อนไขนี้", 0.1, 10.0, 2.0, 0.1) if use_final else 0.0
    fit = sc.fit_sir(day_obs, I_obs, N_real, final_R=total_cases if use_final else None, final_R_weight=w)

    st.markdown("#### ปรับค่าเองให้ใกล้เหตุการณ์จริง")
    manual = st.checkbox("ปรับพารามิเตอร์ด้วยมือ (เริ่มจากค่าที่ fit ได้)")
    if manual:
        mc1, mc2, mc3 = st.columns(3)
        beta_u = mc1.slider("β", 0.1, 4.0, float(round(fit["beta"], 2)), 0.01)
        gamma_u = mc2.slider("γ", 0.05, 2.0, float(round(fit["gamma"], 2)), 0.01)
        I0_u = mc3.slider("I(0) ที่วัน 1", 1.0, 20.0, float(round(fit["I0"], 1)), 0.1)
    else:
        beta_u, gamma_u, I0_u = fit["beta"], fit["gamma"], fit["I0"]

    t_plot = np.linspace(day_obs[0], day_obs[0] + 40, 401)
    S, I, R = sc.simulate(beta_u, gamma_u, N_real, I0_u, t_plot)
    pred = sc.simulate(beta_u, gamma_u, N_real, I0_u, day_obs)[1]
    mt = sc.metrics(I_obs, pred)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(day_obs, I_obs, color="black", zorder=3, label="Observed (in bed)")
        ax.plot(t_plot, I, color=COL["I"], lw=2.2, label="SIR model I(t)")
        ax.set_xlim(day_obs[0], day_obs[-1] + 2)
        ax.set_xlabel("Day"); ax.set_ylabel("Boys"); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(t_plot, S, color=COL["S"], lw=2, label="S")
        ax.plot(t_plot, I, color=COL["I"], lw=2, label="I")
        ax.plot(t_plot, R, color=COL["R"], lw=2, label="R")
        if total_cases:
            ax.axhline(total_cases, color="gray", ls="--", label=f"Observed total cases ({total_cases})")
        ax.set_xlabel("Day"); ax.set_ylabel("Boys"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        st.pyplot(fig)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("β", f"{beta_u:.3f}")
    k2.metric("γ", f"{gamma_u:.3f}", f"แพร่เชื้อเฉลี่ย {1/gamma_u:.2f} วัน", delta_color="off")
    k3.metric("R₀ = β/γ", f"{beta_u/gamma_u:.2f}")
    k4.metric("R² ของการ fit", f"{mt['r2']:.3f}")
    k5.metric("RMSE (คน)", f"{mt['rmse']:.1f}")

    st.markdown("#### ความถูกต้องของการ fit")
    st.dataframe(pd.DataFrame({
        "ตัวชี้วัด": ["R²", "RMSE (คน)", "MAE (คน)", "MAPE (%)", "ความคลาดเคลื่อนของค่าพีค (%)"],
        "ค่า": [round(mt["r2"], 3), round(mt["rmse"], 1), round(mt["mae"], 1),
                round(mt["mape"], 1), round(mt["peak_err_pct"], 1)],
    }), hide_index=True)
    st.caption("MAPE สูงเพราะวันต้นและวันท้ายมีจำนวนคนน้อย (3–4 คน) ความคลาดเคลื่อนเชิงสัมพัทธ์จึงสูงง่าย ควรดู RMSE, R² และค่าพีคประกอบกัน")

    st.markdown("#### ช่วงเชื่อมั่น 95% (จาก Jacobian ของ least squares)")
    if fit["ci"]:
        st.dataframe(pd.DataFrame({
            "พารามิเตอร์": ["β", "γ", "I(0)", "R₀"],
            "ค่าประมาณ": [round(fit["beta"], 3), round(fit["gamma"], 3), round(fit["I0"], 2), round(fit["R0"], 2)],
            "ต่ำสุด": [round(fit["ci"]["beta"][0], 3), round(fit["ci"]["gamma"][0], 3),
                       round(fit["ci"]["I0"][0], 2), round(fit["R0_ci"][0], 2)],
            "สูงสุด": [round(fit["ci"]["beta"][1], 3), round(fit["ci"]["gamma"][1], 3),
                       round(fit["ci"]["I0"][1], 2), round(fit["R0_ci"][1], 2)],
        }), hide_index=True)

    st.markdown("#### วิเคราะห์ R₀ ด้วย 3 วิธี")
    r0_fit = fit["R0"]
    r0_growth, r_rate = sc.r0_from_growth(day_obs, I_obs, fit["gamma"])
    rows = [
        {"วิธี": "1) Fit ทั้งเส้นโค้ง: R₀ = β/γ", "R₀": round(r0_fit, 2)},
        {"วิธี": f"2) อัตราเติบโตช่วงแรก (r = {r_rate:.2f}/วัน): R₀ = 1 + r/γ", "R₀": round(r0_growth, 2)},
    ]
    if total_cases:
        z = total_cases / N_real
        rows.append({"วิธี": f"3) ขนาดสุดท้าย (ติดเชื้อรวม {z*100:.0f}%): z = 1 − e^(−R₀z)",
                     "R₀": round(sc.r0_final_size(z), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    sim_attack = sc.summary(fit["beta"], fit["gamma"], N_real, fit["I0"], 80)["attack_rate"]
    st.warning(
        f"สังเกต: แบบจำลองที่ fit ได้ทำนายว่าจะมีคนติดเชื้อรวม ≈ {sim_attack*100:.0f}% ของโรงเรียน "
        f"แต่ข้อมูลจริงที่นอนพักมี {total_cases/N_real*100:.0f}% (R₀ จากสองวิธีจึงต่างกัน) "
        "สาเหตุที่เป็นไปได้: 'นอนพัก' ไม่เท่ากับ 'ติดเชื้อ' (บางคนป่วยเล็กน้อยหรือไม่แสดงอาการ), "
        "มีช่วงฟักตัวที่ SIR ไม่มี, และผู้ติดเชื้อถูกแยกให้นอนพักจึงแพร่เชื้อน้อยลง"
        if total_cases else "ไม่มีข้อมูลจำนวนติดเชื้อสะสม จึงเปรียบเทียบขนาดสุดท้ายไม่ได้"
    )

# ------------------------------------------------------------------ 5) สรุป
with tab_sum:
    st.subheader("สรุปผล")
    st.markdown(
        f"""
**ข้อมูลที่ใช้:** ไข้หวัดใหญ่โรงเรียนประจำอังกฤษ 1978 (N = 763, 14 วัน)

**วิธีประมาณค่า:** Nonlinear least squares ให้ผลเฉลยเชิงตัวเลขของ ODE (Runge–Kutta) เทียบกับจำนวนผู้ป่วยรายวัน
ปรับ β, γ, I(0) บนสเกล log เพื่อให้เป็นบวก ลองหลายค่าเริ่มต้น และหาช่วงเชื่อมั่นจาก Jacobian

**ผลที่ได้ (ข้อมูลตัวอย่างในแอพ):** β ≈ {fit['beta']:.2f}/วัน, γ ≈ {fit['gamma']:.2f}/วัน
(ระยะแพร่เชื้อ ≈ {1/fit['gamma']:.1f} วัน), **R₀ ≈ {fit['R0']:.2f}** (95% CI {fit['R0_ci'][0]:.2f}–{fit['R0_ci'][1]:.2f}),
R² = {fit['r2']:.3f}, RMSE ≈ {fit['rmse']:.0f} คน, พีคตรงกับข้อมูลประมาณวันที่ 6 (คลาดเคลื่อน {fit['peak_err_pct']:.1f}%)

**ข้อจำกัด:**
1. ใช้ 'จำนวนคนนอนพัก' แทน I — ไม่ใช่ผู้ติดเชื้อทั้งหมด
2. ข้อมูลมีเพียง 14 จุด และ I(0), β มี correlation กัน → ช่วงเชื่อมั่นกว้างพอสมควร
3. SIR ไม่มีช่วงฟักตัว (SEIR อาจเหมาะกว่า) และสมมติให้ประชากรผสมกันสม่ำเสมอ
4. ช่วงเชื่อมั่นจาก Jacobian เป็นการประมาณเชิงเส้น ไม่ได้รวมความผิดพลาดของโครงสร้างแบบจำลอง
5. R₀ ที่ได้จากวิธีขนาดสุดท้ายต่างจากการ fit มาก ซึ่งบอกว่าแบบจำลองยังไม่สมบูรณ์

**ข้อสรุปเชิงนโยบายจากแบบจำลอง:** ลด β (เว้นระยะห่าง) ทำให้พีคต่ำลงและมาช้าลง ส่วนการเพิ่ม γ (รักษา/แยกผู้ป่วยเร็วขึ้น) ลด R₀ ได้เช่นกัน
"""
    )
