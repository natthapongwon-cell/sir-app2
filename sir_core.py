"""ส่วนคำนวณของแบบจำลอง SIR (แยกจากหน้าเว็บเพื่อให้ทดสอบและนำไปใช้ซ้ำได้ง่าย)"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# ข้อมูลจริง: ไข้หวัดใหญ่ในโรงเรียนประจำชายล้วน อังกฤษ ม.ค.-ก.พ. 1978
# (A/H1N1, นักเรียนเสี่ยง 763 คน, มี 512 คนป่วยจนต้องนอนพัก)
# B = จำนวนนักเรียนที่นอนพักอยู่ในแต่ละวัน (ใช้แทนจำนวนผู้ติดเชื้อ I)
# อ้างอิง: Anonymous. Influenza in a boarding school. BMJ 1, 587 (1978)
#          ชุดข้อมูลในแพ็กเกจ R: pomp::bsflu และ epimdr::flu
# ----------------------------------------------------------------------
BSFLU_N = 763
BSFLU_TOTAL_CASES = 512
BSFLU_DAY = np.arange(1, 15)
BSFLU_B = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], float)


def _rhs(t, y, beta, gamma, N):
    S, I, R = y
    new_inf = beta * S * I / N
    return [-new_inf, new_inf - gamma * I, gamma * I]


def simulate(beta, gamma, N, I0, t_eval, R0_init=0.0):
    """แก้ระบบ ODE ของ SIR เชิงตัวเลข (Runge-Kutta, solve_ivp) คืนค่า array [S, I, R]"""
    t_eval = np.asarray(t_eval, float)
    sol = solve_ivp(
        _rhs, (t_eval[0], t_eval[-1]), [N - I0 - R0_init, I0, R0_init],
        t_eval=t_eval, args=(beta, gamma, N), rtol=1e-8, atol=1e-8,
    )
    return sol.y


def summary(beta, gamma, N, I0, days=160):
    """สรุปค่าสำคัญ: R0, จุดพีค, วันที่พีค, สัดส่วนผู้ติดเชื้อทั้งหมด"""
    t = np.linspace(0, days, days * 10 + 1)
    S, I, R = simulate(beta, gamma, N, I0, t)
    return {
        "R0": beta / gamma,
        "peak_I": float(I.max()),
        "peak_day": float(t[I.argmax()]),
        "attack_rate": float(R[-1] / N),
        "t": t, "S": S, "I": I, "R": R,
    }


def fit_sir(t_obs, I_obs, N, final_R=None, final_R_weight=0.0):
    """ประมาณค่า beta, gamma, I0 ด้วยวิธีกำลังสองน้อยที่สุดแบบไม่เชิงเส้น
    (nonlinear least squares) โดยเทียบ I(t) ของแบบจำลองกับข้อมูล

    - ประมาณบนสเกล log เพื่อบังคับให้พารามิเตอร์เป็นบวก
    - ลองหลายค่าเริ่มต้นแล้วเลือกผลที่ cost ต่ำสุด (กันติด local minimum)
    - final_R / final_R_weight: (ทางเลือก) เพิ่มเงื่อนไขให้จำนวนผู้ติดเชื้อสะสมช่วงท้าย
      ใกล้ค่าที่สังเกตได้จริง
    """
    t_obs = np.asarray(t_obs, float)
    I_obs = np.asarray(I_obs, float)

    def resid(p):
        beta, gamma, I0 = np.exp(p)
        S, I, R = simulate(beta, gamma, N, I0, t_obs)
        r = I - I_obs
        if final_R is not None and final_R_weight > 0:
            t_long = np.linspace(t_obs[0], t_obs[0] + 120, 241)
            Rf = simulate(beta, gamma, N, I0, t_long)[2][-1]
            r = np.append(r, final_R_weight * (Rf - final_R))
        return r

    best = None
    for b0 in (0.5, 1.0, 2.0, 3.0):
        for g0 in (0.2, 0.5, 1.0):
            try:
                r = least_squares(resid, np.log([b0, g0, max(I_obs[0], 1.0)]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    beta, gamma, I0 = np.exp(best.x)

    # ช่วงเชื่อมั่น 95% จากเมทริกซ์ Jacobian (ประมาณแบบเชิงเส้นบนสเกล log)
    n_data = len(I_obs)
    dof = max(n_data - 3, 1)
    s2 = np.sum(best.fun[:n_data] ** 2) / dof
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * s2
        se = np.sqrt(np.diag(cov))
        var_r0 = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        r0_lo = (beta / gamma) * np.exp(-1.96 * np.sqrt(var_r0))
        r0_hi = (beta / gamma) * np.exp(1.96 * np.sqrt(var_r0))
        ci = {k: (v * np.exp(-1.96 * s), v * np.exp(1.96 * s))
              for k, v, s in zip(("beta", "gamma", "I0"), (beta, gamma, I0), se)}
    except np.linalg.LinAlgError:
        ci, r0_lo, r0_hi = {}, np.nan, np.nan

    pred = simulate(beta, gamma, N, I0, t_obs)[1]
    return {"beta": beta, "gamma": gamma, "I0": I0, "R0": beta / gamma,
            "R0_ci": (r0_lo, r0_hi), "ci": ci, "pred": pred, **metrics(I_obs, pred)}


def metrics(obs, pred):
    """ตัวชี้วัดความถูกต้องของการ fit"""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    res = obs - pred
    ss_res, ss_tot = np.sum(res ** 2), np.sum((obs - obs.mean()) ** 2)
    mask = obs > 0
    return {
        "rmse": float(np.sqrt(np.mean(res ** 2))),
        "mae": float(np.mean(np.abs(res))),
        "r2": float(1 - ss_res / ss_tot),
        "mape": float(np.mean(np.abs(res[mask] / obs[mask])) * 100),
        "peak_err_pct": float((pred.max() - obs.max()) / obs.max() * 100),
    }


def r0_final_size(attack_rate):
    """R0 จากสมการขนาดสุดท้าย: z = 1 - exp(-R0 z)  =>  R0 = -ln(1-z)/z"""
    z = float(attack_rate)
    return -np.log(1 - z) / z


def r0_from_growth(t, I, gamma, n_points=4):
    """R0 จากอัตราเติบโตช่วงแรก: I ~ exp(r t)  =>  R0 = 1 + r/gamma (สำหรับ SIR)"""
    r = np.polyfit(np.asarray(t)[:n_points], np.log(np.asarray(I)[:n_points]), 1)[0]
    return 1 + r / gamma, r
