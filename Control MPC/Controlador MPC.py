
import faulthandler
faulthandler.enable()
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from scipy.linalg import expm
# 1.  PLANTA
class P:
    L, W, H_inv = 0.85, 0.45, 0.40
    V = L * W * H_inv
    rho, cp_air = 1.204, 1005.0
    Q_max  = 12.441      # W   potencia real del calentador a PWM 255
    UA_tot = 0.6006      # W/K pérdida térmica global
    C_air  = 483.76      # J/K capacidad calorífica efectiva
    A_hum_max = 0.135    # %HR/s a u_h = 127 PWM
    C_PSYCHRO = 3.229
    ANG_MIN, ANG_MAX = 0.0, 25.0
    K_VENT_ANGLES = np.array([0, 3, 6, 9, 12, 16, 19, 22, 25], dtype=float)
    K_VENT_TABLE = np.array([1.890e-04, 2.594e-04, 3.299e-04, 4.003e-04, 4.708e-04, 5.647e-04, 6.351e-04, 7.056e-04, 7.760e-04])
    K_HUM_ANGLES = np.array([0.0, 6.0, 12.0, 19.0, 25.0])
    K_HUM_TABLE  = np.array([9.56e-4, 2.34e-3, 2.53e-3, 3.99e-3, 3.99e-3])    # [1/s]
    U_MAX = np.array([255.0, 127.0, ANG_MAX])   # [u_c, u_h, u_v]
    # Límites de velocidad (red de seguridad, en unidades reales por paso)
    SLEW = np.array([40.0, 20.0, 8.0])
def presion_vapor_saturacion(T):
    return 6.112 * np.exp(17.67 * T / (T + 243.5))
def dlnes_dT(T):
    return 17.67 * 243.5 / (T + 243.5) ** 2
PSYCHRO_FISICO = True
def f_model(x, u, d, eta_heat=1.0, eta_hum=1.0, eta_vent=1.0):
    T_i, H_i = x
    u_c = np.clip(u[0], 0.0, 255.0)
    u_h = np.clip(u[1], 0.0, 127.0)
    u_v = np.clip(u[2], P.ANG_MIN, P.ANG_MAX)
    T_e, H_e = d
    k_vent_nominal = max(0.0, np.interp(u_v, P.K_VENT_ANGLES, P.K_VENT_TABLE))
    k_vent = k_vent_nominal * eta_vent
    Q_heater = (P.Q_max * eta_heat) * (u_c / 255.0)
    Q_loss = P.UA_tot * (T_i - T_e)
    mdot_air = k_vent * P.rho * P.V
    Q_vent = mdot_air * P.cp_air * (T_e - T_i)
    dTi = (Q_heater - Q_loss + Q_vent) / P.C_air
    H_e_eq = H_e * presion_vapor_saturacion(T_e) / presion_vapor_saturacion(T_i)
    k_hum_nominal = np.interp(u_v, P.K_HUM_ANGLES, P.K_HUM_TABLE)
    k_hum = k_hum_nominal * eta_vent
    dH_hum = (P.A_hum_max * eta_hum) * (u_h / 127.0)
    dH_vent = k_hum * (H_e_eq - H_i)
    c_psy = H_i * dlnes_dT(T_i) if PSYCHRO_FISICO else P.C_PSYCHRO
    dH_psychro = -c_psy * dTi
    dHi = dH_hum + dH_vent + dH_psychro
    if H_i >= 100.0:
        dHi = min(0.0, dHi)
    elif H_i <= 0.0:
        dHi = max(0.0, dHi)
    return np.array([dTi, dHi])
def rk4_step(x, u, d, dt, eta_heat=1.0, eta_hum=1.0, eta_vent=1.0):
    k1 = f_model(x, u, d, eta_heat, eta_hum, eta_vent)
    k2 = f_model(x + dt/2 * k1, u, d, eta_heat, eta_hum, eta_vent)
    k3 = f_model(x + dt/2 * k2, u, d, eta_heat, eta_hum, eta_vent)
    k4 = f_model(x + dt   * k3, u, d, eta_heat, eta_hum, eta_vent)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_new[1] = np.clip(x_new[1], 0.0, 100.0)
    return x_new
def linearize(x0, u0, d0, eps=1e-4):
    """
    Jacobiano numérico de f_model en (x0, u0, d0).
    Devuelve (f0, A_c, B_c, Bd_c) en tiempo continuo.
    """
    f0 = f_model(x0, u0, d0)
    A_c = np.zeros((2, 2))
    for i in range(2):
        xp = x0.copy(); xp[i] += eps
        A_c[:, i] = (f_model(xp, u0, d0) - f0) / eps
    UMAX = P.U_MAX
    B_c = np.zeros((2, 3))
    for j in range(3):
        h = eps if u0[j] + eps <= UMAX[j] else -eps
        up = u0.copy();
        up[j] += h
        B_c[:, j] = (f_model(x0, up, d0) - f0) / h
    Bd_c = np.zeros((2, 2))
    for k in range(2):
        dp = d0.copy(); dp[k] += eps
        Bd_c[:, k] = (f_model(x0, u0, dp) - f0) / eps
    return f0, A_c, B_c, Bd_c
# 2.  CHEQUEO DE FACTIBILIDAD DEL SETPOINT
def setpoint_factible(ref, d):
    """
    ¿Existe algún u ∈ [0,255]×[0,127]×[0,25] tal que f(x_ref, u, d) = 0 ?
    """
    T_r, H_r = ref
    T_e, H_e = d
    x_ref = np.array([T_r, H_r], dtype=float)
    for ang in P.K_VENT_ANGLES:
        k_vent = max(0.0, np.interp(ang, P.K_VENT_ANGLES, P.K_VENT_TABLE))
        Q_vent = k_vent * P.rho * P.V * P.cp_air * (T_e - T_r)
        u_c = 255.0 * (P.UA_tot * (T_r - T_e) - Q_vent) / P.Q_max
        if not (0.0 <= u_c <= 255.0):
            continue
        dH_min = f_model(x_ref, np.array([u_c, 0.0,   ang]), np.asarray(d, dtype=float))[1]
        dH_max = f_model(x_ref, np.array([u_c, 127.0, ang]), np.asarray(d, dtype=float))[1]
        if dH_min <= 0.0 <= dH_max:
            return True, ang, u_c
    return False, None, None
# 3.  MPC:  QP CVXPY SOBRE LA LINEALIZACIÓN SUCESIVA
Ts = 10         # Período de muestreo = 10 s
N  = 30        # Horizonte de predicción
Nc = 15         # Horizonte de control: 15
# Pesos. Orden de u: [calentador, humidificador, ventana]
# Pesos equilibrados: permiten anticipación sin saturación reactiva
Q_w  = np.diag([12.0, 1.2])            # Prioridad en temperatura
R_w  = np.diag([0.01, 0.01, 0.05])     # Penalización leve al consumo
dR_w = np.diag([35.0, 35.0, 100.0])    # Evita saltos bruscos entre pasos con la ventana
W_VENT = 0.5
def solve_mpc(x_t, u_prev, ref, d_seq):
    x_t, u_prev, ref, d_seq = np.asarray(x_t, dtype=float), np.asarray(u_prev, dtype=float), np.asarray(ref, dtype=float), np.asarray(d_seq, dtype=float)
    d0 = d_seq[0]
    f0, A_c, B_c, Bd_c = linearize(x_t, u_prev, d0)
    # Discretización exacta (ZOH) de  ẋ = A x + B u + c  con UNA exponencial aumentada:
    n = A_c.shape[0]
    E = expm(np.block([[A_c * Ts, np.eye(n) * Ts], [np.zeros((n, 2 * n))]]))
    Ad, Gam = E[:n, :n], E[:n, n:]
    Bu = Gam @ B_c @ np.diag(P.U_MAX)
    c0 = Gam @ (f0 - A_c @ x_t - B_c @ u_prev)
    Gd = Gam @ Bd_c
    x, u = cp.Variable((2, N + 1)), cp.Variable((3, N))
    u_prev_n, slew_n = u_prev / P.U_MAX, P.SLEW / P.U_MAX
    cost_state = cost_input = cost_rate = cost_terminal = 0
    cons = [x[:, 0] == x_t]
    for k in range(N):
        c_k = c0 + Gd @ (d_seq[k] - d0)
        cons += [x[:, k + 1] == Ad @ x[:, k] + Bu @ u[:, k] + c_k]
        cost_state += cp.quad_form(x[:, k] - ref, Q_w)
        if k < Nc:
            cost_input += cp.quad_form(u[:, k], R_w)
            u_ant = u[:, k - 1] if k > 0 else u_prev_n
            cost_rate += cp.quad_form(u[:, k] - u_ant, dR_w)
            cons += [u[:, k] >= 0, u[:, k] <= 1, cp.abs(u[:, k] - u_ant) <= slew_n]
        else:
            cons += [u[:, k] == u[:, Nc - 1]]
    cost_terminal = cp.quad_form(x[:, N] - ref, Q_w)
    cost_vent = W_VENT * cp.sum(u[2, :])
    cost_total_expr = cost_state + cost_input + cost_rate + cost_terminal + cost_vent
    prob = cp.Problem(cp.Minimize(cost_total_expr), cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=100000, polish=True, adaptive_rho=True)
        solver_used = "OSQP"
    except cp.error.SolverError:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        solver_used = "CLARABEL"

    if prob.status in ("infeasible", "unbounded") or u.value is None or x.value is None:
        raise ValueError(f"MPC no resuelto (status={prob.status})")
    stats = prob.solver_stats
    if stats is not None:
        solve_time, num_iters = stats.solve_time, stats.num_iters
        if solver_used == "OSQP":
            try:
                extra = stats.extra_stats
                if hasattr(extra, "info"):
                    if hasattr(extra.info, "pri_res"): primal_residual = float(extra.info.pri_res)
                    if hasattr(extra.info, "dua_res"): dual_residual = float(extra.info.dua_res)
            except Exception: pass

    u_pred_n = np.clip(np.asarray(u.value, dtype=float).copy(), 0.0, 1.0)
    u_first_n = u_pred_n[:, 0].copy()
    u_pred_real = u_pred_n * P.U_MAX[:, None]
    u_real = u_pred_real[:, 0].copy()
    x_pred = np.asarray(x.value, dtype=float).copy()
    prediction_error = ref[:, None] - x_pred
    cost_state_val = cost_input_val = cost_rate_val = cost_terminal_val = 0.0
    for k in range(N):
        e = ref - x_pred[:, k]
        cost_state_val += float(e.T @ Q_w @ e)
    for k in range(N):
        if k < Nc:
            uk = u_pred_n[:, k]
            cost_input_val += float(uk.T @ R_w @ uk)
            u_ant = u_prev_n if k == 0 else u_pred_n[:, k - 1]
            du = uk - u_ant
            cost_rate_val += float(du.T @ dR_w @ du)
    e_terminal = ref - x_pred[:, N]
    cost_terminal_val = float(e_terminal.T @ Q_w @ e_terminal)
    cost_state_total = cost_state_val + cost_terminal_val
    cost_vent_val = float(W_VENT * np.sum(u_pred_n[2, :]))
    cost_reconstructed = cost_state_val + cost_input_val + cost_rate_val + cost_terminal_val + cost_vent_val
    cost_solver = float(prob.value)
    cost_gap = cost_solver - cost_reconstructed
    abs_error = np.abs(prediction_error)
    max_abs_error = float(np.max(abs_error))
    max_temp_error = float(np.max(abs_error[0, :]))
    max_hum_error = float(np.max(abs_error[1, :]))
    final_error = prediction_error[:, -1].copy()
    info = {
        "status": prob.status, "solver": solver_used,
        "solve_time_s": float(solve_time) if solve_time is not None else None,
        "num_iters": int(num_iters) if num_iters is not None else None,
        "primal_residual": primal_residual, "dual_residual": dual_residual,
        "cost_total": cost_solver, "cost_state": float(cost_state_total),
        "cost_state_running": float(cost_state_val), "cost_terminal": float(cost_terminal_val),
        "cost_input": float(cost_input_val), "cost_rate": float(cost_rate_val),
        "cost_reconstructed": float(cost_reconstructed), "cost_consistency_error": float(cost_gap),
        "u_first_normalized": u_first_n.copy(), "u_first_real": u_real.copy(),
        "u_pred_normalized": u_pred_n.copy(), "u_pred_real": u_pred_real.copy(),
        "predicted_x": x_pred.copy(), "predicted_error": prediction_error.copy(),
        "predicted_final_state": x_pred[:, -1].copy(), "predicted_final_error": final_error.copy(),
        "predicted_max_abs_error": max_abs_error, "predicted_max_temp_error": max_temp_error,
        "predicted_max_hum_error": max_hum_error, "x_current": x_t.copy(),
        "u_previous_real": u_prev.copy(), "u_previous_normalized": u_prev_n.copy(),
        "reference": ref.copy(), "d_sequence": d_seq.copy(),
        "N": int(N), "Nc": int(Nc), "Ts": float(Ts),
        "U_MAX": np.asarray(P.U_MAX).copy(), "SLEW": np.asarray(P.SLEW).copy(),
    }
    return u_real, info
# 4.  ESCENARIOS POR FASE DE CULTIVO
ESC_BASE = {
    # INVIERNO PACEÑO
    1: ("Invierno Noche - Frío Suave Germinación", 3.0, 35.0, (18.0, 70.0), (8.0, 45.0)),
    2: ("Invierno Mañana - Aire Seco Germinación", 8.0, 28.0, (20.0, 65.0), (10.0, 40.0)),
    3: ("Invierno Día Sol - Veg. Desarrollo", 15.0, 32.0, (22.0, 60.0), (14.0, 50.0)),
    4: ("Invierno Tarde - Templado Floración", 12.0, 30.0, (24.0, 55.0), (16.0, 40.0)),
    5: ("Invierno Madrugada - Frío Cuajado", 2.0, 40.0, (19.0, 65.0), (6.0, 45.0)),
    # VERANO / LLUVIAS
    6: ("Verano Lluvia - Humedad Max Germinación", 13.0, 80.0, (18.0, 75.0), (12.0, 75.0)),
    7: ("Verano Nublado - Veg. Desarrollo", 16.0, 70.0, (21.0, 70.0), (15.0, 65.0)),
    8: ("Verano Mediodía - Sol Pico Térmico", 21.0, 55.0, (25.0, 60.0), (20.0, 50.0)),
    9: ("Verano Noche - Lluvia Continua Floración", 11.0, 85.0, (17.0, 65.0), (12.0, 75.0)),
    10: ("Verano Tarde - Tormenta Cuajado Fruto", 15.0, 75.0, (23.0, 65.0), (16.0, 70.0)),
    # PRIMAVERA
    11: ("Primavera Día - Cálido Germinación", 18.0, 42.0, (22.0, 70.0), (15.0, 50.0)),
    12: ("Primavera Noche - Templado Veg. Desarrol.", 7.0, 52.0, (18.0, 65.0), (10.0, 55.0)),
    13: ("Primavera Mediodía - Sol Intenso Florac.", 20.0, 38.0, (25.0, 55.0), (19.0, 45.0)),
    14: ("Primavera Tarde - Lechuga Crespa", 16.0, 45.0, (18.0, 70.0), (14.0, 52.0)),
    # OTOÑO
    15: ("Otoño Mañana - Fresco Germinativo", 6.0, 48.0, (19.0, 68.0), (8.0, 42.0)),
    16: ("Otoño Sol - Seco Veg. Desarrollo", 14.0, 35.0, (20.0, 60.0), (12.0, 40.0)),
    17: ("Otoño Noche - Fresco Floración", 5.0, 55.0, (18.0, 60.0), (9.0, 52.0)),
    # TRANSITORIOS
    18: ("Transitorio - Ventilación de Mañana", 8.0, 40.0, (20.0, 65.0), (11.0, 44.0)),
    19: ("Transitorio - Acumulación Térmica Sol", 19.0, 35.0, (18.0, 70.0), (25.0, 45.0)),
    20: ("Transitorio - Riego Reciente", 13.0, 75.0, (22.0, 60.0), (16.0, 88.0)),
    21: ("Transitorio - Ventilación de Mañana", 8.0, 40.0, (24.0, 65.0), (15.0, 60.0)),
    22: ("Transitorio - Acumulación Térmica Sol", 19.0, 35.0, (25.0, 70.0), (15.0, 60.0)),
    23: ("Transitorio - Riego Reciente", 13.0, 75.0, (26.0, 75.0), (15.0, 60.0)),
    # VARIANTES
    24: ("Invierno Noche - Frío Suave Germinación", 5.0, 40.0, (18.0, 70.0), (8.0, 45.0)),
    25: ("Invierno Mañana - Aire Seco Germinación", 9.0, 15.0, (20.0, 65.0), (10.0, 40.0)),
    26: ("Invierno Día Sol - Veg. Desarrollo", 16.0, 45.0, (22.0, 60.0), (14.0, 50.0)),
    27: ("Invierno Tarde - Templado Floración", 15.0, 40.0, (24.0, 55.0), (16.0, 40.0)),
    28: ("Invierno Madrugada - Frío Cuajado", 8.0, 50.0, (19.0, 65.0), (6.0, 45.0)),
    29: ("Verano Lluvia - Humedad Max Germinación", 14.0, 65.0, (20.0, 75.0), (12.0, 75.0)),
    30: ("Verano Nublado - Veg. Desarrollo", 17.0, 65.0, (21.0, 70.0), (15.0, 65.0)),
    31: ("Verano Mediodía - Sol Pico Térmico", 20.0, 50.0, (25.0, 60.0), (20.0, 50.0)),
    32: ("Verano Noche - Lluvia Continua Floración", 12.0, 80.0, (17.0, 65.0), (12.0, 75.0)),
    33: ("Verano Tarde - Tormenta Cuajado Fruto", 12.0, 65.0, (23.0, 65.0), (16.0, 70.0)),
    34: ("Primavera Día - Cálido Germinación", 16.0, 45.0, (22.0, 70.0), (15.0, 50.0)),
    35: ("Primavera Noche - Templado Veg. Desarrol.", 7.0, 60.0, (18.0, 65.0), (10.0, 55.0)),
    36: ("Primavera Mediodía - Sol Intenso Florac.", 22.0, 45.0, (25.0, 55.0), (19.0, 45.0)),
    37: ("Primavera Tarde - Lechuga Crespa", 18.0, 46.0, (18.0, 70.0), (14.0, 52.0)),
    38: ("Otoño Mañana - Fresco Germinativo", 8.0, 55.0, (19.0, 68.0), (8.0, 42.0)),
    39: ("Otoño Sol - Seco Veg. Desarrollo", 15.0, 40.0, (20.0, 60.0), (12.0, 40.0)),
    40: ("Otoño Noche - Fresco Floración", 8.0, 50.0, (18.0, 60.0), (9.0, 52.0)),
    41: ("Transitorio - Ventilación de Mañana", 10.0, 45.0, (20.0, 65.0), (11.0, 44.0)),
    42: ("Transitorio - Acumulación Térmica Sol", 20.0, 40.0, (18.0, 70.0), (25.0, 45.0)),
    43: ("Transitorio - Riego Reciente", 12.0, 70.0, (22.0, 60.0), (16.0, 88.0)),
    44: ("Transitorio - Ventilación de Mañana", 8.0, 35.0, (24.0, 65.0), (15.0, 60.0)),
    45: ("Transitorio - Acumulación Térmica Sol", 18.0, 38.0, (25.0, 70.0), (15.0, 60.0)),
    46: ("Transitorio - Riego Reciente", 14.0, 70.0, (26.0, 75.0), (15.0, 60.0)),
}
# ── 51–63: familia "calentamiento con exterior húmedo"──────
ESC_FAMILIA = {
    51: ("Calent. ext húmedo T0=10 H0=65 He=62", 16.0, 62.0, (25.0, 65.0), (10.0, 65.0)),
    54: ("Calent. ext húmedo T0=14 H0=55 He=62", 16.0, 62.0, (25.0, 65.0), (14.0, 55.0)),
    55: ("Calent. ext húmedo T0=14 H0=55 He=68", 16.0, 68.0, (25.0, 65.0), (14.0, 55.0)),
    57: ("Calent. ext húmedo T0=14 H0=65 He=62", 16.0, 62.0, (25.0, 65.0), (14.0, 65.0)),
    60: ("Calent. ext húmedo T0=18 H0=55 He=62", 16.0, 62.0, (25.0, 65.0), (18.0, 55.0)),
    63: ("Calent. ext húmedo T0=18 H0=65 He=62", 16.0, 62.0, (25.0, 65.0), (18.0, 65.0)),
}
ESC_NUEVOS = {
    70: ("T casi en ref - Interior seco He=55", 14.0, 55.0, (25.0, 65.0), (23.0, 50.0)),
    71: ("T casi en ref - Interior seco He=62", 17.0, 62.0, (25.0, 65.0), (24.0, 52.0)),
    72: ("T en ref - Interior muy seco", 20.0, 60.0, (25.0, 65.0), (24.5, 45.0)),
    73: ("T en ref - Interior seco, ref 23 °C", 18.0, 62.0, (23.0, 65.0), (23.0, 50.0)),
    74: ("T en ref - Interior muy seco, ext seco", 12.0, 40.0, (22.0, 65.0), (22.0, 42.0)),
    75: ("T en ref - Interior húmedo, ext seco", 15.0, 50.0, (25.0, 65.0), (24.0, 75.0)),
}
ESCENARIOS = {**ESC_BASE, **ESC_FAMILIA, **ESC_NUEVOS}
T_STEPS = 270
CSV_PATH = 'datos_simulacion_v5.csv'
PERIODO_S = 24 * 3600.0
def perturbaciones(t, var_T, var_H):
    w = 2.0 * np.pi / PERIODO_S
    T_e = np.clip(var_T + 1 * np.cos(w * t), 0.0, 27)
    H_e = np.clip(var_H + 1 * np.sin(w * t), 0.0, 100.0)
    return np.array([T_e, H_e])
def run_escenario(n, nombre, var_T, var_H, ref, x0, eta_heat=1.0, eta_hum=1.0, eta_vent=1.0, verbose=True):

    if verbose and (eta_heat != 1.0 or eta_hum != 1.0 or eta_vent != 1.0):
        print(
            f"PLANTA PERTURBADA: Heater={eta_heat * 100:.0f}%, Hum={eta_hum * 100:.0f}%, Vent={eta_vent * 100:.0f}%")
    ref = np.array(ref, dtype=float)
    x   = np.array(x0,  dtype=float)

    d0 = perturbaciones(0.0, var_T, var_H)
    ok, ang_eq, uc_eq = setpoint_factible(ref, d0)

    if verbose:
        print("\n" + "=" * 72)
        print(f"  ESCENARIO {n}: {nombre}")
        print(f"  Setpoint (T,H) = ({ref[0]}, {ref[1]})   "
              f"x0 = ({x0[0]}, {x0[1]})   Exterior ≈ ({var_T}°C, {var_H}%HR)")
        if ok:
            print(f"  Equilibrio factible: ventana≈{ang_eq:.0f}°, heater≈{uc_eq:.0f}/255")
        else:
            print("  *** AVISO DE FACTIBILIDAD ***")
            print("  No existe combinación de actuadores que mantenga este")
            print(f"  setpoint con el exterior actual. El recinto cerrado tiende a")
            #print(f"  {P.H_EQ_NAT:.1f} %HR y no hay deshumidificador: sólo se puede")
            print("  secar ventilando hacia H_e. El MPC hará lo mejor posible,")
            print("  pero quedará un error permanente.")
        print("=" * 72)
    X_true = np.zeros((2, T_STEPS + 1)); X_true[:, 0] = x
    X_meas = np.zeros((2, T_STEPS + 1)); X_meas[:, 0] = x
    U = np.zeros((3, T_STEPS))          # [u_c, u_h, u_v] reales
    D = np.zeros((2, T_STEPS + 1))
    u_prev = np.array([0.0, 0.0, 0.0])
    ALPHA_T = 0.35
    ALPHA_H = 0.60

    for t in range(T_STEPS):
        t_abs = t * Ts
        D[:, t] = perturbaciones(t_abs, var_T, var_H)
        d_seq = np.array([perturbaciones(t_abs + k * Ts, var_T, var_H)
                          for k in range(N)])
        try:
            u_prev, info = solve_mpc(X_meas[:, t], u_prev, ref, d_seq)
        except ValueError as e:
            print(f"  Error en el paso {t}: {e}")
            X_true = X_true[:, :t+1]; X_meas = X_meas[:, :t+1]; U = U[:, :t]; D = D[:, :t+1]
            break
        U[:, t] = u_prev
        X_true[:, t+1] = rk4_step(X_true[:, t], u_prev, D[:, t], Ts, eta_heat, eta_hum, eta_vent)
        noise_T = np.random.normal(0, 0.02)
        noise_H = np.random.normal(0, 0.05)
        X_meas[0, t+1] = ALPHA_T * X_meas[0, t] + (1.0 - ALPHA_T) * X_true[0, t+1] + noise_T
        X_meas[1, t+1] = np.clip(ALPHA_H * X_meas[1, t] + (1.0 - ALPHA_H) * X_true[1, t+1] + noise_H, 0.0, 100.0)

        if verbose and t % 60 == 0:
            T_i, H_i = X_meas[:, t + 1]
            e_T = ref[0] - T_i
            e_H = ref[1] - H_i
            T_e, H_e = d_seq[0]
            status = info.get("status", "N/A")
            solver = info.get("solver", "N/A")
            solve_time = info.get("solve_time_s", None)
            num_iters = info.get("num_iters", None)
            cost_total = info.get("cost_total", np.nan)
            cost_state = info.get("cost_state", np.nan)
            cost_terminal = info.get("cost_terminal", np.nan)
            cost_input = info.get("cost_input", np.nan)
            cost_rate = info.get("cost_rate", np.nan)
            cost_gap = info.get(
                "cost_consistency_error",
                np.nan
            )
            u_first_n = info.get(
                "u_first_normalized",
                np.full(3, np.nan)
            )
            u_first_real = info.get(
                "u_first_real",
                np.full(3, np.nan)
            )
            predicted_final = info.get(
                "predicted_final_state",
                np.full(2, np.nan)
            )
            predicted_final_error = info.get(
                "predicted_final_error",
                np.full(2, np.nan)
            )
            max_error = info.get(
                "predicted_max_abs_error",
                np.nan
            )
            max_T_error = info.get(
                "predicted_max_temp_error",
                np.nan
            )
            max_H_error = info.get(
                "predicted_max_hum_error",
                np.nan
            )
            primal = info.get(
                "primal_residual",
                None
            )
            dual = info.get(
                "dual_residual",
                None
            )
            print(
                "\n"
                + "-" * 72
                + f"\n  [t={t * Ts / 60:5.1f} min]"
                + "\n"
                + "-" * 72
            )
            print(
                f"  ESTADO\n"
                f"    Interior        : "
                f"T={T_i:7.2f} °C | "
                f"H={H_i:7.2f} %HR\n"
                f"    Setpoint        : "
                f"T={ref[0]:7.2f} °C | "
                f"H={ref[1]:7.2f} %HR\n"
                f"    Error actual    : "
                f"ΔT={e_T:+7.2f} °C | "
                f"ΔH={e_H:+7.2f} %HR\n"
                f"    Exterior        : "
                f"T={T_e:7.2f} °C | "
                f"H={H_e:7.2f} %HR"
            )
            print(
                "\n  CONTROL"
                f"\n    Aplicado real   : "
                f"heater={u_prev[0]:6.1f} | "
                f"humid={u_prev[1]:6.1f} | "
                f"vent={u_prev[2]:6.1f}°"
                f"\n    MPC real        : "
                f"heater={u_first_real[0]:6.1f} | "
                f"humid={u_first_real[1]:6.1f} | "
                f"vent={u_first_real[2]:6.1f}°"
                f"\n    MPC normalizado : "
                f"heater={u_first_n[0]:6.3f} | "
                f"humid={u_first_n[1]:6.3f} | "
                f"vent={u_first_n[2]:6.3f}"
            )
            print(
                "\n  PREDICCIÓN AL FINAL DEL HORIZONTE"
                f"\n    Estado previsto : "
                f"T={predicted_final[0]:7.2f} °C | "
                f"H={predicted_final[1]:7.2f} %HR"
                f"\n    Error previsto  : "
                f"ΔT={predicted_final_error[0]:+7.2f} °C | "
                f"ΔH={predicted_final_error[1]:+7.2f} %HR"
                f"\n    Máx |error|     : "
                f"{max_error:7.3f}"
                f"\n    Máx |error T|   : "
                f"{max_T_error:7.3f} °C"
                f"\n    Máx |error H|   : "
                f"{max_H_error:7.3f} %HR"
            )
            print(
                "\n  SOLVER"
                f"\n    Estado          : {status}"
                f"\n    Solver          : {solver}"
                f"\n    Tiempo          : "
                f"{solve_time:.6f} s"
                if solve_time is not None
                else "\n    Tiempo          : N/A"
            )
            if num_iters is not None:
                print(
                    f"    Iteraciones     : {num_iters}"
                )
            if primal is not None:
                print(
                    f"    Primal residual : "
                    f"{primal:.3e}"
                )
            if dual is not None:
                print(
                    f"    Dual residual   : "
                    f"{dual:.3e}"
                )

            print(
                "\n  COSTOS"
                f"\n    Total           : {cost_total:12.6f}"
                f"\n    Estado          : {cost_state:12.6f}"
                f"\n    Terminal        : {cost_terminal:12.6f}"
                f"\n    Entrada         : {cost_input:12.6f}"
                f"\n    Rate            : {cost_rate:12.6f}"
                f"\n    Consistencia    : {cost_gap:+.3e}"
            )

            print(
                "-" * 72
            )

    n_ok = U.shape[1]
    D[:, n_ok] = perturbaciones(n_ok * Ts, var_T, var_H)
    if verbose and n_ok > 0:
        eT = X_meas[0, 1:n_ok+1] - ref[0]
        eH = X_meas[1, 1:n_ok+1] - ref[1]
        print(f"  RMSE T (sensor) = {np.sqrt(np.mean(eT**2)):.3f} °C   "
              f"RMSE H (sensor) = {np.sqrt(np.mean(eH**2)):.3f} %HR   "
              f"(sesgo: {np.mean(eT):+.2f} °C / {np.mean(eH):+.2f} %HR)")

    return X_meas, U, D, ref, X_true
# 5.  GRÁFICAS Y LOG
def plot_escenario(n, nombre, X_meas, U, D, ref, X_true=None):
    n_ok = U.shape[1]
    t_min = np.arange(n_ok + 1) * Ts / 60.0
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t_min, X_meas[0, :n_ok+1], label='T_in (sensor medido)', color='tab:blue', lw=1.5)
    if X_true is not None:
        ax[0].plot(t_min, X_true[0, :n_ok+1], ':', label='T_in (aire real)', color='navy', alpha=0.6)
    ax[0].plot(t_min, D[0, :n_ok+1], '--', label='T_e exterior', color='tab:gray')
    ax[0].axhline(ref[0], color='k', ls=':', lw=1, label=f'ref {ref[0]} °C')
    ax[0].set_ylabel('Temperatura [°C]'); ax[0].grid(True)
    ax[0].legend(fontsize=8, loc='best')

    ax[1].plot(t_min, X_meas[1, :n_ok+1], label='H_in (sensor medido)', color='tab:blue', lw=1.5)
    if X_true is not None:
        ax[1].plot(t_min, X_true[1, :n_ok+1], ':', label='H_in (aire real)', color='navy', alpha=0.6)
    ax[1].plot(t_min, D[1, :n_ok+1], '--', label='H_e exterior', color='tab:gray')
    ax[1].axhline(ref[1], color='k', ls=':', lw=1, label=f'ref {ref[1]} %HR')
    ax[1].set_ylabel('Humedad [%HR]'); ax[1].grid(True)
    ax[1].legend(fontsize=8, loc='best')
    # Actuadores en % de su propio rango real -> comparables en un solo eje
    ax[2].step(t_min[:n_ok], U[2, :] / P.ANG_MAX * 100, where='post',
               label=f'Ventana (0-{P.ANG_MAX:.0f}°)', color='tab:green')
    ax[2].step(t_min[:n_ok], U[0, :] / 255.0 * 100, where='post',
               label='Calentador (0-255)', color='tab:red')
    ax[2].step(t_min[:n_ok], U[1, :] / 127.0 * 100, where='post',
               label='Humidificador (0-127)', color='goldenrod')
    ax[2].set_ylabel('% de rango'); ax[2].set_xlabel('Tiempo [min]')
    ax[2].set_ylim(-5, 105); ax[2].grid(True)
    ax[2].legend(fontsize=8, loc='best')

    fig.suptitle(f'Escenario {n} (v5 Sensor Lag) — {nombre}\n'
                 f'Ref: T={ref[0]} °C, H={ref[1]} %HR', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
COLUMNAS = ['temp_ref', 'hum_ref', 'temp_in', 'hum_in', 'T_out', 'H_out',
            'window', 'heater', 'humidifier', 'error_temp', 'error_hum',
            'scenario_id', 'scenario_group_id']
def log_csv(X_meas, U, D, ref, *, scenario_id, scenario_group_id, path=CSV_PATH):
    n_ok = U.shape[1]
    temp_in = X_meas[0, :n_ok]
    hum_in = X_meas[1, :n_ok]

    datos = pd.DataFrame({
        'temp_ref': np.full(n_ok, ref[0]),
        'hum_ref': np.full(n_ok, ref[1]),
        'temp_in': temp_in,
        'hum_in': hum_in,
        'T_out': D[0, :n_ok],
        'H_out': D[1, :n_ok],
        'window': U[2, :],
        'heater': U[0, :],
        'humidifier': U[1, :],
        'error_temp': ref[0] - temp_in,
        'error_hum': ref[1] - hum_in,
        'scenario_id': np.full(n_ok, scenario_id),
        'scenario_group_id': np.full(n_ok, scenario_group_id)
    })
    datos.to_csv(path, mode='a', header=False, index=False)
# 6.  MAIN
if __name__ == '__main__':
    import sys
    MODO_RAPIDO = '--fast' in sys.argv
    COLUMNAS = ['temp_ref', 'hum_ref', 'temp_in', 'hum_in', 'T_out', 'H_out',
                'window', 'heater', 'humidifier', 'error_temp', 'error_hum',
                'scenario_id', 'scenario_group_id']
    pd.DataFrame(columns=COLUMNAS).to_csv(CSV_PATH, index=False)

    if MODO_RAPIDO:
        print("⚡ MODO RÁPIDO ACTIVADO (--fast): Generando subconjunto representativo de 26 escenarios")
        ESC_RAPIDOS = [1, 3, 6, 8, 10, 11, 13, 15, 18, 19, 20, 22, 23, 25, 27, 30, 31, 36, 40, 42, 51, 54, 57, 70, 72, 75]
        escenarios_dict = {k: ESCENARIOS[k] for k in ESC_RAPIDOS if k in ESCENARIOS}
        scenarios_pert_list = [1, 6, 11, 20, 22, 70]
    else:
        print("Iniciando generación completa")
        print("💡 TIP: Puedes ejecutar con 'python MPC_AI_v5.py --fast' para un conjunto representativo rápido.")
        escenarios_dict = ESCENARIOS
        scenarios_pert_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 70, 71, 72, 73, 74, 75]
    print("Iniciando generación de datos nominales...")

    for n, (nombre, var_T, var_H, ref, x0) in escenarios_dict.items():
        X_meas, U, D, ref_arr, X_true = run_escenario(n, nombre, var_T, var_H, ref, x0)

        if U.shape[1] > 0:
            log_csv(X_meas, U, D, ref_arr, scenario_id=n, scenario_group_id=n)
            fig = plot_escenario(n, nombre, X_meas, U, D, ref_arr, X_true)
            fig.savefig(f'escenario_{n:02d}.png', dpi=130)
            plt.close(fig)

    print("\nIniciando generación de datos con Data Augmentation (Robustez)...")
    PERTURBACIONES = [
        {"nombre": "DEGRADADO_85", "eta_heat": 0.85, "eta_hum": 0.85, "eta_vent": 0.85},
        {"nombre": "POTENCIADO_115", "eta_heat": 1.15, "eta_hum": 1.15, "eta_vent": 1.15},
        {"nombre": "HUM_DEBIL_115", "eta_heat": 1.00, "eta_hum": 1.15, "eta_vent": 1.00}
    ]
    for n in scenarios_pert_list:
        if n not in ESCENARIOS:
            continue
        nombre_base, var_T, var_H, ref, x0 = ESCENARIOS[n]
        for j, p in enumerate(PERTURBACIONES):
            nombre_perturbado = f"{nombre_base}_{p['nombre']}"
            scenario_id_aug = 10000 + n * 10 + j
            print(f"\n>>> Generando: {nombre_perturbado} (ID: {scenario_id_aug})")
            X_meas, U, D, ref_arr, X_true = run_escenario(
                scenario_id_aug, nombre_perturbado, var_T, var_H, ref, x0,
                eta_heat=p['eta_heat'], eta_hum=p['eta_hum'], eta_vent=p['eta_vent'],
                verbose=False
            )
            if U.shape[1] > 0:
                log_csv(
                    X_meas, U, D, ref_arr,
                    scenario_id=scenario_id_aug,
                    scenario_group_id=n
                )
    print(f"\n{'=' * 70}")
    print(f"✓ SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print(f"✓ Datos guardados en: {CSV_PATH}")
    print(f"{'=' * 70}")
