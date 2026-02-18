import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 3D optional
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

st.set_page_config(page_title="868 MHz LoRa Coverage Calculator", layout="wide")

st.title("LoRa @ 868 MHz – pokrytí (FSPL / log-distance) + antény + profil terénu + 3D heatmap")
st.caption("Orientační výpočet: bez clutter modelu (budovy/les), bez fadingu, bez rušení, bez variability SF/BW/CR. Ber jako inženýrský odhad.")

# ----------------------------
# Helpers: propagation
# ----------------------------
def fspl_db(f_mhz: float, d_m: np.ndarray) -> np.ndarray:
    d_km = np.maximum(d_m, 1e-9) / 1000.0
    return 32.44 + 20.0 * np.log10(f_mhz) + 20.0 * np.log10(d_km)

def log_distance_pl_db(f_mhz: float, d_m: np.ndarray, n: float, d0_m: float) -> np.ndarray:
    d0_m = max(d0_m, 0.1)
    pl_d0 = fspl_db(f_mhz, np.array([d0_m]))[0]
    ratio = np.maximum(d_m, d0_m) / d0_m
    return pl_d0 + 10.0 * n * np.log10(ratio)

def compute_prx_dbm(ptx_dbm, gtx_dbi, grx_dbi, ltx_db, lrx_db, pl_db):
    return ptx_dbm + gtx_dbi + grx_dbi - ltx_db - lrx_db - pl_db

def max_range_for_sensitivity(d_m: np.ndarray, prx_dbm: np.ndarray, sens_dbm: float):
    ok = prx_dbm >= sens_dbm
    if not np.any(ok):
        return None
    return float(np.max(d_m[ok]))

# ----------------------------
# Helpers: terrain / diffraction
# ----------------------------
def fresnel_radius_m(f_hz: float, d1_m: np.ndarray, d2_m: np.ndarray, n_zone: int = 1) -> np.ndarray:
    c = 299_792_458.0
    lam = c / f_hz
    return np.sqrt(n_zone * lam * (d1_m * d2_m) / np.maximum(d1_m + d2_m, 1e-9))

def knife_edge_diffraction_loss_db(v: float) -> float:
    if v <= -0.78:
        return 0.0
    return float(6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1))

def terrain_diffraction_from_profile(f_hz: float, dist_m: np.ndarray, elev_m: np.ndarray,
                                     tx_ant_agl_m: float, rx_ant_agl_m: float):
    order = np.argsort(dist_m)
    x = dist_m[order].astype(float)
    z = elev_m[order].astype(float)

    D = float(np.max(x))
    if D <= 0:
        return None

    z_tx = float(z[0] + tx_ant_agl_m)
    z_rx = float(z[-1] + rx_ant_agl_m)

    z_los = z_tx + (z_rx - z_tx) * (x / D)

    d1 = x
    d2 = D - x
    r1 = fresnel_radius_m(f_hz, d1, d2, n_zone=1)

    h = z - z_los  # >0 terrain above LOS

    c = 299_792_458.0
    lam = c / f_hz
    inv = (1.0 / np.maximum(d1, 1e-6)) + (1.0 / np.maximum(d2, 1e-6))
    v = h * np.sqrt(2.0 / (lam * inv))

    clearance = z_los - z
    fresnel_clear = clearance - r1

    v_max = float(np.nanmax(v[1:-1])) if len(v) > 2 else float(np.nanmax(v))
    Ld = knife_edge_diffraction_loss_db(v_max)

    return {
        "x": x,
        "z": z,
        "z_los": z_los,
        "r1": r1,
        "clearance": clearance,
        "fresnel_clear": fresnel_clear,
        "v": v,
        "v_max": v_max,
        "Ld_db": Ld,
        "D_m": D,
        "z_tx": z_tx,
        "z_rx": z_rx,
    }

# ----------------------------
# Helpers: antenna patterns
# ----------------------------
def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def pattern_isotropic(theta, phi, **kw):
    return np.ones_like(theta, dtype=float)

def pattern_dipole_halfwave(theta, phi, **kw):
    # "donut": max at horizon (theta=0), nulls to ±90°
    return np.cos(theta) ** 2

def pattern_qfh(theta, phi, **kw):
    return 0.7 + 0.3 * (np.cos(theta) ** 2)

def pattern_patch(theta, phi, beam_deg=70, steer_az_deg=0.0, **kw):
    beam = np.deg2rad(max(10.0, float(beam_deg)))
    q = np.log(0.5) / np.log(np.cos(beam / 2.0))
    steer = np.deg2rad(float(steer_az_deg))
    dphi = wrap_pi(phi - steer)
    az = np.cos(dphi) ** q
    az = np.where(np.abs(dphi) <= np.pi/2, az, 0.0)  # front hemisphere
    el = np.cos(theta) ** 1.5
    return az * el

def pattern_yagi(theta, phi, beam_deg=55, front_to_back_db=15, steer_az_deg=0.0, **kw):
    beam = np.deg2rad(max(10.0, float(beam_deg)))
    q = np.log(0.5) / np.log(np.cos(beam / 2.0))
    steer = np.deg2rad(float(steer_az_deg))
    dphi = wrap_pi(phi - steer)

    main = np.cos(dphi) ** q
    main = np.where(np.abs(dphi) <= np.pi/2, main, 0.0)
    back = 10 ** (-float(front_to_back_db) / 10.0)
    patt_phi = main + back * (1.0 - main)

    patt_theta = np.cos(theta) ** 2
    return patt_phi * patt_theta

def pattern_collinear(theta, phi, collinear_p: float = 4.0, **kw):
    # Vertical collinear: narrower "donut" as p grows.
    p = float(collinear_p)
    p = max(0.5, min(p, 40.0))
    return np.cos(theta) ** p


ANTENNAS = {
    "Isotropní": {
        "pattern_fn": pattern_isotropic,
        "params": {}
    },
    "Dipól 1/2λ": {
        "pattern_fn": pattern_dipole_halfwave,
        "params": {}
    },
    "QFH": {
        "pattern_fn": pattern_qfh,
        "params": {}
    },
    "Kolineár vertikál": {
        "pattern_fn": pattern_collinear,
        "params": {}
    },
    "Patch": {
        "pattern_fn": pattern_patch,
        "params": {"beam_deg": 70, "steer_az_deg": 0.0}
    },
    "Yagi": {
        "pattern_fn": pattern_yagi,
        "params": {"beam_deg": 55, "front_to_back_db": 15, "steer_az_deg": 0.0}
    },
}

def antenna_gain_at_direction_db(
    ant_key: str,
    theta: np.ndarray,
    phi: np.ndarray,
    gain_dbi: float,
    override_params: dict | None = None
) -> np.ndarray:
    """Return directional gain in dBi where `gain_dbi` is the PEAK gain of the antenna."""
    ant = ANTENNAS[ant_key]
    fn = ant["pattern_fn"]
    params = dict(ant.get("params", {}))
    if override_params:
        params.update(override_params)

    p = fn(theta, phi, **params)
    m = np.nanmax(p)
    if not np.isfinite(m) or m <= 0:
        p = np.ones_like(theta, dtype=float)
        m = 1.0
    p = p / m  # normalize, so gain_dbi is peak
    return float(gain_dbi) + db10(p)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Model šíření")
    model = st.radio("Výpočet ztrát", ["FSPL (volný prostor)", "log-distance"], index=0)

    st.header("Frekvence")
    f_mhz = st.number_input("Frekvence (MHz)", min_value=100.0, max_value=6000.0, value=868.0, step=1.0)
    f_hz = f_mhz * 1e6

    st.header("Vzdálenost pro grafy")
    d_min = st.number_input("Min vzdálenost (m)", min_value=0.1, value=1.0, step=1.0)
    d_max = st.number_input("Max vzdálenost (m)", min_value=1.0, value=5000.0, step=50.0)
    points = st.slider("Počet bodů", min_value=200, max_value=5000, value=1200, step=100)

    st.header("Log-distance parametry")
    n = st.slider("Path-loss exponent n", min_value=1.5, max_value=6.0, value=2.7, step=0.1)
    d0_m = st.number_input("Referenční vzdálenost d0 (m)", min_value=0.1, value=1.0, step=0.5)

    st.header("Antény (typ + diagram)")
    tx_ant = st.selectbox("TX anténa (diagram)", list(ANTENNAS.keys()), index=1)
    rx_ant = st.selectbox("RX anténa (diagram)", list(ANTENNAS.keys()), index=1)

    tx_gain_dbi = st.number_input("TX zisk (dBi)", value=2.0, step=0.5)
    rx_gain_dbi = st.number_input("RX zisk (dBi)", value=2.0, step=0.5)
    collinear_auto_shape = st.checkbox("Kolineár: odvozovat tvar ze zisku (ON/OFF)", value=True)
    st.caption("Když je zapnuto, vyšší dBi u kolineáru zužuje vertikální lalok (donut je „placatější“).")

    tx_az = st.number_input("TX azimut (°) – směr hlavního laloku", value=0.0, step=5.0)
    rx_az = st.number_input("RX azimut (°)", value=180.0, step=5.0)
    st.caption("Azimut v mapě: 0° = +x doprava, 90° = +y nahoru. Dipól/QFH jsou skoro omni v azimutu.")
    st.header("Link budget (volitelné)")
    use_link_budget = st.checkbox("Počítat PRX / dosah podle citlivosti", value=True)

    def opt_number(label, default):
        c = st.checkbox(f"Zadat {label}", value=True)
        if c:
            return st.number_input(label, value=float(default), step=0.5)
        return None

    if use_link_budget:
        ptx_dbm = opt_number("TX výkon (dBm)", 14.0)
        ltx_db  = opt_number("Ztráty TX (kabel/konektory) (dB)", 1.0)
        lrx_db  = opt_number("Ztráty RX (kabel/konektory) (dB)", 1.0)
        sens_dbm = opt_number("Citlivost RX (dBm)", -120.0)
    else:
        ptx_dbm = ltx_db = lrx_db = sens_dbm = None

    st.header("Heatmap (2D/3D)")
    show_map = st.checkbox("Zobrazit 2D heatmap", value=True)
    map_radius_m = st.number_input("Poloměr mapy (m)", min_value=10.0, value=1500.0, step=50.0)
    map_res = st.slider("Rozlišení mapy (pix na osu)", min_value=100, max_value=500, value=220, step=20)

    show_3d = st.checkbox("Zobrazit 3D heatmap (surface)", value=False, disabled=not HAS_PLOTLY)
    if not HAS_PLOTLY:
        st.caption("3D heatmap vyžaduje `plotly` (pip install plotly).")

    st.subheader("Výšky antén pro mapu (AGL)")
    tx_h_map = st.number_input("TX výška (m)", min_value=0.0, value=2.0, step=0.5)
    rx_h_map = st.number_input("RX výška (m)", min_value=0.0, value=2.0, step=0.5)

    st.header("1D graf – směr (volitelné)")
    use_1d_dir = st.checkbox("Aplikovat diagram i do 1D grafu (azimut)", value=False)
    graph_az = st.number_input("Azimut pro 1D graf (°)", value=0.0, step=5.0)

    st.header("Vyzařovací diagram (vizualizace)")
    show_pattern = st.checkbox("Zobrazit 2D řezy + 3D kouli antény", value=True)
    pattern_for = st.radio("Kterou anténu zobrazit", ["TX", "RX"], index=0)
    pattern_res = st.slider("Rozlišení koule (větší = pomalejší)", 30, 140, 80, 10, disabled=not HAS_PLOTLY)

    st.header("Profil terénu (volitelné)")
    use_terrain = st.checkbox("Použít profil terénu pro 1 link (TX→RX)", value=False)


def _override_params_for_antenna(ant_key: str, steer_az_deg: float, gain_dbi: float, collinear_auto_shape: bool) -> dict:
    params = {"steer_az_deg": float(steer_az_deg)}
    if ant_key == "Kolineár vertikál" and collinear_auto_shape:
        # Empirical mapping: higher gain => narrower vertical lobe.
        p = 0.8 * float(gain_dbi)
        p = max(1.2, min(p, 30.0))
        params["collinear_p"] = p
    return params

# ----------------------------
# Base curves (distance)
# ----------------------------
d_m = np.geomspace(max(d_min, 0.1), max(d_max, d_min + 0.1), int(points))

if model.startswith("FSPL"):
    pl_db = fspl_db(f_mhz, d_m)
else:
    pl_db = log_distance_pl_db(f_mhz, d_m, n=n, d0_m=d0_m)

have_lb = all(v is not None for v in [ptx_dbm, ltx_db, lrx_db])

# antenna gains for 1D curve
if use_1d_dir:
    phi = np.deg2rad(graph_az) * np.ones_like(d_m)
    theta = np.arctan2((rx_h_map - tx_h_map), np.maximum(d_m, 1e-6))
    gtx_1d = antenna_gain_at_direction_db(
        tx_ant, theta, phi,
        gain_dbi=tx_gain_dbi,
        override_params=_override_params_for_antenna(tx_ant, tx_az, tx_gain_dbi, collinear_auto_shape)
    )
    grx_1d = antenna_gain_at_direction_db(
        rx_ant, -theta, wrap_pi(phi + np.pi),
        gain_dbi=rx_gain_dbi,
        override_params=_override_params_for_antenna(rx_ant, rx_az, rx_gain_dbi, collinear_auto_shape)
    )
else:
    gtx_1d = float(tx_gain_dbi)
    grx_1d = float(rx_gain_dbi)

if have_lb:
    prx_dbm = compute_prx_dbm(ptx_dbm, gtx_1d, grx_1d, ltx_db, lrx_db, pl_db)
else:
    prx_dbm = None

# ----------------------------
# Radiation pattern visualization (2D cuts + 3D "sphere")
# ----------------------------
if show_pattern:
    st.subheader("Vyzařovací diagram antény (2D řezy + 3D „koule“)")
    if pattern_for == "TX":
        ant_key = tx_ant
        ant_gain_dbi = float(tx_gain_dbi)
        steer = float(tx_az)
        label = "TX"
    else:
        ant_key = rx_ant
        ant_gain_dbi = float(rx_gain_dbi)
        steer = float(rx_az)
        label = "RX"

    g_peak = float(ant_gain_dbi)

    c1, c2 = st.columns([1, 1])
    with c1:
        # Azimuth cut: theta = 0 (horizon), phi sweep 0..360
        phis = np.linspace(-np.pi, np.pi, 721)
        thetas = np.zeros_like(phis)
        g_az = antenna_gain_at_direction_db(
            ant_key, thetas, phis,
            gain_dbi=ant_gain_dbi,
            override_params=_override_params_for_antenna(ant_key, steer, ant_gain_dbi, collinear_auto_shape)
        )

        fig = plt.figure()
        plt.plot(np.rad2deg(phis), g_az)
        plt.xlabel("Azimut φ (°)")
        plt.ylabel("Zisk (dBi)")
        plt.title(f"{label} – azimutový řez (elevace 0° = horizont)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        st.pyplot(fig, clear_figure=True)

    with c2:
        # Elevation cut: phi = steer direction (0 in antenna coords), theta sweep -90..+90
        thetas = np.linspace(-np.pi/2, np.pi/2, 721)
        phis = np.zeros_like(thetas)  # along steering direction
        g_el = antenna_gain_at_direction_db(
            ant_key, thetas, phis,
            gain_dbi=ant_gain_dbi,
            override_params=_override_params_for_antenna(ant_key, 0.0, ant_gain_dbi, collinear_auto_shape)  # phi=0 already aligned; steer not needed here
        )
        # For non-isotropic this shows the "donut" (dipole), etc.
        fig = plt.figure()
        plt.plot(np.rad2deg(thetas), g_el)
        plt.xlabel("Elevace θ (°)  (0° = horizont, ±90° = nahoru/dolů)")
        plt.ylabel("Zisk (dBi)")
        plt.title(f"{label} – elevace řez (azimut ve směru hlavního laloku)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        st.pyplot(fig, clear_figure=True)

    st.caption(
        "Pozn.: Tohle je diagram v úhlech (θ/φ). Proto tady dipól ukáže donut, zatímco X/Y heatmap ukazuje pokrytí po zemi."
    )

    # 3D "sphere" in plotly
    if HAS_PLOTLY:
        res = int(pattern_res)
        # theta: -90..+90 (elevation), phi: -180..+180
        theta = np.linspace(-np.pi/2, np.pi/2, res)
        phi = np.linspace(-np.pi, np.pi, res)
        TH, PH = np.meshgrid(theta, phi)

        # Gain in dBi for each direction
        G = antenna_gain_at_direction_db(
            ant_key, TH, PH,
            gain_dbi=ant_gain_dbi,
            override_params=_override_params_for_antenna(ant_key, steer, ant_gain_dbi, collinear_auto_shape)
        )

        # Make a 3D "ball": use radius proportional to power ratio relative to peak
        G_rel = G - np.nanmax(G)  # <= 0
        R = 10 ** (G_rel / 10.0)  # power ratio
        # normalize to nicer range (avoid very tiny radii)
        R = 0.15 + 0.85 * (R / np.nanmax(R))

        # Convert spherical coords to xyz.
        # Here TH is elevation from horizon: 0 = horizon, +90 up.
        # Let z be "up".
        X = R * np.cos(TH) * np.cos(PH)
        Y = R * np.cos(TH) * np.sin(PH)
        Z = R * np.sin(TH)

        fig3 = go.Figure(
            data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=G, showscale=True)]
        )
        fig3.update_layout(
            title=f"{label} – 3D vyzařovací diagram (barva = zisk dBi, tvar = relativní vyzařování)",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("3D „koule“ vyžaduje plotly: `pip install plotly`")

# ----------------------------
# Layout: graphs + summary
# ----------------------------
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Grafy")

    fig1 = plt.figure()
    plt.plot(d_m, pl_db)
    plt.xscale("log")
    plt.xlabel("Vzdálenost (m) [log]")
    plt.ylabel("Path loss (dB)")
    plt.title(f"Ztráty šíření: {model} @ {f_mhz:.1f} MHz")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    st.pyplot(fig1, clear_figure=True)

    if prx_dbm is not None:
        fig2 = plt.figure()
        plt.plot(d_m, prx_dbm)
        plt.xscale("log")
        plt.xlabel("Vzdálenost (m) [log]")
        plt.ylabel("PRX (dBm)")
        title_extra = f" (diagram @ az {graph_az:.0f}°)" if use_1d_dir else " (peak zisky = zadané dBi)"
        plt.title("Přijatý výkon vs vzdálenost" + title_extra)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        if sens_dbm is not None:
            plt.axhline(sens_dbm, linewidth=1.0)
            plt.text(d_m[0], sens_dbm, f"  citlivost {sens_dbm:.1f} dBm", va="bottom")
        st.pyplot(fig2, clear_figure=True)
    else:
        st.info("PRX graf se zobrazí po zadání TX výkonu a ztrát (TX/RX). Zisky antén zadáváš ručně (typ antény určuje jen tvar diagramu).")

with colB:
    st.subheader("Souhrn & export")

    st.markdown("**Model:** " + ("FSPL" if model.startswith("FSPL") else f"log-distance (n={n:.1f}, d0={d0_m:g} m)"))
    st.markdown(f"**Frekvence:** {f_mhz:.1f} MHz")
    st.markdown(f"**TX anténa:** {tx_ant} (azimut {tx_az:.0f}°)")
    st.markdown(f"**RX anténa:** {rx_ant} (azimut {rx_az:.0f}°)")
    st.markdown(f"**Path loss @ {d_max:g} m:** {float(pl_db[-1]):.2f} dB")

    if prx_dbm is not None:
        st.markdown(f"**PRX @ {d_max:g} m:** {float(prx_dbm[-1]):.2f} dBm")
        if sens_dbm is not None:
            rmax = max_range_for_sensitivity(d_m, prx_dbm, sens_dbm)
            if rmax is None:
                st.warning("Podle zadané citlivosti nedosáhne PRX ani na nejmenší vzdálenost v grafu.")
            else:
                st.success(f"Odhad max dosahu (PRX ≥ citlivost): **{rmax:.1f} m**")

        if ptx_dbm is not None and ltx_db is not None:
            eirp = ptx_dbm + float(tx_gain_dbi) - ltx_db
            st.markdown(f"**EIRP (odhad):** {eirp:.2f} dBm")
    else:
        st.markdown("**PRX:** nezobrazeno (chybí TX výkon nebo ztráty).")

    df = pd.DataFrame({"distance_m": d_m, "path_loss_db": pl_db})
    if prx_dbm is not None:
        df["prx_dbm"] = prx_dbm

    st.dataframe(df.head(25), use_container_width=True)
    st.download_button("Stáhnout výsledky CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="lora_868_results.csv", mime="text/csv")

# ----------------------------
# 2D / 3D Heatmap (with antenna pattern)
# ----------------------------
if show_map or show_3d:
    st.subheader("Heatmap pokrytí (radiální model + diagram antén)")

    R = float(map_radius_m)
    res = int(map_res)
    xs = np.linspace(-R, R, res)
    ys = np.linspace(-R, R, res)
    X, Y = np.meshgrid(xs, ys)
    D = np.sqrt(X**2 + Y**2)
    D_safe = np.maximum(D, max(d_min, 0.5))

    if model.startswith("FSPL"):
        PL_map = fspl_db(f_mhz, D_safe)
    else:
        PL_map = log_distance_pl_db(f_mhz, D_safe, n=n, d0_m=d0_m)

    phi = np.arctan2(Y, X)
    theta = np.arctan2((rx_h_map - tx_h_map), np.maximum(D_safe, 1e-6))

    gtx_dir = antenna_gain_at_direction_db(
        tx_ant, theta, phi,
        gain_dbi=tx_gain_dbi,
        override_params=_override_params_for_antenna(tx_ant, tx_az, tx_gain_dbi, collinear_auto_shape)
    )
    phi_back = wrap_pi(phi + np.pi)
    grx_dir = antenna_gain_at_direction_db(
        rx_ant, -theta, phi_back,
        gain_dbi=rx_gain_dbi,
        override_params=_override_params_for_antenna(rx_ant, rx_az, rx_gain_dbi, collinear_auto_shape)
    )

    if have_lb:
        PRX_map = ptx_dbm + gtx_dir + grx_dir - ltx_db - lrx_db - PL_map
        Z = PRX_map
        zlabel = "PRX (dBm)"
    else:
        Z = -(PL_map - (gtx_dir + grx_dir))
        zlabel = "-(PL - Gtx - Grx) [dB]  (vyšší = lepší)"

    Z = np.where(D <= R, Z, np.nan)

    if show_map:
        figm = plt.figure()
        im = plt.imshow(Z, extent=[-R, R, -R, R], origin="lower", aspect="equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("2D heatmap (diagram antén aplikován)")
        cbar = plt.colorbar(im)
        cbar.set_label(zlabel)

        if have_lb and sens_dbm is not None:
            try:
                plt.contour(X, Y, PRX_map, levels=[sens_dbm], linewidths=1.0)
            except Exception:
                pass

        st.pyplot(figm, clear_figure=True)

    if show_3d and HAS_PLOTLY:
        max_3d = 200
        if res > max_3d:
            step = int(np.ceil(res / max_3d))
            X3, Y3, Z3 = X[::step, ::step], Y[::step, ::step], Z[::step, ::step]
        else:
            X3, Y3, Z3 = X, Y, Z

        fig3d = go.Figure(data=[go.Surface(x=X3, y=Y3, z=Z3, showscale=True)])
        fig3d.update_layout(
            title="3D heatmap (surface) – diagram antén aplikován",
            scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title=zlabel),
            height=650,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig3d, use_container_width=True)

# ----------------------------
# Terrain profile (single link)
# ----------------------------
if use_terrain:
    st.subheader("Profil terénu pro konkrétní link (TX → RX)")

    left, right = st.columns([1.4, 1])

    with right:
        st.markdown("**🛈 Jak získat profil terénu**")
        with st.expander("Klikni pro návod (zdroje profilu + formát CSV)"):
            st.markdown(
                """
**Cíl:** získat body *vzdálenost (m)* a *nadmořská výška (m)* podél přímky mezi TX a RX.

Možnosti:
- **QGIS**: nakresli čáru TX→RX → profile tool / sampling nad DEM (SRTM/ALOS/EU-DEM). Export CSV.
- **Online nástroje**: „elevation profile“ po trase (ideálně s exportem do CSV).
- Důležité je, aby vzdálenost začínala **0** a končila v místě RX.

**Formát CSV (minimálně):**
- `distance_m` (0 … D)
- `elevation_m` (nadmořská výška v metrech)
                """
            )

        st.markdown("**Výšky antén (AGL)**")
        tx_agl = st.number_input("TX výška antény nad terénem (m)", min_value=0.0, value=2.0, step=0.5)
        rx_agl = st.number_input("RX výška antény nad terénem (m)", min_value=0.0, value=2.0, step=0.5)

        apply_diffraction = st.checkbox("Přičíst difrakční útlum (single knife-edge) do link budgetu", value=True)

    with left:
        st.markdown("**Nahrání profilu**")
        uploaded = st.file_uploader("CSV profil terénu (distance_m, elevation_m) 🛈", type=["csv"])

        if uploaded is None:
            st.info("Nahraj CSV s `distance_m` a `elevation_m`.")
        else:
            try:
                prof = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Nepovedlo se načíst CSV: {e}")
                prof = None

            if prof is not None:
                cols = {c.lower().strip(): c for c in prof.columns}
                if "distance_m" not in cols or "elevation_m" not in cols:
                    st.error("CSV musí obsahovat sloupce `distance_m` a `elevation_m` (přesně tak).")
                else:
                    dist = prof[cols["distance_m"]].to_numpy(dtype=float)
                    elev = prof[cols["elevation_m"]].to_numpy(dtype=float)

                    out = terrain_diffraction_from_profile(
                        f_hz=f_hz, dist_m=dist, elev_m=elev,
                        tx_ant_agl_m=float(tx_agl), rx_ant_agl_m=float(rx_agl)
                    )

                    if out is None:
                        st.error("Profil je neplatný (nulová délka).")
                    else:
                        x = out["x"]
                        z = out["z"]
                        z_los = out["z_los"]
                        r1 = out["r1"]
                        Dlink = out["D_m"]
                        Ld = out["Ld_db"]
                        vmax = out["v_max"]

                        figp = plt.figure()
                        plt.plot(x, z, label="Terén (m n. m.)")
                        plt.plot(x, z_los, label="LOS (anténa→anténa)")
                        plt.plot(x, z_los + r1, linewidth=0.8, label="Fresnel F1 (horní)")
                        plt.plot(x, z_los - r1, linewidth=0.8, label="Fresnel F1 (dolní)")
                        plt.xlabel("Vzdálenost po trase (m)")
                        plt.ylabel("Výška (m n. m.)")
                        plt.title("Profil terénu + LOS + 1. Fresnelova zóna")
                        plt.grid(True, linestyle="--", linewidth=0.5)
                        plt.legend()
                        st.pyplot(figp, clear_figure=True)

                        if model.startswith("FSPL"):
                            pl_link = float(fspl_db(f_mhz, np.array([Dlink]))[0])
                        else:
                            pl_link = float(log_distance_pl_db(f_mhz, np.array([Dlink]), n=n, d0_m=d0_m)[0])

                        pl_link_eff = pl_link + (Ld if apply_diffraction else 0.0)

                        st.markdown(f"**Délka linku dle profilu:** {Dlink:.1f} m")
                        st.markdown(f"**Základní path loss:** {pl_link:.2f} dB")
                        st.markdown(f"**Difrakční útlum (single knife-edge):** {Ld:.2f} dB (v_max={vmax:.2f})")
                        st.markdown(f"**Efektivní path loss:** {pl_link_eff:.2f} dB")

                        gtx_link = float(tx_gain_dbi)
                        grx_link = float(rx_gain_dbi)

                        if have_lb:
                            prx_link = float(ptx_dbm + gtx_link + grx_link - ltx_db - lrx_db - pl_link_eff)
                            st.success(f"Odhad PRX pro tento profil (peak zisky = zadané dBi): **{prx_link:.2f} dBm**")
                            if sens_dbm is not None:
                                st.success("LINK OK" if prx_link >= sens_dbm else "LINK NEVYCHÁZÍ")
                        else:
                            st.info("Chceš-li PRX pro tento profil, zapni link budget a vyplň TX výkon + ztráty.")

                        summary = pd.DataFrame([{
                            "link_distance_m": Dlink,
                            "base_path_loss_db": pl_link,
                            "diffraction_loss_db": Ld if apply_diffraction else 0.0,
                            "effective_path_loss_db": pl_link_eff,
                            "tx_gain_peak_dbi": gtx_link,
                            "rx_gain_peak_dbi": grx_link,
                            "v_max": vmax,
                        }])

                        st.dataframe(summary, use_container_width=True)
                        st.download_button(
                            "Stáhnout souhrn linku (CSV)",
                            data=summary.to_csv(index=False).encode("utf-8"),
                            file_name="terrain_link_summary.csv",
                            mime="text/csv"
                        )
