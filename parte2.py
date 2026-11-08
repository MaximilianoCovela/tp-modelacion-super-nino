"""TP El Niño - Parte 2: severidad estacional, desfase y sensibilidad paramétrica.

Uso:
    python parte2.py a          # un inciso
    python parte2.py a b        # varios
"""
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from solucion_referencia import resolver_euler

# Paso temporal elegido a partir de la Parte 1: error de Euler muy chico y
# muy por debajo del umbral de inestabilidad (entre 1.25 y 1.5 meses).
H_PARTE2 = 0.01
T0, TF = 0.0, 60.0
Y0 = [1.0, 0.0, 0.0]
INICIO_ULTIMO_CICLO = 48.0
PERIODO = 12.0  # meses (omega = 2*pi/12)

# Parametros nominales
NOMINALES = {
    "T_MAX": 4.0,
    "R_OCE": 0.30,
    "ALPHA": 0.05,
    "BETA": 0.20,
    "GAMMA": 0.10,
    "DELTA": 0.40,
    "SIGMA": 0.70,
    "A": 0.45,
}
OMEGA = 2 * np.pi / PERIODO

# Valores elegidos para el inciso 2.b: equiespaciados dentro de los intervalos
# del enunciado e incluyendo siempre el valor nominal. Cada entrada tiene la
# lista de valores y la unidad del parametro para el eje del grafico.
VALORES_2B = {
    "A": ([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60], "mes⁻¹"),
    "BETA": ([0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30], "u.h. mes⁻¹ °C⁻²"),
    "GAMMA": ([0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20], "mes⁻¹ u.p.⁻¹"),
    "SIGMA": ([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], "mes⁻¹"),
}


def crear_modelo(p):
    """Devuelve f(t, y) del sistema con los parametros del diccionario p."""
    def f(t, y):
        T, H, P = y
        R_t = p["R_OCE"] + p["A"] * np.cos(OMEGA * t - np.pi)
        dT = R_t * T * (1 - T / p["T_MAX"]) - p["ALPHA"] * H
        dH = p["BETA"] * T**2 - p["GAMMA"] * H * P
        dP = p["DELTA"] * H**2 - p["SIGMA"] * P
        return np.array([dT, dH, dP])
    return f


def simular(p, h=H_PARTE2):
    return resolver_euler(crear_modelo(p), T0, TF, h, Y0)


def indicadores(t, y):
    """SP, desfase y estadisticas de P, usando solo el ultimo ciclo anual."""
    mascara = t >= INICIO_ULTIMO_CICLO - 1e-9
    tc, T, P = t[mascara], y[mascara, 0], y[mascara, 2]

    P_max, P_min, P_med = P.max(), P.min(), P.mean()
    SP = (P_max - P_min) / P_med

    t_T_max = tc[np.argmax(T)]
    # Maximo de P posterior al maximo de T. La solucion es periodica: si el pico
    # de P del ciclo cae antes que el de T, el "posterior" es el del ciclo
    # siguiente, es decir, se le suma un periodo.
    t_P_max = tc[np.argmax(P)]
    if t_P_max < t_T_max:
        t_P_max += PERIODO
    delta_t = t_P_max - t_T_max

    return {
        "SP": SP, "dt_TP": delta_t,
        "P_max": P_max, "P_min": P_min, "P_med": P_med,
        "t_T_max": t_T_max, "t_P_max": t_P_max,
    }


def inciso_2a(timestamp):
    t, y = simular(NOMINALES)
    ind = indicadores(t, y)

    print("\n--- Resultados Punto 2.a (parametros nominales, h = %g) ---" % H_PARTE2)
    print(f"Pmax  = {ind['P_max']:.6f}")
    print(f"Pmin  = {ind['P_min']:.6f}")
    print(f"Pmed  = {ind['P_med']:.6f}")
    print(f"SP    = (Pmax - Pmin) / Pmed = {ind['SP']:.6f}")
    print(f"tT,max = {ind['t_T_max']:.2f} meses | tP,max = {ind['t_P_max']:.2f} meses")
    print(f"dtTP  = tP,max - tT,max = {ind['dt_TP']:.2f} meses")

    nombres = ["T(t) [°C]", "H(t) [u.h.]", "P(t) [u.p.]"]
    colores = ["tab:red", "tab:blue", "tab:green"]
    ultimo = t >= INICIO_ULTIMO_CICLO - 1e-9

    # Figura 1: 60 meses completos
    fig, ejes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(ejes):
        ax.plot(t, y[:, i], color=colores[i])
        ax.axvspan(INICIO_ULTIMO_CICLO, TF, color="gray", alpha=0.15,
                   label="Último ciclo (48 a 60)" if i == 0 else None)
        ax.set_ylabel(nombres[i])
        ax.grid(True, linestyle=":", alpha=0.7)
    ejes[0].legend(loc="lower right")
    ejes[0].set_title("Inciso 2.a: T, H y P durante 60 meses (parámetros nominales)",
                      fontweight="bold")
    ejes[-1].set_xlabel("Tiempo (meses)")
    nombre1 = f"grafico_2a_60meses_{timestamp}.png"
    fig.savefig(nombre1, dpi=300, bbox_inches="tight")

    # Figura 2: ultimo ciclo anual, con los instantes usados para el desfase
    fig2, ejes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(ejes2):
        ax.plot(t[ultimo], y[ultimo, i], color=colores[i])
        ax.set_ylabel(nombres[i])
        ax.grid(True, linestyle=":", alpha=0.7)
    ejes2[0].axvline(ind["t_T_max"], color="k", linestyle="--")
    ejes2[0].plot(ind["t_T_max"], y[np.argmin(np.abs(t - ind["t_T_max"])), 0], "ko",
                  label=f"máx. T (t = {ind['t_T_max']:.2f})")
    ejes2[0].legend(loc="lower right")
    tP = ind["t_P_max"]
    tP_en_ciclo = tP if tP <= TF + 1e-9 else tP - PERIODO
    ejes2[2].axvline(tP_en_ciclo, color="k", linestyle="--")
    ejes2[2].axhline(ind["P_med"], color="gray", linestyle=":", label=f"Pmed = {ind['P_med']:.3f}")
    ejes2[2].axhline(ind["P_max"], color="green", linestyle=":", label=f"Pmax = {ind['P_max']:.3f}")
    ejes2[2].axhline(ind["P_min"], color="olive", linestyle=":", label=f"Pmin = {ind['P_min']:.3f}")
    ejes2[2].legend(loc="best", fontsize=8)
    ejes2[0].set_title(
        f"Inciso 2.a: último ciclo anual (SP = {ind['SP']:.3f}, ΔtTP = {ind['dt_TP']:.2f} meses)",
        fontweight="bold")
    ejes2[-1].set_xlabel("Tiempo (meses)")
    nombre2 = f"grafico_2a_ultimo_ciclo_{timestamp}.png"
    fig2.savefig(nombre2, dpi=300, bbox_inches="tight")

    print(f"Gráficos guardados: {nombre1} y {nombre2}")
    plt.show()
    plt.close("all")

def inciso_2b_sensibilidad(param_nombre, valores_param, timestamp, unidad=""):
    """
    Analiza y grafica cómo cambian SP y ΔtTP al modificar un único parámetro.
    Recibe el nombre del parámetro (ej. 'A', 'BETA'), una lista de valores a probar,
    el timestamp para guardar el archivo, y opcionalmente la unidad del parámetro.
    """
    resultados_sp = []
    resultados_dt = []

    # Obtenemos el valor nominal actual para marcarlo en el gráfico
    valor_nominal = NOMINALES[param_nombre]

    print(f"\n--- Resultados Punto 2.b: variando {param_nombre} (resto nominal) ---")  
    print(f"{'Valor':>10} | {'SP':>10} | {'ΔtTP (meses)':>15}")
    print("-" * 43)

    for valor in valores_param:
        # Copiamos los parámetros nominales y alteramos solo el actual
        p_actual = NOMINALES.copy()
        p_actual[param_nombre] = valor

        # Simulamos y calculamos indicadores  
        t, y = simular(p_actual)
        ind = indicadores(t, y)

        resultados_sp.append(ind["SP"])
        resultados_dt.append(ind["dt_TP"])

        print(f"{valor:10.4f} | {ind['SP']:10.4f} | {ind['dt_TP']:15.2f}")

    # --- Generación de Gráficos ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Subplot 1: Severidad Estacional (SP) vs Parámetro
    ax1.plot(valores_param, resultados_sp, marker='o', color='tab:blue', linewidth=2, label='SP')
    ax1.axvline(valor_nominal, color='tab:red', linestyle='--', label=f'Valor Nominal ({valor_nominal})')
    ax1.set_ylabel('Severidad Estacional (SP)')
    ax1.set_title(f'Sensibilidad de SP y ΔtTP respecto al parámetro {param_nombre}', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # Subplot 2: Desfase (ΔtTP) vs Parámetro
    ax2.plot(valores_param, resultados_dt, marker='s', color='tab:green', linewidth=2, label='ΔtTP')
    ax2.axvline(valor_nominal, color='tab:red', linestyle='--', label=f'Valor Nominal ({valor_nominal})')
    ax2.set_ylabel('Desfase ΔtTP (meses)')

    # Armamos la etiqueta del eje X sumando la unidad si fue provista
    etiqueta_x = param_nombre
    if unidad:
        etiqueta_x += f" [{unidad}]"
    ax2.set_xlabel(etiqueta_x)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    # Guardamos la figura usando el mismo esquema del inciso 2a  
    nombre_archivo = f"grafico_2b_sensibilidad_{param_nombre}_{timestamp}.png"
    fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()  

    print(f"Gráfico guardado como: {nombre_archivo}")

    return valores_param, resultados_sp, resultados_dt




def main():
    parser = argparse.ArgumentParser(description="TP El Niño - Parte 2")
    parser.add_argument("incisos", nargs="*", choices=["a", "b", "todos"], default=["todos"],
                        help="Incisos a ejecutar (ej: python parte2.py a b). Por defecto: todos")
    args = parser.parse_args()
    incisos = {"a", "b"} if "todos" in args.incisos else set(args.incisos)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "a" in incisos:
        inciso_2a(timestamp)

    if "b" in incisos:
        for nombre, (valores, unidad) in VALORES_2B.items():
            inciso_2b_sensibilidad(nombre, valores, timestamp, unidad=unidad)


if __name__ == "__main__":
    main()
