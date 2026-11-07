import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

T_MAX = 4.0
R_OCE = 0.30
ALPHA = 0.05
BETA = 0.20
GAMMA = 0.10
DELTA = 0.40
SIGMA = 0.70
AMPLITUD = 0.45
OMEGA = 2 * np.pi / 12

def modelo_el_nino(t, y):
    T, H, P = y
    R_t = R_OCE + AMPLITUD * np.cos(OMEGA * t - np.pi)

    dT = R_t * T * (1 - T / T_MAX) - ALPHA * H
    dH = BETA * T**2 - GAMMA * H * P
    dP = DELTA * H**2 - SIGMA * P

    return np.array([dT, dH, dP])

def resolver_euler(funcion_sistema, t0, tf, h, y0):
    # round (y no int) para evitar truncar por errores de punto flotante.
    # Los h usados dividen exactamente a (tf - t0), asi el paso real es h.
    N = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, N + 1)

    y = np.zeros((N + 1, len(y0)))
    y[0] = y0

    for n in range(N):
        y[n+1] = y[n] + h * funcion_sistema(t[n], y[n])

    return t, y

ARCHIVO_REFERENCIA = "referencia.csv"

def exportar_a_csv(t, y):
    datos = np.column_stack((t, y))
    columnas = ['Tiempo', 'Temperatura', 'Humedad', 'Precipitacion']
    df = pd.DataFrame(datos, columns=columnas)

    df.to_csv(ARCHIVO_REFERENCIA, index=False)

def exportar_grafico_temperatura(t, y, timestamp):
    plt.plot(t, y[:, 0], label='Temperatura (T) - Referencia')
    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Anomalía [°C]')
    plt.legend()

    nombre_archivo = f"grafico_{timestamp}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()

def calcular_errores_1b(t_ref, y_ref, t0, tf, y0):
    # Cada h es el doble del anterior y todos dividen exactamente a 60
    pasos_h = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    errores = []

    print("--- Resultados Punto 1.b ---")
    for h in pasos_h:
        t_test, y_test = resolver_euler(modelo_el_nino, t0, tf, h, y0)

        # Sincronizamos los tiempos: extraemos la T de referencia solo en los instantes de t_test
        T_ref_sincronizada = np.interp(t_test, t_ref, y_ref[:, 0])
        T_test = y_test[:, 0]

        # Calculamos el error absoluto máximo para este h
        error_maximo = np.max(np.abs(T_test - T_ref_sincronizada))
        errores.append(error_maximo)

        print(f"Paso h = {h:.4f} | Error máximo en T: {error_maximo:.6f}")

    return pasos_h, errores

def exportar_grafico_comparativo_1b(t_ref, y_ref, t0, tf, y0, timestamp):
    plt.figure(figsize=(10, 6))

    # Definimos los pasos a comparar y sus colores
    pasos_h = [0.05, 0.2, 0.8]
    colores = ['blue', 'orange', 'red']

    # Calculamos y graficamos cada aproximación PRIMERO (líneas continuas)
    for h, color in zip(pasos_h, colores):
        t_test, y_test = resolver_euler(modelo_el_nino, t0, tf, h, y0)
        plt.plot(t_test, y_test[:, 0], label=f'Euler h={h}', color=color, alpha=0.7)

    # Graficamos la solución de referencia AL FINAL para que quede por encima
    plt.plot(t_ref, y_ref[:, 0], label='Referencia', color='black', linewidth=1.5, linestyle=':', zorder=5)

    plt.xlabel('Tiempo (meses)')
    plt.ylabel('Temperatura (T) [°C]')
    plt.title('Comparación temporal: Método de Euler con distintos pasos h')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    nombre_archivo = f"grafico_comparativo_1b_{timestamp}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()

def calcular_orden_convergencia_1c(pasos_h, errores):
    print("\n--- Resultados Punto 1.c (Orden de Convergencia) ---")

    for i in range(1, len(pasos_h)):
        h_chico = pasos_h[i-1]
        h_grande = pasos_h[i]

        err_h = errores[i-1]
        err_2h = errores[i]

        p = np.log(err_2h / err_h) / np.log(2)

        print(f"h: {h_chico:.4f} -> {h_grande:.4f} | Orden p: {p:.4f}")

def analizar_inestabilidad_1d(t_ref, y_ref, t0, tf, y0, timestamp):
    # Pasos grandes que dividen exactamente a 60 (ya no se duplican exactamente)
    pasos_h_grandes = [0.8, 1.5, 2.0, 3.0, 4.0, 6.0]
    colores = ['purple', 'brown', 'magenta', 'darkred', 'green', 'teal']

    print("\n--- Resultados Punto 1.d (Análisis de Inestabilidad) ---")
    print(f"{'Paso h':<10} | {'Error Máximo en T':<25}")
    print("-" * 40)

    plt.figure(figsize=(10, 6))
    # Graficamos la referencia de fondo
    plt.plot(t_ref, y_ref[:, 0], label='Referencia ($h_{ref} = 0.0001$)', color='black', linewidth=2, linestyle='--')

    for h, color in zip(pasos_h_grandes, colores):
        t_test, y_test = resolver_euler(modelo_el_nino, t0, tf, h, y0)

        # Sincronizamos los tiempos
        T_ref_sincronizada = np.interp(t_test, t_ref, y_ref[:, 0])
        T_test = y_test[:, 0]

        error_maximo = np.max(np.abs(T_test - T_ref_sincronizada))
        print(f"Paso h = {h:<5.2f} | Error máximo en T: {error_maximo:.6e}")

        # Graficamos la solución inestable
        plt.plot(t_test, y_test[:, 0], label=f'Euler h={h}', color=color, alpha=0.8, linewidth=1.2)

    plt.xlabel('Tiempo (meses)', fontsize=11)
    plt.ylabel('Anomalía de Temperatura $T(t)$ [°C]', fontsize=11)
    plt.title('Inciso 1.d: Evidencia de Inestabilidad con Pasos Temporales Grandes', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.7)

    nombre_archivo = f"grafico_inestabilidad_1d_{timestamp}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfico de inestabilidad guardado como: {nombre_archivo}")
    plt.close()

def obtener_referencia(t0, tf, h_ref, y0, recalcular=False):
    # La referencia tarda en calcularse (600.000 pasos): se guarda en
    # referencia.csv y se reutiliza en las siguientes ejecuciones
    if os.path.exists(ARCHIVO_REFERENCIA) and not recalcular:
        df = pd.read_csv(ARCHIVO_REFERENCIA)
        t_ref = df['Tiempo'].to_numpy()
        # Solo se reutiliza si tiene el mismo paso (h_ref) y llega hasta tf
        if np.isclose(t_ref[1] - t_ref[0], h_ref) and np.isclose(t_ref[-1], tf):
            print(f"Referencia cargada desde {ARCHIVO_REFERENCIA}")
            return t_ref, df[['Temperatura', 'Humedad', 'Precipitacion']].to_numpy()

    print("Calculando referencia (puede tardar)...")
    t_ref, y_ref = resolver_euler(modelo_el_nino, t0, tf, h_ref, y0)
    exportar_a_csv(t_ref, y_ref)
    return t_ref, y_ref

def main():
    parser = argparse.ArgumentParser(description="TP El Niño - Parte 1")
    parser.add_argument("incisos", nargs="*", choices=["a", "b", "c", "d", "todos"],
                        default=["todos"],
                        help="Incisos a ejecutar (ej: python solucion_referencia.py b c). Por defecto: todos")
    parser.add_argument("--recalcular", action="store_true",
                        help="Ignora la referencia guardada y la vuelve a calcular")
    args = parser.parse_args()
    incisos = {"a", "b", "c", "d"} if "todos" in args.incisos else set(args.incisos)

    t0 = 0
    tf = 60
    h_ref = 0.0001
    y0 = [1.0, 0.0, 0.0]

    # La referencia (1.a) la necesitan todos los incisos
    t_ref, y_ref = obtener_referencia(t0, tf, h_ref, y0, args.recalcular)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # El CSV de la referencia ya se guardó en obtener_referencia
    if "a" in incisos:
        exportar_grafico_temperatura(t_ref, y_ref, timestamp)

    # 1.c necesita los errores de 1.b, por eso se calculan en ambos casos
    if "b" in incisos or "c" in incisos:
        pasos_h, errores = calcular_errores_1b(t_ref, y_ref, t0, tf, y0)

        if "b" in incisos:
            exportar_grafico_comparativo_1b(t_ref, y_ref, t0, tf, y0, timestamp)

        if "c" in incisos:
            calcular_orden_convergencia_1c(pasos_h, errores)

    if "d" in incisos:
        analizar_inestabilidad_1d(t_ref, y_ref, t0, tf, y0, timestamp)

if __name__ == "__main__":
    main()
