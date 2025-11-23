import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================================================
# 1. LA CAJA NEGRA (MODELO MATEMÁTICO)
# =========================================================

def modelo_financiero(precio_venta, costo_materia_prima, costo_mano_obra, costos_fijos_mensuales, demanda_diaria, dias_operacion):
    """
    Función central (la 'Caja Negra') que calcula las métricas financieras clave.
    Ahora el costo variable es la suma de materia prima y mano de obra.
    """
    
    costo_variable_total = costo_materia_prima + costo_mano_obra

    # Cálculos Totales Mensuales
    demanda_mensual = demanda_diaria * dias_operacion
    ingresos_mensuales = demanda_mensual * precio_venta
    costos_variables_mensuales = demanda_mensual * costo_variable_total
    costo_total_mensual = costos_variables_mensuales + costos_fijos_mensuales
    utilidad_bruta_mensual = ingresos_mensuales - costo_total_mensual
    
    # Cálculo del Punto de Equilibrio
    margen_contribucion_unitario = precio_venta - costo_variable_total
    
    if margen_contribucion_unitario <= 0:
        pe_mensual = float('inf')
    else:
        pe_mensual = costos_fijos_mensuales / margen_contribucion_unitario

    pe_diario = pe_mensual / dias_operacion if dias_operacion > 0 else float('inf')

    # Devolver los resultados
    return {
        "demanda_diaria_simulada": demanda_diaria,
        "Utilidad Mensual (MXN)": utilidad_bruta_mensual,
        "Ingresos Mensuales (MXN)": ingresos_mensuales,
        "Costo Total Mensual (MXN)": costo_total_mensual,
        "Punto de Equilibrio (Garrafones/Mes)": pe_mensual,
        "Punto de Equilibrio (Garrafones/Día)": pe_diario,
    }

def ejecutar_simulacion(n_simulaciones, precio_venta, costo_materia_prima, costo_mano_obra, costos_fijos_mensuales, demanda_media, demanda_std, dias_operacion):
    """
    Ejecuta la simulación de Monte Carlo 'n' veces.
    """
    resultados_simulacion = []
    costo_variable_total = costo_materia_prima + costo_mano_obra

    for _ in range(n_simulaciones):
        # Generar una demanda diaria aleatoria (asegurando que no sea negativa)
        demanda_diaria_simulada = max(0, np.random.normal(loc=demanda_media, scale=demanda_std))
        
        # Calcular la utilidad para esta demanda simulada
        demanda_mensual = demanda_diaria_simulada * dias_operacion
        ingresos = demanda_mensual * precio_venta
        costos_variables = demanda_mensual * costo_variable_total
        utilidad = ingresos - (costos_variables + costos_fijos_mensuales)
        resultados_simulacion.append(utilidad)
        
    return resultados_simulacion

# =========================================================
# 2. INTERFAZ WEB CON STREAMLIT
# =========================================================

# --- Título y Configuración del Diseño ---
st.set_page_config(
    page_title="Simulación de Viabilidad: Agua Purificada",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💧 Proyecto de Simulación: Viabilidad Agua Purificada")
st.markdown("---")


# --- 3. BARRA LATERAL (ENTRADAS DE DATOS) ---
st.sidebar.header("⚙️ Parámetros de Simulación (Inputs)")

# Parámetros Base (Según tu archivo)
BASE_PARAMS = {
    "precio_venta": 20.0,
    "costo_materia_prima": 3.5,
    "costo_mano_obra": 2.0,
    "costos_fijos_mensuales": 25000.0,
    "demanda_media": 100,
    "demanda_std": 20,
    "dias_operacion": 30
}

# Sliders y Campos de Entrada
st.sidebar.subheader("1. Parámetros de Demanda (Aleatoriedad)")
demanda_media = st.sidebar.slider(
    "Demanda Diaria Promedio (Garrafones)",
    min_value=50, max_value=200, value=BASE_PARAMS["demanda_media"], step=5
)
demanda_std = st.sidebar.slider(
    "Desviación Estándar de la Demanda",
    min_value=5, max_value=50, value=BASE_PARAMS["demanda_std"], step=1,
    help="Qué tanto puede variar la demanda diaria. Un valor más alto significa más incertidumbre."
)
dias_operacion = st.sidebar.number_input(
    "Días de Operación al Mes",
    min_value=20, max_value=31, value=BASE_PARAMS["dias_operacion"], step=1
)

st.sidebar.subheader("2. Precios y Costos Variables")
precio_venta = st.sidebar.number_input(
    "Precio de Venta por Garrafón (MXN)",
    min_value=15.0, max_value=30.0, value=BASE_PARAMS["precio_venta"], step=0.5
)
costo_materia_prima = st.sidebar.number_input(
    "Costo de Materia Prima por Garrafón (MXN)",
    min_value=1.0, max_value=10.0, value=BASE_PARAMS["costo_materia_prima"], step=0.1
)
costo_mano_obra = st.sidebar.number_input(
    "Costo de Mano de Obra por Garrafón (MXN)",
    min_value=1.0, max_value=10.0, value=BASE_PARAMS["costo_mano_obra"], step=0.1
)

st.sidebar.subheader("3. Costos Fijos")
costos_fijos_mensuales = st.sidebar.number_input(
    "Costos Fijos Mensuales (MXN)",
    min_value=10000.0, max_value=50000.0, value=BASE_PARAMS["costos_fijos_mensuales"], step=1000.0
)

st.sidebar.subheader("4. Configuración de Simulación")
n_simulaciones = st.sidebar.select_slider(
    "Número de Simulaciones",
    options=[100, 500, 1000, 5000, 10000],
    value=1000,
    help="Número de escenarios aleatorios a simular. Más simulaciones dan un resultado más estable."
)

# Botón para ejecutar la simulación
run_simulation = st.sidebar.button("🚀 Iniciar Simulación")


def mostrar_metricas_principales(utilidad_promedio, pe_diario, margen_contribucion, prob_ganancia):
    """Función para mostrar las 4 métricas principales."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Utilidad Promedio Mensual",
        value=f"${utilidad_promedio:,.2f} MXN",
        help="El promedio de todas las utilidades mensuales simuladas."
    )
    col2.metric(
        label="Punto de Equilibrio Diario",
        value=f"{pe_diario:,.0f} Garrafones",
        help="Calculado con la demanda promedio. Es el # de garrafones a vender por día para cubrir todos los costos."
    )
    col3.metric(
        label="Margen de Contribución por Unidad",
        value=f"${margen_contribucion:,.2f} MXN",
        help="Ganancia bruta por cada garrafón vendido, antes de cubrir costos fijos."
    )
    col4.metric(
        label="Probabilidad de Ganancia",
        value=f"{prob_ganancia:.1f}%",
        help="Porcentaje de escenarios simulados donde la utilidad fue mayor a cero."
    )


# --- 4. EJECUCIÓN DEL MODELO Y SALIDAS (OUTPUTS) ---

if run_simulation:
    # --- 4.1 SIMULACIÓN PRINCIPAL (MONTE CARLO) ---
    resultados_simulacion = ejecutar_simulacion(
        n_simulaciones, precio_venta, costo_materia_prima, costo_mano_obra, 
        costos_fijos_mensuales, demanda_media, demanda_std, dias_operacion
    )

    costo_variable_total = costo_materia_prima + costo_mano_obra
    pe_mensual = costos_fijos_mensuales / (precio_venta - costo_variable_total) if (precio_venta - costo_variable_total) > 0 else float('inf')
    pe_diario = pe_mensual / dias_operacion if dias_operacion > 0 else float('inf')

    df_simulacion = pd.DataFrame(resultados_simulacion, columns=["Utilidad Mensual"])
    utilidad_promedio = df_simulacion["Utilidad Mensual"].mean()
    prob_ganancia = (df_simulacion["Utilidad Mensual"] > 0).mean() * 100
    margen_contribucion = precio_venta - costo_variable_total

    # --- 5. PRESENTACIÓN DE RESULTADOS ---
    st.header("📊 Resultados de la Simulación Principal")
    mostrar_metricas_principales(utilidad_promedio, pe_diario, margen_contribucion, prob_ganancia)

    st.markdown("---")

    st.subheader("Análisis de Incertidumbre de la Utilidad Mensual")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### Gráfico de Frecuencia (Histograma)")
        hist_chart = alt.Chart(df_simulacion).mark_bar().encode(
            alt.X("Utilidad Mensual:Q", bin=alt.Bin(maxbins=50), title="Utilidad Mensual (MXN)"),
            alt.Y('count()', title="Frecuencia (Nº de Escenarios)"),
            tooltip=[alt.Tooltip("Utilidad Mensual:Q", format="$,.2f"), 'count()']
        ).properties(
            title=f"Resultados de {n_simulaciones} Simulaciones"
        )
        st.altair_chart(hist_chart, use_container_width=True)
        st.info(
            """
            **¿Cómo leerlo?** Las barras altas muestran los resultados más comunes. Te dice **qué tan seguido** ocurre cada rango de utilidad.
            """
        )

    with col2:
        st.markdown("###### Gráfico de Probabilidad Acumulada (CDF)")
        
        # Crear los datos para el gráfico CDF
        df_cdf = df_simulacion.sort_values("Utilidad Mensual").copy()
        df_cdf['Probabilidad Acumulada'] = np.arange(1, len(df_cdf) + 1) / len(df_cdf)

        cdf_chart = alt.Chart(df_cdf).mark_line().encode(
            alt.X('Utilidad Mensual:Q', title="Utilidad Mensual (MXN)"),
            alt.Y('Probabilidad Acumulada:Q', axis=alt.Axis(format='%'), title="Probabilidad Acumulada"),
            tooltip=[
                alt.Tooltip('Utilidad Mensual:Q', title="Utilidad <= a", format="$,.2f"),
                alt.Tooltip('Probabilidad Acumulada:Q', title="Probabilidad", format='.1%')
            ]
        ).properties(
            title="Probabilidad de obtener una utilidad"
        )
        
        # Línea de referencia en el punto de 0 utilidad para ver la probabilidad de pérdida
        zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='red', strokeDash=[3,3]).encode(x='x')

        st.altair_chart(cdf_chart + zero_line, use_container_width=True)
        st.info(
            """
            **¿Cómo leerlo?** Sigue una línea vertical desde una utilidad (eje X) hasta la curva y luego a la izquierda. Te dice la **probabilidad de ganar *menos* de esa cantidad**. La línea roja marca el punto de cero ganancias.
            """
        )

    st.subheader("Veredicto del Modelo")
    if prob_ganancia > 75:
        st.success(f"**PRONÓSTICO VIABLE ({prob_ganancia:.1f}% de Prob. de Ganancia).** La demanda promedio ({demanda_media} garrafones/día) está por encima del punto de equilibrio ({pe_diario:,.0f} garrafones/día). La utilidad mensual promedio esperada es de **${utilidad_promedio:,.2f} MXN**.")
    elif prob_ganancia > 50:
        st.warning(f"**VIABILIDAD MODERADA ({prob_ganancia:.1f}% de Prob. de Ganancia).** Aunque hay una probabilidad mayor al 50% de obtener ganancias, existe un riesgo considerable de pérdidas. La utilidad mensual promedio esperada es de **${utilidad_promedio:,.2f} MXN**. Se recomienda analizar estrategias para aumentar el margen o la demanda.")
    else:
        st.error(f"**PRONÓSTICO NO VIABLE ({prob_ganancia:.1f}% de Prob. de Ganancia).** La probabilidad de obtener ganancias es baja. La demanda promedio ({demanda_media} garrafones/día) es probablemente inferior al punto de equilibrio ({pe_diario:,.0f} garrafones/día). La utilidad mensual promedio esperada es de **${utilidad_promedio:,.2f} MXN**, indicando una posible pérdida.")

    # --- 6. ANÁLISIS DE SENSIBILIDAD AUTOMÁTICO ---
    st.markdown("---")
    st.header("📈 Análisis de Sensibilidad")
    st.info(
        """
        A continuación se muestra cómo cambia la utilidad promedio al variar cada parámetro clave en un rango del +/- 20% de su valor actual.
        **Una línea más inclinada significa que el negocio es más sensible a los cambios en esa variable.**
        """
    )

    variables_a_analizar = {
        "Precio de Venta": precio_venta,
        "Costo de Materia Prima": costo_materia_prima,
        "Demanda Diaria Promedio": demanda_media
    }

    # Usar columnas para mostrar los gráficos
    col1, col2, col3 = st.columns(3)
    columnas = [col1, col2, col3]
    
    for i, (nombre_variable, valor_base) in enumerate(variables_a_analizar.items()):
        with columnas[i]:
            resultados_sensibilidad = []
            rango_analisis = np.linspace(valor_base * 0.80, valor_base * 1.20, 10) # 10 pasos

            for valor in rango_analisis:
                # Actualizar parámetros para la simulación actual
                current_precio = valor if nombre_variable == "Precio de Venta" else precio_venta
                current_costo_mp = valor if nombre_variable == "Costo de Materia Prima" else costo_materia_prima
                current_demanda = valor if nombre_variable == "Demanda Diaria Promedio" else demanda_media

                # Ejecutar simulación para este punto del análisis
                sim_results = ejecutar_simulacion(
                    int(n_simulaciones / 10), # Usar menos simulaciones para que sea más rápido
                    current_precio, current_costo_mp, costo_mano_obra, 
                    costos_fijos_mensuales, current_demanda, demanda_std, dias_operacion
                )
                
                # Guardar el resultado promedio
                utilidad_promedio_punto = np.mean(sim_results)
                resultados_sensibilidad.append({
                    "valor_variable": valor,
                    "utilidad_promedio": utilidad_promedio_punto
                })

            # Crear DataFrame y gráfico
            df_sensibilidad = pd.DataFrame(resultados_sensibilidad)
            
            chart = alt.Chart(df_sensibilidad).mark_line(point=True).encode(
                alt.X('valor_variable', title=nombre_variable, axis=alt.Axis(format='~s')),
                alt.Y('utilidad_promedio', title="Utilidad Promedio (MXN)", axis=alt.Axis(format='~s')),
            ).properties(
                title=f"Impacto de '{nombre_variable}'"
            )
            st.altair_chart(chart, use_container_width=True)
else:
    st.info("⬅️ Ajusta los parámetros y presiona 'Iniciar Simulación' para ver los resultados.")