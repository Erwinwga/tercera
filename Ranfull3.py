import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

def formato_numero(valor):
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Configurar el layout de la página para ocupar todo el ancho disponible
st.set_page_config(layout="wide")

# Eliminar padding y márgenes innecesarios y centrar contenido
st.markdown(
    """
    <style>
    .css-18e3th9 {  
        padding: 0;
        margin: 0;
        width: 100%;
    }
    .css-1d391kg {  
        padding-left: 0;
        padding-right: 0;
    }
    .streamlit-expander {
        width: 90%;
        margin: 0 auto;
    }
    .css-1q8dd3e {  
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Colocar una firma al final de la página
st.markdown(
    """
    <style>
    .firma {
        position: fixed;
        bottom: 0;
        right: 0;
        font-size: 14px;
        color: gray;
        opacity: 0.7;
        margin: 10px;
    }
    </style>
    <div class="firma">AGBROTHERS</div>
    """,
    unsafe_allow_html=True
)
# URL directa de la imagen en GitHub
image_url = 'https://raw.githubusercontent.com/Erwinwga/tercera/main/agaleatorio.jpg'

# Mostrar la imagen usando Streamlit
st.image(image_url, caption="Este programa es propiedad de AGBROTHERS", width=800)

# Inicializar session_state para cada panel
if "panel1_data" not in st.session_state:
    st.session_state["panel1_data"] = None

# Crear las pestañas
tabs = st.tabs(["Panel 1", "Panel 2"])

# Pestaña "Panel 1"
with tabs[0]:
    st.title("Aleatorio")
    st.write("### TABLA DE RESULTADOS")

    # Menú de opciones en la barra lateral
    st.sidebar.header("Opciones")

    # Expander para el Panel 1
    with st.sidebar.expander("Opciones de Panel 1"):
        # Input para el Capital (en dólares)
        capital = st.number_input("Capital (USD)", min_value=0.0, value=10000.0)

        # Input para el Riesgo (en porcentaje)
        riesgo = st.number_input("Riesgo (%)", min_value=0.0, max_value=100.0, value=1.0)

        # Input para el porcentaje de acierto en la barra lateral (ahora es un campo para digitar)
        porcentaje_acierto = st.number_input("Porcentaje de Acierto (%)", min_value=0, max_value=100, value=70)

        # Input para el ratio
        ratio = st.number_input("Ratio", min_value=0.5, max_value=4.0, value=1.0, step=0.5)

        # Input para ingresar la cantidad de operaciones por mes
        operaciones_por_mes = st.number_input("Operaciones por Mes", min_value=1, max_value=100, value=20)
        
        # Input para ingresar los años de estudio
        anios_estudio = st.number_input("Años de Estudio", min_value=1, max_value=100, value=2)

        # Calcular la cantidad total de datos de estudio
        cantidad_numeros = operaciones_por_mes * anios_estudio * 12

        # Mostrar la cantidad de datos de estudio debajo del campo de operaciones por mes
        st.write(f"Cantidad de Datos de Estudio: {cantidad_numeros}") 

        # Input para la comisión por contrato
        comision_por_contrato = st.number_input("Comisión por contrato", min_value=0.0, value=5.0)

    # Botón Generar para Panel 1 fuera del expander
    if st.sidebar.button("Generar - Panel 1"):
        # Número total de valores basado en el input del usuario
        total_valores = cantidad_numeros

        # Calcular el número aproximado de aciertos y errores basados en el porcentaje
        num_aciertos_base = int(total_valores * porcentaje_acierto / 100)
        num_errores_base = total_valores - num_aciertos_base

        # Variación aleatoria de aciertos y errores
        variacion_aciertos = np.random.randint(-10, 11)  # Variar el número de aciertos en ±10
        num_aciertos = max(0, num_aciertos_base + variacion_aciertos)
        num_errores = total_valores - num_aciertos

        # Crear los valores aleatorios (1 para aciertos, -1 para errores)
        valores = np.array([1] * num_aciertos + [-1] * num_errores)

        # Mezclar aleatoriamente los valores (1 y -1)
        np.random.shuffle(valores)

        # Aplicar el ratio a los valores positivos
        valores = np.where(valores == 1, valores * ratio, valores)

        # Calcular el riesgo en función del capital
        riesgo_valor = capital * (riesgo / 100)

        # Columna "RR" con valores random entre 45 y el riesgo máximo calculado
        rr = np.random.randint(45, int(riesgo_valor), size=total_valores)

        # Calcular el número de contratos, redondeando hacia abajo
        contratos = np.floor(riesgo_valor / rr)

        # Calcular el riesgo real multiplicando "RR" por la cantidad de contratos
        riesgo_real = rr * contratos

        # Calcular la comisión multiplicando el valor de "Contratos" por la comisión por contrato
        comision = contratos * comision_por_contrato

        # Multiplicar cada valor de "Resultados" por el riesgo real para obtener el PNL
        pnl = valores * riesgo_real - comision  # Restar la comisión del PNL

        # Calcular la columna de Acumulado
        acumulado = np.cumsum(pnl)

        # Crear el DataFrame solo si todas las longitudes coinciden
        df = pd.DataFrame({
            "Resultados": valores,
            "RR": rr,
            "Contratos": contratos,
            "Comisión": comision,
            "Riesgo Real": riesgo_real,
            "PNL": pnl,
            "Acumulado": acumulado
        })

        # Calcular las rachas positivas y negativas
        cuenta_racha_negativa = []
        cuenta_racha_positiva = []
        actual_racha_negativa = 0
        actual_racha_positiva = 0

        for valor in pnl:
            if valor < 0:
                actual_racha_negativa += 1
                if actual_racha_positiva > 0:
                    for i in range(1, actual_racha_positiva + 1):
                        cuenta_racha_positiva.append(i)
                    actual_racha_positiva = 0
            elif valor > 0:
                actual_racha_positiva += 1
                if actual_racha_negativa > 0:
                    for i in range(1, actual_racha_negativa + 1):
                        cuenta_racha_negativa.append(i)
                    actual_racha_negativa = 0

        if actual_racha_negativa > 0:
            for i in range(1, actual_racha_negativa + 1):
                cuenta_racha_negativa.append(i)
        if actual_racha_positiva > 0:
            for i in range(1, actual_racha_positiva + 1):
                cuenta_racha_positiva.append(i)

        racha_range = range(1, 101)
        racha_negativa_counts = [cuenta_racha_negativa.count(i) for i in racha_range]
        racha_positiva_counts = [cuenta_racha_positiva.count(i) for i in racha_range]

        racha_df = pd.DataFrame({
            "Racha": racha_range,
            "Racha Negativa": racha_negativa_counts,
            "Racha Positiva": racha_positiva_counts
        })

        # Mostrar ambas tablas (Operaciones y Rachas) en el mismo expander, centrado
        with st.expander("Mostrar Detalles de Operaciones y Rachas"):
            col_op, col_racha = st.columns(2)
            with col_op:
                st.write(df)
            with col_racha:
                st.write(racha_df)

        # Calcular la rentabilidad
        rentabilidad = (acumulado[-1] / capital) * 100

        # Calcular el drawdown
        def max_drawdown_with_capital_and_percentage(acumulado, capital_inicial):
            peak = capital_inicial
            max_dd = 0
            start = end = 0
            for i in range(1, len(acumulado)):
                actual_value = capital_inicial + acumulado[i]
                if actual_value > peak:
                    peak = actual_value
                dd = peak - actual_value
                if dd > max_dd:
                    max_dd = dd
                    end = i
                    start = np.argmax([capital_inicial + a for a in acumulado[:i+1]])
            max_dd_percentage = (max_dd / peak) * 100
            return max_dd, max_dd_percentage, start, end

        max_dd, max_dd_percentage, start, end = max_drawdown_with_capital_and_percentage(acumulado, capital)

        # Calcular el valor mínimo alcanzado y su porcentaje respecto al capital inicial
        def corrected_min_value_percentage(acumulado, capital_inicial):
            min_value = np.min(acumulado + capital_inicial)
            min_value_percentage = ((capital_inicial - min_value) / capital_inicial) * 100  
            return min_value, min_value_percentage

        min_value_corrected, min_value_percentage_corrected = corrected_min_value_percentage(acumulado, capital)

        # Calcular el valor final acumulado
        valor_final = acumulado[-1] + capital

        # Calcular el profit factor
        total_ganancias = np.sum(pnl[pnl > 0])
        total_perdidas = -np.sum(pnl[pnl < 0])
        profit_factor = total_ganancias / total_perdidas if total_perdidas > 0 else np.inf

        # Contar cuántos valores corresponden a cada resultado
        count_positivos = np.sum(valores > 0)
        count_negativos = np.sum(valores == -1)

        # Cálculo del Recovery Factor
        recovery_factor = rentabilidad / max_dd if max_dd > 0 else np.inf

        # Cálculo del Sharpe Ratio
        media_pnl = np.mean(pnl)
        desviacion_pnl = np.std(pnl)
        sharpe_ratio = media_pnl / desviacion_pnl if desviacion_pnl > 0 else np.inf

        # Cálculo del Sortino Ratio
        retornos_negativos = pnl[pnl < 0]
        downside_deviation = np.std(retornos_negativos) if len(retornos_negativos) > 0 else 0
        sortino_ratio = media_pnl / downside_deviation if downside_deviation > 0 else np.inf

        # Cálculo de la Esperanza Matemática
        promedio_ganancias = np.mean(pnl[pnl > 0]) if len(pnl[pnl > 0]) > 0 else 0
        promedio_perdidas = np.mean(pnl[pnl < 0]) if len(pnl[pnl < 0]) > 0 else 0
        prob_ganancia = count_positivos / total_valores
        prob_perdida = count_negativos / total_valores

        esperanza_matematica = (prob_ganancia * promedio_ganancias) - (prob_perdida * promedio_perdidas)

        # Cálculo de la Desviación Estándar
        desviacion_estandar = np.std(pnl)

        # Cálculo de la Ganancia Media (Promedio de las operaciones ganadoras)
        ganancia_media = np.mean(pnl[pnl > 0]) if len(pnl[pnl > 0]) > 0 else 0

        # Cálculo de la Pérdida Media (Promedio de las operaciones perdedoras)
        perdida_media = np.mean(pnl[pnl < 0]) if len(pnl[pnl < 0]) > 0 else 0   

        # Cálculo del Ratio Riesgo-Beneficio (Risk-Reward Ratio)
        ratio_riesgo_beneficio = ganancia_media / abs(perdida_media) if perdida_media != 0 else np.inf

        # Métricas clave con tarjetas visuales
        st.write("### Métricas Claves")


        col3, col4, col5, col6, col7 = st.columns(5)

        # Grupo de métricas en el orden correcto
        col3.metric("Rentabilidad (%)", f"{formato_numero(rentabilidad)}%")
        col4.metric("Profit Factor", f"{formato_numero(profit_factor)}")
        col5.metric("Porcentaje de Acierto", f"{formato_numero((count_positivos / total_valores) * 100)}%")
        col6.metric("Esperanza Matemática", f"{formato_numero(esperanza_matematica)}")
        col7.metric("Valor Final (USD)", f"{formato_numero(valor_final)} USD")


        col8, col9, col10, col11,col12 = st.columns(5)

        col8.metric(f"Cantidad de {ratio:.1f}", f"{count_positivos:,}".replace(",", "."))
        col9.metric(f"Cantidad de -1", f"{count_negativos:,}".replace(",", "."))
        col10.metric("Máximo Drawdown (USD y %)", f"{formato_numero(max_dd)} ({formato_numero(max_dd_percentage)}%)")
        col11.metric("Max DD sobre Capital Inicial (%)", f"{formato_numero((max_dd / capital) * 100)}%")
        col12.metric("Recovery Factor", f"{formato_numero(recovery_factor)}")
        


        col13, col14, col15, col16,col17 = st.columns(5)

        col13.metric("Valor Mínimo (USD y %)", f"{formato_numero(min_value_corrected)} ({formato_numero(min_value_percentage_corrected)}%)")
        col14.metric("Ganancia Media", f"{formato_numero(ganancia_media)} USD")
        col15.metric("Pérdida Media", f"{formato_numero(perdida_media)} USD")
        col16.metric("Sharpe Ratio", f"{formato_numero(sharpe_ratio)}")
        col17.metric("Sortino Ratio", f"{formato_numero(sortino_ratio)}")


        col18,col19 = st.columns(2)
        col18.metric("Desviación Estándar", f"{formato_numero(desviacion_estandar)}")
        col19.metric("Ratio Riesgo-Beneficio", f"{formato_numero(ratio_riesgo_beneficio)}")

        # Gráfico interactivo con el capital acumulado
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(1, len(acumulado) + 1), y=acumulado + capital, mode='lines', name=f'Acumulado ({valor_final:.2f} USD)',line=dict(color='blue')))

        # Destacar el drawdown con el porcentaje
        fig.add_trace(go.Scatter(x=[start+1, end+1], y=[acumulado[start] + capital, acumulado[end] + capital], 
                                mode='markers+lines', name=f'Drawdown USD: {max_dd} ({max_dd_percentage:.2f}%)', 
                                line=dict(color='red', dash='dash'), marker=dict(size=8, color='red')))

        # Destacar el valor mínimo corregido
        fig.add_trace(go.Scatter(x=[np.argmin(acumulado) + 1], y=[min_value_corrected], 
                                mode='markers', marker=dict(size=12, color='firebrick'), 
                                name=f"Valor Mínimo USD: {min_value_corrected} ({min_value_percentage_corrected:.2f}%)"))

        # Añadir línea para el capital inicial
        fig.add_hline(y=capital, line_dash="dash", line_color="darkblue", annotation_text="CAPITAL INICIAL", 
                    annotation_position="top left", )

        # Agregar anotación del Acumulado Final
        fig.add_annotation(
            x=len(acumulado),  # Última operación
            y=acumulado[-1] + capital,  # Acumulado final
            text=f"Acumulado Final: {valor_final:.2f} USD",  # Texto de la anotación
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color="blue"),
            align="center",
            arrowcolor="blue",
            arrowsize=1.5,
            ax=0,  # Desplazamiento horizontal de la flecha
            ay=-40  # Desplazamiento vertical de la flecha
        )

        # Layout del gráfico
        fig.update_layout(
        title='Gráfico Acumulado con Máximo Drawdown y Valor Mínimo',
        xaxis_title='Cantidad de Operaciones',
        yaxis_title='Valor Acumulado (USD)',
        template='plotly_white',
        width=1800,
        height=750  # Ajuste de la altura del gráfico (puedes cambiar este valor para hacerlo más alto o más bajo)
        )

        # Mostrar el gráfico
        st.plotly_chart(fig)

        # Cálculo de PnL por mes
        pnl_por_mes = np.split(pnl, np.arange(operaciones_por_mes, len(pnl), operaciones_por_mes))
        pnl_por_mes = [np.sum(mes) for mes in pnl_por_mes]

        # Calcular la ganancia media y la pérdida media
        ganancia_media_mes = np.mean([x for x in pnl_por_mes if x > 0]) if len([x for x in pnl_por_mes if x > 0]) > 0 else 0
        perdida_media_mes = np.mean([x for x in pnl_por_mes if x < 0]) if len([x for x in pnl_por_mes if x < 0]) > 0 else 0

        # Cálculo del Profit Factor
        ganancias_totales = np.sum([x for x in pnl_por_mes if x > 0])
        perdidas_totales = abs(np.sum([x for x in pnl_por_mes if x < 0]))
        profit_factor = ganancias_totales / perdidas_totales if perdidas_totales != 0 else np.inf

        # Cálculo del Ratio Riesgo-Beneficio
        ratio_riesgo_beneficio = ganancia_media_mes / abs(perdida_media_mes) if perdida_media_mes != 0 else np.inf

        # Crear gráfico de barras para el PnL por mes
        fig_bar = go.Figure()

        # Colores condicionales: azul para positivo, rojo para negativo
        colors = ['Teal' if x > 0 else 'darkred' for x in pnl_por_mes]

        # Agregar las barras
        fig_bar.add_trace(go.Bar(
            x=[f"Mes {i+1}" for i in range(len(pnl_por_mes))],
            y=pnl_por_mes,
            marker_color=colors
        ))

        # Layout del gráfico de barras
        fig_bar.update_layout(title='Profit and Loss por Mes',
                            xaxis_title='Meses',
                            yaxis_title='Profit and Loss (USD)',
                            template='plotly_white',
                            width=1600,  # Ajuste del ancho del gráfico
                            height=500  # Ajuste del alto del gráfico
        )

        # Mostrar el gráfico de barras
        st.plotly_chart(fig_bar)

        # Cálculo de meses positivos y negativos
        meses_positivos = sum(1 for x in pnl_por_mes if x > 0)
        meses_negativos = sum(1 for x in pnl_por_mes if x < 0)
        total_meses = len(pnl_por_mes)

        # Probabilidades de meses positivos y negativos
        probabilidad_positivo = (meses_positivos / total_meses) * 100
        probabilidad_negativo = (meses_negativos / total_meses) * 100

        # Mostrar las etiquetas visuales
        st.write("### Estadísticas por Mes")
        col20, col21, col22, col23 = st.columns(4)
        col20.metric("Meses Positivos", f"{meses_positivos:,}".replace(",", "."))
        col21.metric("Meses Negativos", f"{meses_negativos:,}".replace(",", "."))
        col22.metric("Probabilidad Positiva (%)", f"{formato_numero(probabilidad_positivo)}%")
        col23.metric("Probabilidad Negativa (%)", f"{formato_numero(probabilidad_negativo)}%")

        col24, col25,col26, col27  = st.columns(4)
        col24.metric("Ganancia Media Mensual (USD)", f"{formato_numero(ganancia_media_mes)}")
        col25.metric("Pérdida Media Mensual (USD)", f"{formato_numero(perdida_media_mes)}")
        col26.metric("Ratio Riesgo-Beneficio", f"{formato_numero(ratio_riesgo_beneficio)}")
        col27.metric("Profit Factor", f"{formato_numero(profit_factor)}")

        # Cálculo de trimestres y PnL por trimestre
        trimestres = np.split(pnl_por_mes, np.arange(3, len(pnl_por_mes), 3))  # Dividir los meses en trimestres
        pnl_por_trimestre = [np.sum(trimestre) for trimestre in trimestres]

        # Cálculo del Profit Factor por trimestre
        ganancias_totales_trimestre = np.sum([x for x in pnl_por_trimestre if x > 0])
        perdidas_totales_trimestre = abs(np.sum([x for x in pnl_por_trimestre if x < 0]))
        profit_factor_trimestre = ganancias_totales_trimestre / perdidas_totales_trimestre if perdidas_totales_trimestre != 0 else np.inf

        # Cálculo del Ratio Riesgo-Beneficio por trimestre
        ganancia_media_trimestre = np.mean([x for x in pnl_por_trimestre if x > 0]) if len([x for x in pnl_por_trimestre if x > 0]) > 0 else 0
        perdida_media_trimestre = np.mean([x for x in pnl_por_trimestre if x < 0]) if len([x for x in pnl_por_trimestre if x < 0]) > 0 else 0
        ratio_riesgo_beneficio_trimestre = ganancia_media_trimestre / abs(perdida_media_trimestre) if perdida_media_trimestre != 0 else np.inf

        # Crear gráfico de barras para PnL por trimestre
        fig_trimestre_bar = go.Figure()
        colors_trimestre = ['#008080' if x > 0 else 'darkred' for x in pnl_por_trimestre]

        fig_trimestre_bar.add_trace(go.Bar(
            x=[f"Trimestre {i+1}" for i in range(len(pnl_por_trimestre))],
            y=pnl_por_trimestre,
            marker_color=colors_trimestre
        ))

        # Layout del gráfico de barras para trimestres
        fig_trimestre_bar.update_layout(title='Profit and Loss por Trimestre',
                                xaxis_title='Trimestres',
                                yaxis_title='Profit and Loss (USD)',
                                template='plotly_white',
                                width=1600,  # Ajuste del ancho del gráfico
                                height=500  # Ajuste del alto del gráfico
        )

        # Mostrar el gráfico de barras de trimestres
        st.plotly_chart(fig_trimestre_bar)

        # Cálculo de trimestres positivos y negativos
        trimestres_positivos = sum(1 for trimestre in trimestres if np.sum(trimestre) > 0)
        trimestres_negativos = sum(1 for trimestre in trimestres if np.sum(trimestre) < 0)
        total_trimestres = len(trimestres)

        # Probabilidades de trimestres positivos y negativos
        probabilidad_trimestre_positivo = (trimestres_positivos / total_trimestres) * 100
        probabilidad_trimestre_negativo = (trimestres_negativos / total_trimestres) * 100
        
        # Mostrar las estadísticas por trimestre
        st.write("### Estadísticas por Trimestre")
        col28, col29, col30, col31 = st.columns(4)
        col28.metric("Trimestres Positivos", f"{trimestres_positivos:,}".replace(",", "."))
        col29.metric("Trimestres Negativos", f"{trimestres_negativos:,}".replace(",", "."))
        col30.metric("Probabilidad Positiva (%)", f"{formato_numero(probabilidad_trimestre_positivo)}%")
        col31.metric("Probabilidad Negativa (%)", f"{formato_numero(probabilidad_trimestre_negativo)}%")

        col32, col33,col34, col35 = st.columns(4)
        col32.metric("Ganancia Media Trimestral (USD)", f"{formato_numero(ganancia_media_trimestre)}")
        col33.metric("Pérdida Media Trimestral (USD)", f"{formato_numero(perdida_media_trimestre)}")
        col34.metric("Ratio Riesgo-Beneficio", f"{formato_numero(ratio_riesgo_beneficio_trimestre)}")
        col35.metric("Profit Factor", f"{formato_numero(profit_factor_trimestre)}")

        ##### --- NUEVAS GRÁFICAS ANUALES ---##### 
        st.write("### Gráficas Anuales del Comportamiento Mensual y Acumulado")

        # Lista de meses para el eje X
        meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

        # Gráficas por cada año de estudio, tres gráficas en fila (operaciones, PnL mensual, acumulado)
        for year in range(1, anios_estudio + 1):
            st.write(f"#### Gráficas para el Año {year}")
            col_op, col_bar, col_acum = st.columns([3, 2, 2])  # Ajustar proporciones de las columnas
            # Índices para cada año: multiplicamos por 12 meses
            start_index = (year - 1) * 12
            end_index = year * 12  # No usar "operaciones_por_mes * 12" aquí, ya que estamos trabajando con datos mensuales

            # Verifica que pnl_por_mes tenga suficientes datos para cada año
            if len(pnl_por_mes) >= end_index:
                pnl_anual = pnl_por_mes[start_index:end_index]
            else:
                pnl_anual = []

            # Gráfico de curva acumulativa para todas las operaciones del año
            with col_op:
                st.write(f"Gráfico de Operaciones para el Año {year}")
                
                # Asegurarse de que se toman solo las operaciones correspondientes al año en curso
                start_index_anual = (year - 1) * operaciones_por_mes * 12
                end_index_anual = year * operaciones_por_mes * 12
                operaciones_anuales = pnl[start_index_anual:end_index_anual]  # Tomar solo las operaciones del año actual

                # Calcular el acumulado de operaciones del año
                acumulado_operaciones_anual = np.cumsum(operaciones_anuales)

                # Calcular el valor final acumulado del año
                valor_final_operaciones_anual = acumulado_operaciones_anual[-1] if len(acumulado_operaciones_anual) > 0 else 0


                # Función para calcular el máximo drawdown para el año
                def max_drawdown_anual(operaciones):
                    peak = operaciones[0]
                    max_dd = 0
                    start = end = 0
                    for i in range(1, len(operaciones)):
                        if operaciones[i] > peak:
                            peak = operaciones[i]
                        dd = peak - operaciones[i]
                        if dd > max_dd:
                            max_dd = dd
                            end = i
                            start = np.argmax(operaciones[:i+1])  # Encontrar el índice del pico máximo
                    max_dd_percentage = (max_dd / capital) * 100  # Cálculo basado en el capital inicial
                    return max_dd, max_dd_percentage, start, end

                # Calcular el máximo drawdown del año basado en las operaciones acumuladas del año
                max_dd_anual, max_dd_percentage_anual, start_anual, end_anual = max_drawdown_anual(acumulado_operaciones_anual)

                # Crear el gráfico de operaciones acumuladas para todo el año
                fig_operaciones_anual = go.Figure()

                # Trazar la línea del acumulado de operaciones para todo el año
                fig_operaciones_anual.add_trace(go.Scatter(
                    x=list(range(len(acumulado_operaciones_anual))),  # Incluir todas las operaciones del año
                    y=acumulado_operaciones_anual,
                    mode='lines',
                    line_color='green',
                    name=f"Acumulado Año {year}"
                ))

                # Agregar anotación del valor final acumulado del año
                fig_operaciones_anual.add_annotation(
                    x=len(acumulado_operaciones_anual),  # Última operación del año
                    y=valor_final_operaciones_anual,  # Valor final acumulado
                    text=f"Acumulado Final: {valor_final_operaciones_anual:.2f} USD",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=12, color="green"),
                    align="center",
                    arrowcolor="green",
                    arrowsize=1.5,
                    ax=0,  # Desplazamiento horizontal de la flecha
                    ay=-40  # Desplazamiento vertical de la flecha
                )

                # Agregar el marcador para el punto máximo desde donde comenzó el drawdown
                fig_operaciones_anual.add_trace(go.Scatter(
                    x=[start_anual], 
                    y=[acumulado_operaciones_anual[start_anual]], 
                    mode='markers',
                    marker=dict(size=8, color='Darkblue'),  # Punto máximo en azul
                    name="Pico Máx del DD"
                ))

                # Agregar el marcador para el punto del drawdown máximo
                fig_operaciones_anual.add_trace(go.Scatter(
                    x=[end_anual], 
                    y=[acumulado_operaciones_anual[end_anual]], 
                    mode='markers',
                    marker=dict(size=8, color='red'),  # Valle del drawdown en rojo
                    name=f'Máx DD: {max_dd_anual:.2f} USD ({max_dd_percentage_anual:.2f}%)'
                ))

                # Dibujar una línea discontinua que conecta el punto máximo con el drawdown
                fig_operaciones_anual.add_trace(go.Scatter(
                    x=[start_anual, end_anual],  # Conectar los puntos de pico y drawdown
                    y=[acumulado_operaciones_anual[start_anual], acumulado_operaciones_anual[end_anual]],
                    mode='lines+markers',
                    line=dict(color='firebrick', dash='dash'),  # Línea discontinua en rojo para destacar el drawdown
                    marker=dict(size=6, color='firebrick'),
                    name="Línea Drawdown"
                ))

                # Layout del gráfico de operaciones acumuladas
                fig_operaciones_anual.update_layout(
                    title=f'Operaciones Acumuladas para el Año {year}',
                    xaxis_title='Operaciones',
                    yaxis_title='Resultados (USD)',
                    template='plotly_white',
                    width=800,  # Ajustar el ancho del gráfico
                    height=500  # Ajustar el alto del gráfico
                )

                #  Mostrar el gráfico de operaciones acumuladas
                st.plotly_chart(fig_operaciones_anual)

            # Gráfico de barras para el comportamiento mensual
            with col_bar:
                st.write(f"Gráfico de Barras para el Año {year}")
                
                
                
                fig_anual_bar = go.Figure()
                colors_anual = ['#4682B4' if x > 0 else 'darkred' for x in pnl_anual]
                fig_anual_bar.add_trace(go.Bar(
                    x=meses[:len(pnl_anual)],
                    y=pnl_anual,
                    marker_color=colors_anual
                ))
                fig_anual_bar.update_layout(
                    title=f'PnL por Mes para el Año {year}',
                    xaxis_title='Meses',
                    yaxis_title='PnL (USD)',
                    template='plotly_white',
                    width=800,  # Ajustar el ancho del gráfico
                    height=500  # Ajustar el alto del gráfico
                )
                st.plotly_chart(fig_anual_bar)

            # Gráfico acumulado para el mismo año
            with col_acum:
                st.write(f"Gráfico Acumulado para el Año {year}")
                acumulado_anual = np.cumsum(pnl_anual)

                fig_acum = go.Figure()
                fig_acum.add_trace(go.Scatter(
                    x=meses[:len(acumulado_anual)],
                    y=acumulado_anual,
                    mode='lines',
                    line_color='#4682B4'
                ))
                fig_acum.update_layout(
                    title=f'Acumulado por Mes para el Año {year}',
                    xaxis_title='Meses',
                    yaxis_title='Acumulado (USD)',
                    template='plotly_white',
                    width=800,  # Ajustar el ancho del gráfico
                    height=500  # Ajustar el alto del gráfico
                )
                st.plotly_chart(fig_acum)

        ################## --- Expander Aleatorio Compuesto ---################################### 
        st.write("### Aleatorio Compuesto")

    #if 'df_compuesto' not in st.session_state:

        acumulado_compuesto = []
        capital_actual = capital

        for i in range(len(acumulado)):
            if i % (operaciones_por_mes * 12) == 0 and i != 0:
                capital_actual = acumulado_compuesto[-1] if acumulado_compuesto else capital_actual + capital
                riesgo_valor = capital_actual * (riesgo / 100)
                for j in range(i, len(rr)):
                    contratos[j] = np.floor(riesgo_valor / rr[j])
                    riesgo_real[j] = rr[j] * contratos[j]
                    comision[j] = contratos[j] * comision_por_contrato
                    pnl[j] = valores[j] * riesgo_real[j] - comision[j]

            acumulado_compuesto.append(pnl[i] + (acumulado_compuesto[-1] if i > 0 else capital_actual))

        # Calcular el máximo drawdown y el valor mínimo corregido para Acumulado Compuesto
        def max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital_inicial):
            peak = capital_inicial
            max_dd = 0
            start = end = 0
            for i in range(1, len(acumulado_compuesto)):
                actual_value = capital_inicial + acumulado_compuesto[i]
                if actual_value > peak:
                    peak = actual_value
                dd = peak - actual_value
                if dd > max_dd:
                    max_dd = dd
                    end = i
                    start = np.argmax([capital_inicial + a for a in acumulado_compuesto[:i+1]])
            max_dd_percentage = (max_dd / peak) * 100
            return max_dd, max_dd_percentage, start, end

        # Llamamos a la función para calcular el máximo drawdown para el acumulado compuesto
        max_dd_compuesto, max_dd_percentage_compuesto, start_compuesto, end_compuesto = max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital)

        df_compuesto = pd.DataFrame({
            "Resultados": valores,
            "RR": rr,
            "Contratos": contratos,
            "Comisión": comision,
            "Riesgo Real": riesgo_real,
            "PNL": pnl,
            "Acumulado": [a + capital for a in acumulado],
            "Acumulado Compuesto": acumulado_compuesto
        })
        # Guardar el DataFrame en el estado de la sesión
        #st.session_state['df_compuesto'] = df_compuesto
    #else:
        # Si ya existe, simplemente lo obtenemos de session_state
        #df_compuesto = st.session_state['df_compuesto']


        with st.expander("Aleatorio Compuesto - Panel 1"):
            st.write(df_compuesto)


        # Calcular el máximo drawdown y el valor mínimo corregido para Acumulado Compuesto
        def max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital_inicial):
            peak = capital_inicial
            max_dd = 0
            start = end = 0
            for i in range(1, len(acumulado_compuesto)):
                actual_value = capital_inicial + acumulado_compuesto[i]
                if actual_value > peak:
                    peak = actual_value
                dd = peak - actual_value
                if dd > max_dd:
                    max_dd = dd
                    end = i
                    start = np.argmax([capital_inicial + a for a in acumulado_compuesto[:i+1]])
            max_dd_percentage = (max_dd / peak) * 100
            return max_dd, max_dd_percentage, start, end

        # Calculo rentabilidad compuesto
        rentabilidad_compuesto = (acumulado_compuesto[-1] / capital) * 100

        # Calculo positvos y negativos
        count_positivos_compuesto = np.sum(np.array(valores) > 0)
        count_negativos_compuesto = np.sum(np.array(valores) < 0)
        porcentaje_acierto_compuesto = (count_positivos_compuesto / len(valores)) * 100

        valor_final_compuesto = acumulado_compuesto[-1]

        # Cálculo del Profit Factor para Acumulado Compuesto
        ganancias_totales_compuesto = np.sum(pnl[pnl > 0])  # Total de ganancias en el acumulado compuesto
        perdidas_totales_compuesto = abs(np.sum(pnl[pnl < 0]))  # Total de pérdidas en el acumulado compuesto
        profit_factor_compuesto = ganancias_totales_compuesto / perdidas_totales_compuesto if perdidas_totales_compuesto > 0 else np.inf

        # Cálculo de la Esperanza Matemática para Acumulado Compuesto
        promedio_ganancias_compuesto = np.mean(pnl[pnl > 0]) if len(pnl[pnl > 0]) > 0 else 0
        promedio_perdidas_compuesto = np.mean(pnl[pnl < 0]) if len(pnl[pnl < 0]) > 0 else 0
        prob_ganancia_compuesto = np.sum(pnl > 0) / len(pnl)
        prob_perdida_compuesto = np.sum(pnl < 0) / len(pnl)
        esperanza_matematica_compuesta = (prob_ganancia_compuesto * promedio_ganancias_compuesto) - (prob_perdida_compuesto * promedio_perdidas_compuesto)

        # Cálculo del Sharpe Ratio para Acumulado Compuesto
        media_pnl_compuesto = np.mean(pnl)
        desviacion_pnl_compuesto = np.std(pnl)
        sharpe_ratio_compuesto = media_pnl_compuesto / desviacion_pnl_compuesto if desviacion_pnl_compuesto > 0 else np.inf

        # Cálculo del Sortino Ratio para Acumulado Compuesto
        retornos_negativos_compuesto = pnl[pnl < 0]
        downside_deviation_compuesto = np.std(retornos_negativos_compuesto) if len(retornos_negativos_compuesto) > 0 else 0
        sortino_ratio_compuesto = media_pnl_compuesto / downside_deviation_compuesto if downside_deviation_compuesto > 0 else np.inf

        # Cálculo del Recovery Factor para Acumulado Compuesto
        recovery_factor_compuesto = (acumulado_compuesto[-1] / capital) / max_dd_compuesto if max_dd_compuesto > 0 else np.inf

        # Cálculo de la Ganancia Media y la Pérdida Media para Acumulado Compuesto
        ganancia_media_compuesta = promedio_ganancias_compuesto
        perdida_media_compuesta = promedio_perdidas_compuesto

        max_dd_compuesto, max_dd_percentage_compuesto, start_compuesto, end_compuesto = max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital)

        # Calcular el valor mínimo alcanzado y su porcentaje respecto al capital inicial
        def corrected_min_value_percentage(acumulado_compuesto, capital_inicial):
            # Convertir la lista acumulado_compuesto a un array de NumPy
            acumulado_compuesto_array = np.array(acumulado_compuesto)
    
            # Calcular el valor mínimo sumando el capital inicial    
            min_value = np.min(acumulado_compuesto_array + capital_inicial)
    
            # Calcular el porcentaje del valor mínimo respecto al capital inicial
            min_value_percentage = ((capital_inicial - min_value) / capital_inicial) * 100  
            return min_value, min_value_percentage

        # Llamar la función corregida
        min_value_corrected_compuesto, min_value_percentage_corrected_compuesto = corrected_min_value_percentage(acumulado_compuesto, capital)

        fig_compuesto = go.Figure()

        # Añadir la curva del Acumulado Compuesto
        fig_compuesto.add_trace(go.Scatter(
            x=np.arange(1, len(acumulado_compuesto) + 1),
            y=acumulado_compuesto,
            mode='lines',
            line_color='blue',
            name=f"Acumulado ({acumulado_compuesto[-1]:.2f} USD)"
            ))

        # Destacar el valor mínimo corregido directamente sin sumar el capital inicial
        fig_compuesto.add_trace(go.Scatter(
            x=[np.argmin(acumulado_compuesto) + 1], 
            y=[min_value_corrected_compuesto], 
            mode='markers', 
            marker=dict(size=12, color='firebrick'), 
            name=f"Valor Mínimo USD: {min_value_corrected_compuesto:.2f} ({min_value_percentage_corrected_compuesto:.2f}%)"
))

# ------------------------- Gráfico solo de Acumulado Compuesto -------------------------
        # Gráfico interactivo con el acumulado compuesto
        fig_acumulado_compuesto = go.Figure()

        # Trazar la línea del acumulado compuesto
        fig_acumulado_compuesto.add_trace(go.Scatter(
            x=np.arange(1, len(acumulado_compuesto) + 1), 
            y=acumulado_compuesto, 
            mode='lines', 
            name=f'Acumulado Compuesto ({acumulado_compuesto[-1]:.2f} USD)', 
            line=dict(color='blue'))
        )

        # Calcular el drawdown del Acumulado Compuesto
        max_dd_compuesto, max_dd_percentage_compuesto, start_compuesto, end_compuesto = max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital)

        # Añadir drawdown en el gráfico del Acumulado Compuesto
        fig_acumulado_compuesto.add_trace(go.Scatter(
            x=[start_compuesto, end_compuesto], 
            y=[acumulado_compuesto[start_compuesto], acumulado_compuesto[end_compuesto]], 
            mode='markers+lines', 
            name=f'Drawdown: {max_dd_compuesto:.2f} USD ({max_dd_percentage_compuesto:.2f}%)', 
            line=dict(color='red', dash='dash'),
            marker=dict(size=8, color='red')
        ))

        # Añadir línea para el capital inicial en el gráfico de Acumulado Compuesto
        fig_acumulado_compuesto.add_hline(y=capital, line_dash="dash", line_color="darkblue", annotation_text="CAPITAL INICIAL", 
                                annotation_position="top left")

        # Agregar anotación del Acumulado Compuesto Final
        valor_final_acumulado_compuesto = acumulado_compuesto[-1]
        fig_acumulado_compuesto.add_annotation(
            x=len(acumulado_compuesto),  # Última operación
            y=valor_final_acumulado_compuesto,  # Acumulado final
            text=f"Acumulado Final: {valor_final_acumulado_compuesto:.2f} USD",  # Texto de la anotación
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color="blue"),
            align="center",
            arrowcolor="blue",
            arrowsize=1.5,
            ax=0,  # Desplazamiento horizontal de la flecha
            ay=-40  # Desplazamiento vertical de la flecha
        )

        # Layout del gráfico solo de Acumulado Compuesto
        fig_acumulado_compuesto.update_layout(
            title='Gráfico Acumulado Compuesto con Máximo Drawdown',
            xaxis_title='Cantidad de Operaciones',
            yaxis_title='Valor Acumulado Compuesto (USD)',
            template='plotly_white',
            width=1800,
            height=750  # Ajuste de la altura del gráfico
        )

        # Mostrar el gráfico solo de Acumulado Compuesto
        st.plotly_chart(fig_acumulado_compuesto)

# ------------------------- Gráfico Comparativo de Acumulado y Acumulado Compuesto -------------------------
        # Gráfico interactivo comparativo de Acumulado y Acumulado Compuesto
        fig_comparativo = go.Figure()

        # Trazar la línea del acumulado (sin componer)
        fig_comparativo.add_trace(go.Scatter(
            x=np.arange(1, len(acumulado) + 1), 
            y=acumulado + capital, 
            mode='lines', 
            name=f'Acumulado ({acumulado[-1] + capital:.2f} USD)', 
            line=dict(color='green'))
        )

        # Trazar la línea del acumulado compuesto
        fig_comparativo.add_trace(go.Scatter(
            x=np.arange(1, len(acumulado_compuesto) + 1), 
            y=acumulado_compuesto, 
            mode='lines', 
            name=f'Acumulado Compuesto ({acumulado_compuesto[-1]:.2f} USD)', 
            line=dict(color='blue'))
        )

        # Calcular el drawdown para el acumulado (sin componer)
        max_dd_acumulado, max_dd_percentage_acumulado, start_acumulado, end_acumulado = max_drawdown_with_capital_and_percentage(acumulado, capital)

        # Añadir drawdown en el gráfico del acumulado (sin componer)
        fig_comparativo.add_trace(go.Scatter(
            x=[start_acumulado, end_acumulado], 
            y=[acumulado[start_acumulado] + capital, acumulado[end_acumulado] + capital], 
            mode='markers+lines', 
            name=f'Drawdown: {max_dd_acumulado:.2f} USD ({max_dd_percentage_acumulado:.2f}%)', 
            line=dict(color='red', dash='dash'),
            marker=dict(size=8, color='red')
        ))

        # Calcular el drawdown para el acumulado compuesto
        max_dd_compuesto, max_dd_percentage_compuesto, start_compuesto, end_compuesto = max_drawdown_with_capital_and_percentage(acumulado_compuesto, capital)

        # Añadir drawdown en el gráfico del acumulado compuesto
        fig_comparativo.add_trace(go.Scatter(
            x=[start_compuesto, end_compuesto], 
            y=[acumulado_compuesto[start_compuesto], acumulado_compuesto[end_compuesto]], 
            mode='markers+lines', 
            name=f'Drawdown Compuesto: {max_dd_compuesto:.2f} USD ({max_dd_percentage_compuesto:.2f}%)', 
            line=dict(color='darkred', dash='dash'),
            marker=dict(size=8, color='darkred')
        ))


        # Añadir línea para el capital inicial en el gráfico comparativo
        fig_comparativo.add_hline(y=capital, line_dash="dash", line_color="darkblue", annotation_text="CAPITAL INICIAL", 
                        annotation_position="top left")

        # Agregar anotación del Acumulado Final (sin componer)
        valor_final_acumulado = acumulado[-1] + capital
        fig_comparativo.add_annotation(
            x=len(acumulado),  # Última operación
            y=valor_final_acumulado,  # Acumulado final
            text=f"Acumulado Final: {valor_final_acumulado:.2f} USD",  # Texto de la anotación
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color="green"),
            align="center",
            arrowcolor="green",
            arrowsize=1.5,
            ax=0,  # Desplazamiento horizontal de la flecha
            ay=-40  # Desplazamiento vertical de la flecha
        )

        # Agregar anotación del Acumulado Compuesto Final
        fig_comparativo.add_annotation(
            x=len(acumulado_compuesto),  # Última operación
            y=valor_final_acumulado_compuesto,  # Acumulado compuesto final
            text=f"Acumulado Compuesto Final: {valor_final_acumulado_compuesto:.2f} USD",  # Texto de la anotación
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color="blue"),
            align="center",
            arrowcolor="blue",
            arrowsize=1.5,
            ax=0,  # Desplazamiento horizontal de la flecha
            ay=-40  # Desplazamiento vertical de la flecha
        )

        # Layout del gráfico comparativo
        fig_comparativo.update_layout(
            title='Comparativo: Acumulado vs Acumulado Compuesto con Máximo Drawdown',
            xaxis_title='Cantidad de Operaciones',
            yaxis_title='Valor Acumulado (USD)',
            template='plotly_white',
            width=1800,
            height=750  # Ajuste de la altura del gráfico
        )

        # Mostrar el gráfico comparativo de Acumulado y Acumulado Compuesto
        st.plotly_chart(fig_comparativo)


        # Métricas clave con tarjetas visuales (para Acumulado Compuesto)
        st.write("### Métricas Claves - Acumulado Compuesto")

        # Distribuir las métricas en grupos para visualizarlas de manera similar a la imagen
        col36, col37, col38, col39, col40 = st.columns(5)

        # Rentabilidad para Acumulado Compuesto (ya calculada)
        col36.metric("Rentabilidad (%)", f"{formato_numero(rentabilidad_compuesto)}%")

        # Profit Factor para Acumulado Compuesto (ya calculado)
        col37.metric("Profit Factor", f"{formato_numero(profit_factor_compuesto)}")

        # Porcentaje de acierto para Acumulado Compuesto (ya calculado)
        col38.metric("Porcentaje de Acierto", f"{formato_numero(porcentaje_acierto_compuesto)}%")

        # Esperanza Matemática para Acumulado Compuesto (ya calculada)
        col39.metric("Esperanza Matemática", f"{formato_numero(esperanza_matematica_compuesta)}")

        # Valor final del acumulado compuesto (ya calculado)
        col40.metric("Valor Final (USD)", f"{formato_numero(valor_final_compuesto)} USD")

        # Más métricas clave
        col41, col42, col43, col44, col45 = st.columns(5)

        # Cantidad de aciertos y errores (ya calculados)
        col41.metric(f"Cantidad de {ratio:.1f}", f"{count_positivos_compuesto:,}".replace(",", "."))
        col42.metric(f"Cantidad de -1", f"{count_negativos_compuesto:,}".replace(",", "."))

        # Máximo Drawdown y porcentaje de drawdown para Acumulado Compuesto (ya calculado)
        col43.metric("Máximo Drawdown (USD y %)", f"{formato_numero(max_dd_compuesto)} ({formato_numero(max_dd_percentage_compuesto)}%)")
        col44.metric("Max DD sobre Capital Inicial (%)", f"{formato_numero((max_dd_compuesto / capital) * 100)}%")

        # Recovery Factor para Acumulado Compuesto (ya calculado)
        col45.metric("Recovery Factor", f"{formato_numero(recovery_factor_compuesto)}")

        # Último grupo de métricas clave
        col46, col47, col48, col49, col50 = st.columns(5)

        # Valor mínimo alcanzado en el Acumulado Compuesto (ya calculado)
        col46.metric("Valor Mínimo (USD y %)", f"{formato_numero(min_value_corrected_compuesto)} ({formato_numero(min_value_percentage_corrected_compuesto)}%)")

        # Ganancia media y pérdida media para Acumulado Compuesto (ya calculados)
        col47.metric("Ganancia Media", f"{formato_numero(promedio_ganancias_compuesto)} USD")
        col48.metric("Pérdida Media", f"{formato_numero(promedio_perdidas_compuesto)} USD")

        # Sharpe Ratio para Acumulado Compuesto (ya calculado)
        col49.metric("Sharpe Ratio", f"{formato_numero(sharpe_ratio_compuesto)}")

        # Sortino Ratio para Acumulado Compuesto (ya calculado)
        col50.metric("Sortino Ratio", f"{formato_numero(sortino_ratio_compuesto)}")



#-------------------- NUEVAS GRÁFICAS ANUALES COMPUESTAS------------------------------------------
        


################################ PESTAÑA TWO #################################################################

# Panel 2: Inputs en el Sidebar
with st.sidebar.expander("Opciones de Panel 2"):
    # Inputs para Estrategia 1 (E1)
    st.write("#### Estrategia 1 (E1)")
    capital_e1 = st.number_input("Capital E1 (USD)", min_value=0.0, value=10000.0, key='capital_e1')
    riesgo_e1 = st.number_input("Riesgo E1 (%)", min_value=0.0, max_value=100.0, value=1.0, key='riesgo_e1')
    porcentaje_acierto_e1 = st.number_input("Porcentaje de Acierto E1 (%)", min_value=0, max_value=100, value=70, key='acierto_e1')
    ratio_e1 = st.number_input("Ratio E1", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key='ratio_e1')
    operaciones_por_mes_e1 = st.number_input("Operaciones por Mes E1", min_value=1, max_value=100, value=20, key='operaciones_e1')
    anios_estudio_e1 = st.number_input("Años de Estudio E1", min_value=1, max_value=100, value=2, key='anios_e1')
    comision_por_contrato_e1 = st.number_input("Comisión por Contrato E1", min_value=0.0, value=5.0, key='comision_e1')

    # Inputs para Estrategia 2 (E2)
    st.write("#### Estrategia 2 (E2)")
    capital_e2 = st.number_input("Capital E2 (USD)", min_value=0.0, value=10000.0, key='capital_e2')
    riesgo_e2 = st.number_input("Riesgo E2 (%)", min_value=0.0, max_value=100.0, value=1.0, key='riesgo_e2')
    porcentaje_acierto_e2 = st.number_input("Porcentaje de Acierto E2 (%)", min_value=0, max_value=100, value=70, key='acierto_e2')
    ratio_e2 = st.number_input("Ratio E2", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key='ratio_e2')
    operaciones_por_mes_e2 = st.number_input("Operaciones por Mes E2", min_value=1, max_value=100, value=20, key='operaciones_e2')
    anios_estudio_e2 = st.number_input("Años de Estudio E2", min_value=1, max_value=100, value=2, key='anios_e2')
    comision_por_contrato_e2 = st.number_input("Comisión por Contrato E2", min_value=0.0, value=5.0, key='comision_e2')

# Botón Generar para Panel 2
if st.sidebar.button("Generar - Panel 2"):
    # Generar los resultados aleatorios compartidos entre E1 y E2
    cantidad_numeros_e1 = operaciones_por_mes_e1 * anios_estudio_e1 * 12
    valores = np.random.choice([1, -1], size=cantidad_numeros_e1)  # Resultados compartidos entre E1 y E2
    rr = np.random.uniform(low=0.5, high=4.0, size=cantidad_numeros_e1)  # RR compartido

    # Cálculos para Estrategia 1 (E1)
    contratos_e1, comision_e1, riesgo_real_e1, pnl_e1, acumulado_n_e1, acumulado_s_e1, acumulado_compuesto_e1 = [], [], [], [], [], [], []
    capital_actual_e1 = capital_e1

    for i in range(cantidad_numeros_e1):
        riesgo_valor_e1 = capital_e1 * (riesgo_e1 / 100)
        contratos_e1.append(np.floor(riesgo_valor_e1 / rr[i]))
        riesgo_real_e1.append(contratos_e1[i] * rr[i])
        comision_e1.append(contratos_e1[i] * comision_por_contrato_e1)
        pnl_e1.append(valores[i] * riesgo_real_e1[i] - comision_e1[i])

        # Acumulados para E1
        acumulado_n_e1.append(pnl_e1[i] if i == 0 else pnl_e1[i] + acumulado_n_e1[i - 1])
        acumulado_s_e1.append(acumulado_n_e1[i] + capital_e1)
        acumulado_compuesto_e1.append(acumulado_s_e1[i] if i == 0 else pnl_e1[i] + acumulado_compuesto_e1[i - 1])

    # Cálculos para Estrategia 2 (E2)
    contratos_e2, comision_e2, riesgo_real_e2, pnl_e2, acumulado_n_e2, acumulado_s_e2, acumulado_compuesto_e2 = [], [], [], [], [], [], []
    capital_actual_e2 = capital_e2

    for i in range(cantidad_numeros_e1):
        riesgo_valor_e2 = capital_e2 * (riesgo_e2 / 100)
        contratos_e2.append(np.floor(riesgo_valor_e2 / rr[i]))
        riesgo_real_e2.append(contratos_e2[i] * rr[i])
        comision_e2.append(contratos_e2[i] * comision_por_contrato_e2)
        pnl_e2.append(valores[i] * riesgo_real_e2[i] - comision_e2[i])

        # Acumulados para E2
        acumulado_n_e2.append(pnl_e2[i] if i == 0 else pnl_e2[i] + acumulado_n_e2[i - 1])
        acumulado_s_e2.append(acumulado_n_e2[i] + capital_e2)
        acumulado_compuesto_e2.append(acumulado_s_e2[i] if i == 0 else pnl_e2[i] + acumulado_compuesto_e2[i - 1])

    # Mostrar ambos DataFrames dentro del mismo expander
    with st.expander("Resultados Aleatorio Compuesto - Panel 2"):
        st.write("### Estrategia 1")
        df_estrategia_1 = pd.DataFrame({
            "Resultados E1": valores,
            "RR E1": rr,
            "Contratos E1": contratos_e1,
            "Comisión E1": comision_e1,
            "Riesgo Real E1": riesgo_real_e1,
            "PNL E1": pnl_e1,
            "Acumulado N E1": acumulado_n_e1,
            "Acumulado S E1": acumulado_s_e1,
            "Acumulado Compuesto E1": acumulado_compuesto_e1,
        })
        st.write(df_estrategia_1)

        st.write("### Estrategia 2")
        df_estrategia_2 = pd.DataFrame({
            "Resultados E2": valores,
            "RR E2": rr,
            "Contratos E2": contratos_e2,
            "Comisión E2": comision_e2,
            "Riesgo Real E2": riesgo_real_e2,
            "PNL E2": pnl_e2,
            "Acumulado N E2": acumulado_n_e2,
            "Acumulado S E2": acumulado_s_e2,
            "Acumulado Compuesto E2": acumulado_compuesto_e2,
        })
        st.write(df_estrategia_2)
