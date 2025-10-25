import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import branca.colormap as cm

# -------------------------
# Configuración y Estilos
# -------------------------
st.set_page_config(
    page_title="🚦 Sistema Analítico de Accidentes de Tránsito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar estilos personalizados (si existe)
style_path = Path("style.css")
if style_path.exists():
    with open(style_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

theme = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "background": "#ffffff",
    "text": "#2c3e50",
    "warning": "#e74c3c",
    "success": "#2ecc71",
}


# -------------------------
# Utilidades
# -------------------------
def calculate_trend(df, column):
    """Calcula pendiente y cambio porcentual entre inicio y fin de la serie."""
    try:
        if df is None or df.empty or column not in df.columns:
            return 0.0, np.nan
        series = df[column].dropna()
        if len(series) < 2:
            return 0.0, np.nan
        # regresión lineal para pendiente
        slope, _, _, _, _ = stats.linregress(range(len(series)), series.values)
        first = series.iloc[0]
        last = series.iloc[-1]
        if first == 0:
            pct = np.nan
        else:
            pct = (last - first) / first * 100.0
        return slope, pct
    except Exception:
        return 0.0, np.nan


@st.cache_data
def load_and_process_data():
    """Carga y preprocesa el dataset con validaciones básicas."""
    try:
        path = Path("Dataset_accidentes_transito.csv")
        if not path.exists():
            st.error(
                "No se encontró 'Dataset_accidentes_transito.csv' en la carpeta del script."
            )
            return None

        df = pd.read_csv(path, encoding="utf-8", low_memory=False)

        # Fecha
        df["FECHA"] = pd.to_datetime(df.get("FECHA"), errors="coerce")
        df = df.dropna(subset=["FECHA"])
        df["AÑO"] = df["FECHA"].dt.year
        df["MES"] = df["FECHA"].dt.month
        df["DÍA"] = df["FECHA"].dt.day
        dias = {
            "Monday": "Lunes",
            "Tuesday": "Martes",
            "Wednesday": "Miércoles",
            "Thursday": "Jueves",
            "Friday": "Viernes",
            "Saturday": "Sábado",
            "Sunday": "Domingo",
        }
        df["DÍA_SEMANA"] = df["FECHA"].dt.day_name().map(dias)

        # Hora (si existe)
        if "HORA" in df.columns:
            df["HORA_INT"] = pd.to_datetime(
                df["HORA"], format="%H:%M:%S", errors="coerce"
            ).dt.hour
        else:
            df["HORA_INT"] = pd.NA

        # Validar coordenadas (rango aproximado Perú)
        if {"LATITUD", "LONGITUD"}.issubset(df.columns):
            df = df[
                df["LATITUD"].between(-18.4, 0.5) & df["LONGITUD"].between(-81.4, -68.0)
            ]
        else:
            st.warning(
                "Las columnas LATITUD/LONGITUD no están completas; el mapa puede no funcionar correctamente."
            )

        # Métricas
        df["CANT_FALLECIDOS"] = (
            pd.to_numeric(df.get("CANT_FALLECIDOS", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["CANT_HERIDOS"] = (
            pd.to_numeric(df.get("CANT_HERIDOS", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["SEVERIDAD"] = (df["CANT_FALLECIDOS"] * 2 + df["CANT_HERIDOS"]) / 3
        df["MES_NOMBRE"] = df["FECHA"].dt.strftime("%B")

        # Normalizar nombres de columnas claves para evitar errores simples
        df.columns = [c.strip() for c in df.columns]

        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar datos: {e}")
        return None


# -------------------------
# Componentes UI
# -------------------------
def render_header():
    st.title("🚦 Análisis de Accidentes de Tránsito Perú 2020–2021")


def create_filters(df):
    with st.sidebar:
        st.header("🎯 Filtros de Análisis")

        # Fechas
        fecha_min = df["FECHA"].min().date()
        fecha_max = df["FECHA"].max().date()
        fecha_inicio = st.date_input(
            "Fecha Inicio", fecha_min, min_value=fecha_min, max_value=fecha_max
        )
        fecha_fin = st.date_input(
            "Fecha Fin", fecha_max, min_value=fecha_min, max_value=fecha_max
        )

        # Departamentos
        departamentos = st.multiselect(
            "Departamentos",
            options=(
                sorted(df["DEPARTAMENTO"].dropna().unique())
                if "DEPARTAMENTO" in df.columns
                else []
            ),
            default=(
                sorted(df["DEPARTAMENTO"].dropna().unique())
                if "DEPARTAMENTO" in df.columns
                else []
            ),
        )

        # Modalidades
        modalidades = st.multiselect(
            "Modalidades",
            options=(
                sorted(df["MODALIDAD"].dropna().unique())
                if "MODALIDAD" in df.columns
                else []
            ),
            default=(
                sorted(df["MODALIDAD"].dropna().unique())
                if "MODALIDAD" in df.columns
                else []
            ),
        )

        # Severidad
        min_sev = float(df["SEVERIDAD"].min()) if "SEVERIDAD" in df.columns else 0.0
        max_sev = float(df["SEVERIDAD"].max()) if "SEVERIDAD" in df.columns else 1.0
        severidad_range = st.slider(
            "Índice de Severidad", min_sev, max_sev, (min_sev, max_sev)
        )

        return {
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "departamentos": departamentos,
            "modalidades": modalidades,
            "severidad_range": severidad_range,
        }


def apply_filters(df, filters):
    mask = (df["FECHA"].dt.date >= filters["fecha_inicio"]) & (
        df["FECHA"].dt.date <= filters["fecha_fin"]
    )
    if "DEPARTAMENTO" in df.columns and filters["departamentos"]:
        mask &= df["DEPARTAMENTO"].isin(filters["departamentos"])
    if "MODALIDAD" in df.columns and filters["modalidades"]:
        mask &= df["MODALIDAD"].isin(filters["modalidades"])
    if "SEVERIDAD" in df.columns:
        mask &= df["SEVERIDAD"].between(*filters["severidad_range"])
    return df[mask]


def render_kpis(df_filtered):
    st.subheader("📊 Indicadores Clave (KPIs)")
    total_accidentes = len(df_filtered)
    total_fallecidos = (
        int(df_filtered["CANT_FALLECIDOS"].sum())
        if "CANT_FALLECIDOS" in df_filtered.columns
        else 0
    )
    total_heridos = (
        int(df_filtered["CANT_HERIDOS"].sum())
        if "CANT_HERIDOS" in df_filtered.columns
        else 0
    )
    severidad_promedio = (
        float(df_filtered["SEVERIDAD"].mean())
        if "SEVERIDAD" in df_filtered.columns
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # calcular delta simple: cambio % entre primer y último mes (si hay serie mensual)
        try:
            monthly = df_filtered.set_index("FECHA").resample("M").size()
            if len(monthly) >= 2:
                first = monthly.iloc[0]
                last = monthly.iloc[-1]
                if first == 0:
                    delta_total = "—"
                else:
                    pct_total = (last - first) / first * 100.0
                    delta_total = f"{pct_total:.1f}%"
            else:
                delta_total = "—"
        except Exception:
            delta_total = "—"
        st.metric("Total Accidentes", f"{total_accidentes:,}", delta=delta_total)
    # ...existing code...
    with col2:
        _, pct = calculate_trend(df_filtered, "CANT_FALLECIDOS")
        delta = f"{pct:.1f}%" if np.isfinite(pct) else "—"
        st.metric("Fallecidos", f"{total_fallecidos:,}", delta=delta)
    with col3:
        _, pct_h = calculate_trend(df_filtered, "CANT_HERIDOS")
        delta_h = f"{pct_h:.1f}%" if np.isfinite(pct_h) else "—"
        st.metric("Heridos", f"{total_heridos:,}", delta=delta_h)
    with col4:
        _, pct_s = calculate_trend(df_filtered, "SEVERIDAD")
        delta_s = f"{pct_s:.1f}%" if np.isfinite(pct_s) else "—"
        st.metric("Índice de Severidad", f"{severidad_promedio:.2f}", delta=delta_s)


def render_map(df_filtered):
    st.subheader("🗺️ Distribución Geográfica de Accidentes")
    # Mapa base centrado en Perú
    m = folium.Map(
        location=[-9.0, -75.0],
        zoom_start=6,
        tiles="cartodbpositron",
        width="100%",
        height="600px",
    )
    marker_cluster = MarkerCluster().add_to(m)

    if df_filtered.empty:
        st.info("No hay puntos para mostrar en el mapa con los filtros actuales.")
        folium_static(m, width=1200, height=600)
        return

    # Colormap
    try:
        vmin = float(df_filtered["SEVERIDAD"].min())
        vmax = float(df_filtered["SEVERIDAD"].max())
        colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax)
        colormap.caption = "Severidad (escala)"
        colormap.add_to(m)
    except Exception:
        colormap = None

    for _, row in df_filtered.iterrows():
        lat = row.get("LATITUD")
        lon = row.get("LONGITUD")
        if pd.isna(lat) or pd.isna(lon):
            continue
        sev = row.get("SEVERIDAD", 0)
        color = colormap(sev) if colormap is not None else "#3186cc"
        popup = folium.Popup(
            html=f"""
            <b>Fecha:</b> {row['FECHA'].strftime('%Y-%m-%d')}<br>
            <b>Modalidad:</b> {row.get('MODALIDAD','-')}<br>
            <b>Fallecidos:</b> {int(row.get('CANT_FALLECIDOS',0))}<br>
            <b>Heridos:</b> {int(row.get('CANT_HERIDOS',0))}<br>
            <b>Severidad:</b> {float(sev):.2f}
            """,
            max_width=300,
        )
        folium.CircleMarker(
            location=[lat, lon], radius=6, color=color, fill=True, popup=popup
        ).add_to(marker_cluster)

    folium_static(m, width=1200, height=600)


# -------------------------
# Main
# -------------------------
def main():
    df = load_and_process_data()
    if df is None:
        return

    render_header()
    filters = create_filters(df)
    df_filtered = apply_filters(df, filters)

    render_kpis(df_filtered)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗺️ Mapa y Distribución",
            "📈 Análisis Temporal",
            "🎯 Análisis por Categorías",
            "📊 Estadísticas Avanzadas",
        ]
    )

    with tab1:
        render_map(df_filtered)

    with tab2:
        st.subheader("📈 Análisis Temporal")
        if df_filtered.empty:
            st.info("No hay datos para el período o filtros seleccionados.")
        else:
            st.markdown(
                """
                - Evolución mensual de accidentes con promedio móvil (3 meses).
                - Heatmap hora vs día para ver concentraciones temporales.
                """
            )
            monthly = (
                df_filtered.set_index("FECHA").resample("M").size().rename("ACCIDENTES")
            )
            if monthly.empty:
                st.info("No hay series mensuales para mostrar.")
            else:
                monthly_df = monthly.to_frame()
                monthly_df["MA_3"] = (
                    monthly_df["ACCIDENTES"].rolling(3, min_periods=1).mean()
                )

                fig_monthly = go.Figure()
                fig_monthly.add_trace(
                    go.Scatter(
                        x=monthly_df.index,
                        y=monthly_df["ACCIDENTES"],
                        mode="lines+markers",
                        name="Accidentes",
                        line=dict(color=theme["primary"]),
                    )
                )
                fig_monthly.add_trace(
                    go.Scatter(
                        x=monthly_df.index,
                        y=monthly_df["MA_3"],
                        mode="lines",
                        name="Promedio móvil (3 meses)",
                        line=dict(color=theme["secondary"], dash="dash"),
                    )
                )
                fig_monthly.update_layout(
                    title="Evolución mensual de accidentes — con promedio móvil",
                    xaxis_title="Fecha",
                    yaxis_title="Accidentes",
                    template="plotly_white",
                    height=420,
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

            # Heatmap: hora vs día de la semana
            if (
                "DÍA_SEMANA" in df_filtered.columns
                and "HORA_INT" in df_filtered.columns
            ):
                pivot = (
                    df_filtered.groupby(["DÍA_SEMANA", "HORA_INT"])
                    .size()
                    .reset_index(name="COUNT")
                    .pivot(index="DÍA_SEMANA", columns="HORA_INT", values="COUNT")
                )
                order = [
                    "Lunes",
                    "Martes",
                    "Miércoles",
                    "Jueves",
                    "Viernes",
                    "Sábado",
                    "Domingo",
                ]
                pivot = pivot.reindex(order).fillna(0)
                fig_heat = px.imshow(
                    pivot,
                    labels=dict(
                        x="Hora del día", y="Día de la semana", color="Accidentes"
                    ),
                    x=list(pivot.columns),
                    y=list(pivot.index),
                    color_continuous_scale=["#e7f5e6", "#ffd966", "#ff6b6b"],
                )
                fig_heat.update_layout(
                    title="Concentración por hora y día (heatmap)", height=420
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info(
                    "Faltan columnas DÍA_SEMANA o HORA_INT para generar el heatmap."
                )

    with tab3:
        st.subheader("🎯 Análisis por Categorías")
        if df_filtered.empty:
            st.info("No hay datos para el período o filtros seleccionados.")
        else:
            st.markdown(
                "- Top departamentos por número de accidentes y proporción por modalidad (treemap)."
            )
            if "DEPARTAMENTO" in df_filtered.columns:
                dept_counts = df_filtered["DEPARTAMENTO"].value_counts().nlargest(10)
                fig_dept = px.bar(
                    x=dept_counts.values,
                    y=dept_counts.index,
                    orientation="h",
                    labels={"x": "Accidentes", "y": "Departamento"},
                    text=dept_counts.values,
                    color=dept_counts.values,
                    color_continuous_scale="Blues",
                )
                fig_dept.update_layout(
                    title="Top 10 departamentos por número de accidentes", height=380
                )
                st.plotly_chart(fig_dept, use_container_width=True)
            else:
                st.info("No hay columna DEPARTAMENTO en el dataset.")

            if "MODALIDAD" in df_filtered.columns:
                modal_agg = (
                    df_filtered.groupby("MODALIDAD")
                    .agg(
                        Accidentes=("MODALIDAD", "size"), Severidad=("SEVERIDAD", "sum")
                    )
                    .reset_index()
                )
                fig_tree = px.treemap(
                    modal_agg,
                    path=["MODALIDAD"],
                    values="Accidentes",
                    color="Severidad",
                    color_continuous_scale=["#8cc6ff", "#ffb18a", "#ff6b6b"],
                    labels={"MODALIDAD": "Modalidad"},
                )
                fig_tree.update_layout(
                    title="Proporción de accidentes por modalidad (color = severidad)",
                    height=420,
                )
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("No hay columna MODALIDAD en el dataset.")

    with tab4:
        st.subheader("📊 Estadísticas Avanzadas")
        if df_filtered.empty:
            st.info("No hay datos para mostrar estadísticas avanzadas.")
        else:
            st.markdown(
                "- Distribución de severidad por mes (boxplot) y mapa de correlaciones entre variables clave."
            )
            df_temp = df_filtered.copy()
            df_temp["MES_NRO"] = df_temp["FECHA"].dt.month

            if "SEVERIDAD" in df_temp.columns:
                fig_box = px.box(
                    df_temp,
                    x="MES_NRO",
                    y="SEVERIDAD",
                    labels={"MES_NRO": "Mes", "SEVERIDAD": "Índice de severidad"},
                    points="outliers",
                    color_discrete_sequence=[theme["primary"]],
                )
                fig_box.update_layout(
                    title="Distribución de severidad por mes", height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No hay columna SEVERIDAD para el boxplot.")

            corr_cols = [
                c
                for c in ["CANT_FALLECIDOS", "CANT_HERIDOS", "SEVERIDAD", "HORA_INT"]
                if c in df_temp.columns
            ]
            if len(corr_cols) >= 2:

                corr_df = df_temp[corr_cols].corr().round(2)
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    labels=dict(x="Variable", y="Variable", color="Correlación"),
                )
                fig_corr.update_layout(
                    title="Correlación entre variables clave", height=380
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                if (
                    "CANT_FALLECIDOS" in corr_df.index
                    and "SEVERIDAD" in corr_df.columns
                ):
                    c_fh = corr_df.loc["CANT_FALLECIDOS", "SEVERIDAD"]
                    if pd.isna(c_fh):
                        conclusion = "No hay suficiente información para calcular correlación entre fallecidos y severidad."
                    elif abs(c_fh) >= 0.5:
                        conclusion = f"La severidad está moderadamente/altamente correlacionada con fallecidos (r = {c_fh:.2f})."
                    else:
                        conclusion = f"No se observa una correlación fuerte entre severidad y fallecidos (r = {c_fh:.2f})."
                else:
                    conclusion = (
                        "No hay las columnas necesarias para evaluar la correlación."
                    )
                st.markdown(f"**Conclusión rápida:** {conclusion}")
            else:
                st.info(
                    "No hay suficientes columnas numéricas para calcular correlaciones."
                )


if __name__ == "__main__":
    main()



st.markdown(
    """
        ---
        <div style="text-align:center; color:gray; font-size:0.9em;">
            🚦 Sistema Analítico de Accidentes de Tránsito Perú 2020–2021<br>
            Desarrollado por <b>Sergio Carbajal</b> — © 2025<br>
            Datos referenciales del INEI.
        </div>
        """,
    unsafe_allow_html=True,
)

