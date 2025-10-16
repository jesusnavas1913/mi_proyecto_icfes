"""
charts.py
Utilidades de gráficos de calidad editorial (tipo examen/ICFES) basadas en Matplotlib/Seaborn.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# ============================
# Configuración y helpers
# ============================

_DEFAULT_FONT = "DejaVu Sans"
_COLOR_PALETTE = sns.color_palette("colorblind")
_BW_PALETTE = [
    (0.2, 0.2, 0.2),
    (0.55, 0.55, 0.55),
    (0.75, 0.75, 0.75),
    (0.35, 0.35, 0.35),
    (0.6, 0.6, 0.6),
]


def set_exam_theme(
    *,
    font: str = _DEFAULT_FONT,
    context: str = "paper",
    use_bw: bool = False,
    grid_alpha: float = 0.45,
    grid_lw: float = 0.6,
) -> None:
    """Aplica un tema sobrio y consistente tipo 'examen'."""
    sns.set_theme(context=context, style="whitegrid", font=font)
    sns.set_palette(_BW_PALETTE if use_bw else _COLOR_PALETTE)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": grid_lw,
            "grid.alpha": grid_alpha,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def fig_size_mm(width_mm: float, height_mm: float) -> Tuple[float, float]:
    """Convierte mm a pulgadas para figsize de Matplotlib."""
    return (width_mm / 25.4, height_mm / 25.4)


def save_figure(fig: plt.Figure, path: Optional[str], *, dpi: int = 400) -> None:
    """Guarda figura en vectorial (pdf/svg) o raster (png) con buena resolución."""
    if not path:
        return
    if path.lower().endswith((".svg", ".pdf")):
        fig.savefig(path)
    else:
        fig.savefig(path, dpi=dpi)


# ============================
# Gráfico de barras
# ============================

def bar_chart_exam(
    categorias: Sequence[str],
    valores: Sequence[Union[int, float]],
    *,
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ordenar_desc: bool = True,
    mostrar_valores: bool = True,
    valores_como_porcentaje: bool = False,
    paleta_bw: bool = False,
    ancho_mm: float = 150,
    alto_mm: float = 95,
    rot_x: int = 0,
    salida: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Dibuja barras limpias estilo examen con opciones de porcentaje y exportación.
    """
    if len(categorias) != len(valores):
        raise ValueError("'categorias' y 'valores' deben tener igual longitud")

    data = list(zip(categorias, valores))
    if ordenar_desc:
        data.sort(key=lambda x: x[1], reverse=True)
    cats, vals = zip(*data) if data else ([], [])

    # Tema
    set_exam_theme(use_bw=paleta_bw)

    fig, ax = plt.subplots(figsize=fig_size_mm(ancho_mm, alto_mm))

    # Barras
    sns.barplot(x=list(cats), y=list(vals), ax=ax, edgecolor="black", linewidth=0.7)

    # Eje Y como porcentaje si aplica
    if valores_como_porcentaje:
        total = sum(vals) if vals else 1
        if total <= 0:
            total = 1
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100 if max(vals) <= 1 else total))

    # Valores sobre barras
    if mostrar_valores:
        for p in ax.patches:
            v = p.get_height()
            if valores_como_porcentaje:
                # Detecta si los valores ya son 0-1 o 0-100
                if (max(vals) if vals else 1) <= 1:
                    label = f"{v*100:.1f}%"
                else:
                    total = sum(vals) if vals else 1
                    pct = (v / total * 100) if total else 0
                    label = f"{pct:.1f}%"
            else:
                label = f"{v:g}"
            ax.annotate(
                label,
                (p.get_x() + p.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    # Etiquetas y estilo
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=rot_x)

    save_figure(fig, salida)
    return fig, ax


# ============================
# Gráfico de pastel / donut
# ============================

def pie_chart_exam(
    labels: Sequence[str],
    valores: Sequence[Union[int, float]],
    *,
    titulo: str = "",
    donut: bool = True,
    paleta_bw: bool = False,
    max_categorias: int = 6,
    ancho_mm: float = 120,
    alto_mm: float = 120,
    mostrar_porcentaje: bool = True,
    salida: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Dibuja un gráfico circular tipo examen (donut opcional), validando número de categorías.
    """
    if len(labels) != len(valores):
        raise ValueError("'labels' y 'valores' deben tener igual longitud")
    if len(labels) == 0:
        raise ValueError("Se requiere al menos una categoría")
    if len(labels) > max_categorias:
        raise ValueError(
            f"Demasiadas categorías ({len(labels)}). Use <= {max_categorias} para legibilidad."
        )

    set_exam_theme(use_bw=paleta_bw)

    fig, ax = plt.subplots(figsize=fig_size_mm(ancho_mm, alto_mm))

    total = sum(valores)
    if total <= 0:
        total = 1

    def _fmt(pct: float) -> str:
        return f"{pct:.1f}%" if mostrar_porcentaje else ""

    wedges, texts, autotexts = ax.pie(
        valores,
        labels=None,
        startangle=90,
        counterclock=False,
        autopct=_fmt if mostrar_porcentaje else None,
        pctdistance=0.75 if donut else 0.62,
        wedgeprops=dict(linewidth=0.8, edgecolor="black"),
    )

    if donut:
        centre_circle = plt.Circle((0, 0), 0.55, color="white", fc="white", linewidth=0)
        fig.gca().add_artist(centre_circle)

    # Leyenda a la derecha con etiquetas limpias
    ax.legend(
        wedges,
        [str(l) for l in labels],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="",
    )

    ax.set_title(titulo, fontsize=12)
    ax.set_aspect("equal")

    save_figure(fig, salida)
    return fig, ax
