"""
Dashboard interativo para análise de dados de perfilagem de poços.
Dissertação (Rayza) 
"""

import os
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from flask import Flask, session, redirect, request, url_for, jsonify

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Carrega variáveis do arquivo .env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "perfilagem2025")
SECRET_KEY = os.getenv("SECRET_KEY", "dash-perfilagem-chave-secreta-2025")

# ============================================================
# Configurações gerais
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dados")
DEPTH_COL = "PROF (m)"

POCO_ALVO_CLIP = "4-BRSA-422-BA"

VARS_OFICIAIS = ["PROF (m)", "GR (API)", "RHO (g/cm³)", "PHI (dec)", "PE (b/e)", "DT (μs/ft)"]
VARS_PETROFISICAS = ["GR (API)", "RHO (g/cm³)", "PHI (dec)", "PE (b/e)", "DT (μs/ft)"]

OUTLIER_THRESHOLD = 3.5  # MAD (Iglewicz & Hoaglin)

LIMITES_FISICOS = {
    "GR (API)":    (0.0,  500.0),
    "RHO (g/cm³)": (0.8,    4.0),
    "PHI (dec)":   (0.0,    1.0),
    "PE (b/e)":    (0.0,   20.0),
    "DT (μs/ft)":  (35.0, 250.0),
}

NULL_CODES = {
    -999.25, -999.0, -9999.0, -9999.25, -99999.0,
    999.25, 999.0, 9999.0, 9999.25, 99999.0,
    1e30, -1e30
}
AUTO_NULL_ABS_THRESHOLD = 9000.0
EXTRA_NULL_CODES = {9966.0, 9996.0, 9999.0}

RHO_FILL_VALUE = 4.63
RHO_FILL_EPS = 0.02

APLICAR_CLIP_FISICO = True

# ============================================================
# Aparência
# ============================================================
FONT_BASE = 18
TITLE_SIZE = 22
LEGEND_SIZE = 15
LINE_WIDTH = 2.4

# Fonte apropriada para LaTeX (Computer Modern, Serif)
FONT_FAMILY = "Computer Modern, Times New Roman, serif"
FONT_COLOR = "#000000"

# Configuração de gráfico com opção de baixar com qualidade HD (SVG/PNG)
GRAPH_CONFIG = {
    "displaylogo": False, 
    "responsive": True, 
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png", # PNG de altíssima qualidade (funciona perfeito e fácil no Overleaf)
        "filename": "grafico_alta_resolucao",
        "height": 700,
        "width": 1200,
        "scale": 4 # Aumentando a escala para 4 gera uma imagem PNG com ótimo DPI (aprox 400dpi)
    }
}

# Configuração especial para Min-Max: sem altura fixa para incluir todos os poços no download
GRAPH_CONFIG_MINMAX = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "minmax_todos_pocos",
        "width": 1400,
        "scale": 3
    }
}
TEMPLATE = "plotly_white"

# Aplica globalmente o padrão de fonte para os gráficos
import plotly.io as pio
pio.templates[TEMPLATE].layout.font.family = FONT_FAMILY
pio.templates[TEMPLATE].layout.font.size = FONT_BASE
pio.templates[TEMPLATE].layout.font.color = FONT_COLOR
pio.templates[TEMPLATE].layout.title.font.family = FONT_FAMILY
pio.templates[TEMPLATE].layout.title.font.color = FONT_COLOR
pio.templates[TEMPLATE].layout.legend.font.family = FONT_FAMILY
pio.templates[TEMPLATE].layout.legend.font.color = FONT_COLOR
H_MAIN = 680

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dash_dissertacao")

# ============================================================
# Helpers de formatação
# ============================================================
def fmt_int_pt(x: int) -> str:
    return f"{int(x):,}".replace(",", ".")

def fmt_float_pt(x: float, nd=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ============================================================
# Utilitários
# ============================================================
def ensure_data_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.warning("Pasta de dados criada: %s (adicione seus .xlsx aqui)", path)

def listar_excels(base_path: str) -> List[str]:
    if not os.path.isdir(base_path):
        return []
    return sorted([f for f in os.listdir(base_path) if f.lower().endswith(".xlsx")])

def drop_unnamed_firstcol(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df
    col0 = str(df.columns[0]).strip().lower()
    if col0.startswith("unnamed"):
        return df.iloc[:, 1:].copy()
    return df

def corrigir_nomes_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {}
    for col in df.columns:
        c = str(col).strip()
        col_clean = c.lower()

        if col_clean == "prof (m)" or col_clean in ["prof(m)", "prof"]:
            rename_dict[c] = "PROF (m)"
        elif col_clean in ["gr (api)", "gr(api)", "gr api", "gr"]:
            rename_dict[c] = "GR (API)"
        elif col_clean in ["rho (g/cc)", "rho (g/cm3)", "rho (g/cm³)", "rho g/cc", "rho"]:
            rename_dict[c] = "RHO (g/cm³)"
        elif col_clean in ["phi (dec)", "phi", "phid", "porosidade", "porosity"]:
            rename_dict[c] = "PHI (dec)"
        elif col_clean in ["pe (b/e)", "pe(b/e)", "pe"]:
            rename_dict[c] = "PE (b/e)"
        elif col_clean in ["dt (uspf)", "dt (us/ft)", "dt uspf", "dt", "sonico", "sônico"]:
            rename_dict[c] = "DT (μs/ft)"

    return df.rename(columns=rename_dict)

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def suavizar(y: np.ndarray, janela: Optional[int]) -> np.ndarray:
    if janela is None or int(janela) < 3:
        return y
    janela = int(janela)
    janela = janela + (1 - janela % 2)
    return pd.Series(y).rolling(janela, center=True, min_periods=max(1, janela // 2)).mean().to_numpy()

def gerar_marks_slider(min_val: int, max_val: int) -> Dict[int, str]:
    step = 500
    marks: Dict[int, str] = {}
    aligned = min_val - (min_val % step)
    current = aligned
    while current <= max_val:
        if current >= 1000:
            label = f"{current/1000:.0f}k" if current % 1000 == 0 else f"{current/1000:.1f}k"
        else:
            label = str(current)
        marks[int(current)] = label
        current += step
    marks[int(min_val)] = str(int(min_val))
    marks[int(max_val)] = str(int(max_val))
    return marks

# ============================================================
# QC
# ============================================================
def _replace_null_codes(s: pd.Series, null_codes=NULL_CODES) -> Tuple[pd.Series, int]:
    x = to_numeric(s)
    xv = x.to_numpy(dtype=float)

    all_codes = set(null_codes) | set(EXTRA_NULL_CODES)
    mask_exact = np.zeros(len(xv), dtype=bool)
    for code in all_codes:
        if np.isfinite(code):
            mask_exact |= np.isclose(xv, float(code), atol=1e-9, rtol=0.0)

    mask_auto = np.isfinite(xv) & (np.abs(xv) >= AUTO_NULL_ABS_THRESHOLD)
    mask = mask_exact | mask_auto
    n = int(mask.sum())

    if n > 0:
        x = x.where(~mask, other=np.nan)
    return x, n

def aplicar_qc(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    info = {
        "phi_percent_conv": 0,
        "null_to_nan": {v: 0 for v in VARS_PETROFISICAS},
        "rho_fill_to_nan": 0,
        "clip_fisico": {v: 0 for v in VARS_PETROFISICAS},
    }

    for v in VARS_PETROFISICAS:
        if v in df.columns:
            s2, nnull = _replace_null_codes(df[v])
            info["null_to_nan"][v] = nnull
            df[v] = s2

    if "PHI (dec)" in df.columns:
        phi = to_numeric(df["PHI (dec)"])
        mask_percent = (phi > 1.0) & (phi <= 100.0)
        nconv = int(mask_percent.sum())
        if nconv:
            phi = phi.copy()
            phi.loc[mask_percent] = phi.loc[mask_percent] / 100.0
        info["phi_percent_conv"] = nconv
        df["PHI (dec)"] = phi

    if "RHO (g/cm³)" in df.columns:
        rho = to_numeric(df["RHO (g/cm³)"])
        mask_fill = np.isfinite(rho) & (np.abs(rho - RHO_FILL_VALUE) <= RHO_FILL_EPS)
        nfill = int(mask_fill.sum())
        if nfill:
            rho = rho.copy()
            rho.loc[mask_fill] = np.nan
        info["rho_fill_to_nan"] = nfill
        df["RHO (g/cm³)"] = rho

    if APLICAR_CLIP_FISICO:
        for v, (lo, hi) in LIMITES_FISICOS.items():
            if v in df.columns:
                s = to_numeric(df[v])
                finite = np.isfinite(s)
                before = s.copy()
                s2 = s.copy()
                s2.loc[finite] = s2.loc[finite].clip(lo, hi)
                info["clip_fisico"][v] = int(((s2 != before) & np.isfinite(before)).sum())
                df[v] = s2

    return df, info

# ============================================================
# MAD + z-score
# ============================================================
def mad_winsorize(series: pd.Series, threshold: float = 3.5) -> Tuple[pd.Series, np.ndarray]:
    x = to_numeric(series).to_numpy(dtype=float)
    finite = np.isfinite(x)

    if finite.sum() == 0:
        return pd.Series(x, index=series.index, name=series.name), np.zeros_like(x, dtype=bool)

    vals = x[finite]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))

    if mad == 0 or not np.isfinite(mad):
        return pd.Series(x, index=series.index, name=series.name), np.zeros_like(x, dtype=bool)

    mz = np.full_like(x, np.nan, dtype=float)
    mz[finite] = 0.6745 * (vals - med) / mad

    mask = np.zeros_like(x, dtype=bool)
    mask[finite] = np.abs(mz[finite]) > threshold

    delta = (threshold / 0.6745) * mad
    lo, hi = med - delta, med + delta

    x2 = x.copy()
    x2[mask] = np.clip(x2[mask], lo, hi)

    return pd.Series(x2, index=series.index, name=series.name), mask

def zscore_intra_poco(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    mean = df[cols].mean(skipna=True)
    std = df[cols].std(ddof=1, skipna=True)
    z = (df[cols] - mean) / std.replace(0, np.nan)
    const_cols = list(std.index[(std == 0) | (~np.isfinite(std))])
    if const_cols:
        z[const_cols] = 0.0
    df = df.copy()
    df[cols] = z
    return df

# ============================================================
# IO + cache com mtime
# ============================================================
def file_mtime(fp: str) -> float:
    try:
        return os.path.getmtime(fp)
    except Exception:
        return -1.0

def ler_excel(fp: str) -> pd.DataFrame:
    df = pd.read_excel(fp, engine="openpyxl").rename(columns=lambda c: str(c).strip())
    df = corrigir_nomes_variaveis(df)
    df = drop_unnamed_firstcol(df)
    return df

@lru_cache(maxsize=64)
def ler_para_series_cached(fp: str, mtime: float, threshold: float):
    return ler_para_series(fp, threshold=threshold)

def ler_para_series(fp: str, threshold: float = OUTLIER_THRESHOLD):
    df = ler_excel(fp)

    depth_col = DEPTH_COL if DEPTH_COL in df.columns else None
    if depth_col:
        df[depth_col] = to_numeric(df[depth_col])

    keep = [c for c in VARS_OFICIAIS if c in df.columns]
    df = df[keep].copy()

    if depth_col:
        df = df[df[depth_col].notna()].copy()

    n0 = int(len(df))
    poco = os.path.splitext(os.path.basename(fp))[0]

    df_qc, info_qc = aplicar_qc(df)

    cols = [c for c in df_qc.columns if c != depth_col and c in VARS_PETROFISICAS]

    outliers_por_variavel = {c: 0 for c in cols}
    prof_max_outlier = {c: None for c in cols}

    if depth_col and cols and n0 > 0:
        depth_vals = to_numeric(df_qc[depth_col]).to_numpy(dtype=float)
        for c in cols:
            s_clip, mask = mad_winsorize(df_qc[c], threshold=threshold)
            df_qc[c] = s_clip
            outliers_por_variavel[c] = int(mask.sum())
            if mask.any():
                vv = depth_vals[mask]
                mx = np.nanmax(vv) if np.any(np.isfinite(vv)) else None
                prof_max_outlier[c] = float(mx) if mx is not None and np.isfinite(mx) else None

    total_mad = int(sum(outliers_por_variavel.values()))
    df_z = zscore_intra_poco(df_qc, cols)
    corr_spear = df_z[cols].corr(method="spearman") if len(cols) >= 2 else pd.DataFrame()

    info = {
        "poco": poco,
        "total_original": n0,
        "qc_phi_percent_conv": int(info_qc.get("phi_percent_conv", 0)),
        "qc_null_to_nan": dict(info_qc.get("null_to_nan", {})),
        "qc_rho_fill_to_nan": int(info_qc.get("rho_fill_to_nan", 0)),
        "qc_clip_fisico": dict(info_qc.get("clip_fisico", {})),
        "total_clippados_mad": total_mad,
        "outliers_por_variavel": outliers_por_variavel,
        "prof_max_outlier": prof_max_outlier,
    }

    return df_z, corr_spear, depth_col, cols, info

def ler_para_series_raw(fp: str) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """Bruto (sem QC; sem MAD; sem z-score)."""
    df = ler_excel(fp)
    depth_col = DEPTH_COL if DEPTH_COL in df.columns else None
    if depth_col:
        df[depth_col] = to_numeric(df[depth_col])

    keep = [c for c in VARS_OFICIAIS if c in df.columns]
    df = df[keep].copy()
    if depth_col:
        df = df[df[depth_col].notna()].copy()

    cols = [c for c in df.columns if c != depth_col and c in VARS_PETROFISICAS]
    return df, depth_col, cols

# ============================================================
# Agregações globais
# ============================================================
def coletar_resumo_global(base_path: str, threshold: float = OUTLIER_THRESHOLD) -> Dict:
    resumo = {
        "total_linhas": 0,
        "phi_percent_conv": 0,
        "rho_fill_to_nan": 0,
        "null_to_nan_total": 0,
        "clip_fisico_total": 0,
        "mad_total": 0,
        "por_variavel": {v: {"null_to_nan": 0, "clip_fisico": 0, "mad": 0} for v in VARS_PETROFISICAS},
        "por_poco": {},
        "prof_max_outlier": {},
    }

    for arquivo in listar_excels(base_path):
        fp = os.path.join(base_path, arquivo)
        poco = os.path.splitext(arquivo)[0]
        try:
            mt = file_mtime(fp)
            _, _, _, _, info = ler_para_series_cached(fp, mt, threshold)
            n0 = int(info["total_original"])
            resumo["total_linhas"] += n0

            resumo["phi_percent_conv"] += int(info.get("qc_phi_percent_conv", 0))
            resumo["rho_fill_to_nan"] += int(info.get("qc_rho_fill_to_nan", 0))

            null_dict = info.get("qc_null_to_nan", {}) or {}
            clip_dict = info.get("qc_clip_fisico", {}) or {}
            mad_dict = info.get("outliers_por_variavel", {}) or {}

            null_tot = int(sum(int(null_dict.get(v, 0) or 0) for v in VARS_PETROFISICAS))
            clip_tot = int(sum(int(clip_dict.get(v, 0) or 0) for v in VARS_PETROFISICAS))
            mad_tot = int(sum(int(mad_dict.get(v, 0) or 0) for v in VARS_PETROFISICAS))

            resumo["null_to_nan_total"] += null_tot
            resumo["clip_fisico_total"] += clip_tot
            resumo["mad_total"] += mad_tot

            for v in VARS_PETROFISICAS:
                resumo["por_variavel"][v]["null_to_nan"] += int(null_dict.get(v, 0) or 0)
                resumo["por_variavel"][v]["clip_fisico"] += int(clip_dict.get(v, 0) or 0)
                resumo["por_variavel"][v]["mad"] += int(mad_dict.get(v, 0) or 0)

            resumo["por_poco"][poco] = {
                "linhas": n0,
                "null_to_nan": null_tot,
                "clip_fisico": clip_tot,
                "mad": mad_tot,
                "total": null_tot + clip_tot + mad_tot,
            }

            resumo["prof_max_outlier"][poco] = info.get("prof_max_outlier", {})

        except Exception as e:
            logger.error("Erro ao processar %s: %s", arquivo, e)

    print(f"INFO: Coleta global concluída ({resumo['total_linhas']} linhas).")
    return resumo

def coletar_minmax_por_poco_raw(base_path: str) -> pd.DataFrame:
    registros = []
    for f in listar_excels(base_path):
        fp = os.path.join(base_path, f)
        poco = os.path.splitext(f)[0]
        try:
            df, depth_col, cols = ler_para_series_raw(fp)

            if depth_col and depth_col in df.columns:
                prof = df[depth_col].dropna()
                if len(prof) > 0:
                    registros.append({"Poço": poco, "Variável": depth_col,
                                      "Min": float(prof.min()), "Max": float(prof.max())})

            for var in cols:
                v = to_numeric(df[var]).dropna()
                if len(v) > 0:
                    registros.append({"Poço": poco, "Variável": var,
                                      "Min": float(v.min()), "Max": float(v.max())})
        except Exception as e:
            logger.error("Erro minmax em %s: %s", f, e)

    out = pd.DataFrame(registros)
    if not out.empty:
        out = out[out["Variável"].isin(VARS_OFICIAIS)].copy()
    return out

def coletar_maximos_globais_raw(base_path: str) -> Dict[str, Dict]:
    maximos: Dict[str, Dict] = {}
    for f in listar_excels(base_path):
        fp = os.path.join(base_path, f)
        poco = os.path.splitext(f)[0]
        try:
            df, depth_col, cols = ler_para_series_raw(fp)
            if not cols or not depth_col or df.empty:
                continue

            for col in cols:
                v = to_numeric(df[col])
                max_val = v.max(skipna=True)
                if pd.isna(max_val):
                    continue

                if col not in maximos or float(max_val) > float(maximos[col]["valor"]):
                    idx = v.idxmax()
                    prof = df.loc[idx, depth_col] if idx in df.index else np.nan
                    maximos[col] = {
                        "poço": poco,
                        "valor": float(max_val),
                        "profundidade": float(prof) if pd.notna(prof) else np.nan,
                    }
        except Exception as e:
            logger.error("Erro máximos globais em %s: %s", f, e)

    return maximos

def selecionar_top_poco(base_path: str) -> Optional[str]:
    arquivos = listar_excels(base_path)
    if not arquivos:
        return None

    best = None
    best_max = -np.inf
    for f in arquivos:
        fp = os.path.join(base_path, f)
        try:
            df = ler_excel(fp)
            if DEPTH_COL in df.columns:
                prof = to_numeric(df[DEPTH_COL]).dropna()
                if len(prof) == 0:
                    continue
                mx = float(prof.max())
                if np.isfinite(mx) and mx > best_max:
                    best_max = mx
                    best = f
        except Exception as e:
            logger.warning("Falha ao avaliar top poço em %s: %s", f, e)

    return best if best is not None else arquivos[0]

# ============================================================
# Metadados (CSV opcional)
# ============================================================
INFO_POCOS_DF = pd.DataFrame({
    "Poço": ["3-BRSA-1148-SE","3-BRSA-1098-SE","3-BRSA-1093-SE","3-SZ-464-SE","3-BRSA-912-SE",
             "3-BRSA-1279-SE","3-BRSA-1262-SE","3-BRSA-1307D-SE","6-BRSA-1202D-BA","1-BRSA-470D-BA",
             "4-BRSA-422-BA","3-BRSA-349-RN","3-BRSA-1295-RN","3-BRSA-1193-RN","3-BRSA-1224-RN",
             "3-BRSA-1210-RN","3-BRSA-1325-RN","3-BRSA-1320-RN","3-BRSA-1341-RN","3-OGX-95-MA",
             "4-PGN-25-MA","1-OGX-117-MA","4-PGN-19-MA","7-GVB-8-MA","7-GVB-6-MA","7-RUC-59DP-AM",
             "3-BRSA-515-AM","3-BRSA-1057D-AM","3-BRSA-670-AM","3-BRSA-1121-AM","3-BRSA-1057D"],
    "Campo": ["Siririzinho","Siririzinho","Siririzinho","Siririzinho","Carmópolis","Carmópolis","Carmópolis","Carmópolis",
              "Dom João","Guanambi","CEXIS","Canto do Amaro","Canto do Amaro","Canto do Amaro","Estreito","Estreito",
              "Estreito","Alto do Rodrigues","Alto do Rodrigues","Gavião Branco","Gavião Tesoura","Gavião Tesoura",
              "Gavião Tesoura","Gavião Branco","Gavião Branco","Rio Urucu","Rio Urucu","Rio Urucu","Rio Urucu","Rio Urucu","NA"],
    "Bacia": ["Sergipe","Sergipe","Sergipe","Sergipe","Sergipe","Sergipe","Sergipe","Sergipe",
              "Recôncavo","Recôncavo","Recôncavo","Potiguar","Potiguar","Potiguar","Potiguar","Potiguar",
              "Potiguar","Potiguar","Potiguar","Parnaíba","Parnaíba","Parnaíba","Parnaíba","Parnaíba",
              "Parnaíba","Solimões","Solimões","Solimões","Solimões","Solimões","NA"]
})

def obter_informacoes_poco(arquivo: str, metadata: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
    chave = os.path.splitext(os.path.basename(arquivo))[0]
    row = INFO_POCOS_DF.loc[INFO_POCOS_DF["Poço"] == chave]
    if not row.empty:
        campo = row["Campo"].iloc[0] if pd.notna(row["Campo"].iloc[0]) else "(sem cadastro)"
        bacia = row["Bacia"].iloc[0] if pd.notna(row["Bacia"].iloc[0]) else "(sem cadastro)"
        return str(campo), str(bacia)
    return "(sem cadastro)", "(sem cadastro)"

# ============================================================
# UI helpers
# ============================================================
def card_metric(titulo: str, valor: str, subtitulo: str, header_color: str):
    return dbc.Card(
        [
            dbc.CardHeader(
                titulo,
                style={
                    "backgroundColor": header_color,
                    "color": "white",
                    "fontWeight": "700",
                    "fontSize": "14px",
                    "borderTopLeftRadius": "12px",
                    "borderTopRightRadius": "12px",
                    "letterSpacing": "0.2px",
                },
            ),
            dbc.CardBody(
                [
                    html.Div(valor, style={"fontSize": "26px", "fontWeight": "800", "lineHeight": "1"}),
                    html.Div(subtitulo, style={"marginTop": "8px", "color": "#6c757d", "fontSize": "13px"}),
                ],
                style={"padding": "14px"},
            ),
        ],
        style={
            "border": f"1px solid {header_color}",
            "borderRadius": "12px",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
        },
    )

# ============================================================
# Carregamento preguiçoso (Lazy Loading)
# ============================================================
_cache = {}

def get_resumo_global():
    if "resumo" not in _cache:
        print("INFO: Carregando resumo global...")
        _cache["resumo"] = coletar_resumo_global(DATA_PATH, threshold=OUTLIER_THRESHOLD)
    return _cache["resumo"]

def get_minmax_raw():
    if "minmax" not in _cache:
        print("INFO: Carregando min-max...")
        _cache["minmax"] = coletar_minmax_por_poco_raw(DATA_PATH)
    return _cache["minmax"]

def get_maximos_globais():
    if "maximos" not in _cache:
        print("INFO: Carregando máximos globais...")
        _cache["maximos"] = coletar_maximos_globais_raw(DATA_PATH)
    return _cache["maximos"]

ensure_data_dir(DATA_PATH)
arquivos = listar_excels(DATA_PATH)
valor_inicial = selecionar_top_poco(DATA_PATH) if arquivos else None
opcoes_dropdown = [{"label": f, "value": f} for f in arquivos]

# ============================================================
# Dash init
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"],
    suppress_callback_exceptions=True
)

# Configura a chave secreta de forma explícita no Flask
app.server.config['SECRET_KEY'] = SECRET_KEY
app.server.secret_key = SECRET_KEY

def empty_data_banner():
    return dbc.Alert(
        [
            html.Div("Nenhum arquivo .xlsx encontrado.", className="fw-bold"),
            html.Div("Coloque seus arquivos em:"),
            html.Code(DATA_PATH),
            html.Div("Depois reinicie o app."),
        ],
        color="warning",
        className="mt-2"
    )

controls_card = dbc.Card(
    [
        dbc.CardHeader("Controles", className="bg-primary text-white"),
        dbc.CardBody(
            [
                html.Label("Selecione o Poço:", className="fw-bold"),
                dcc.Dropdown(
                    id="arquivo-dropdown",
                    options=opcoes_dropdown,
                    value=valor_inicial,
                    clearable=False,
                    className="mb-2",
                    placeholder="Sem arquivos .xlsx em ./dados" if not arquivos else None
                ),
                dbc.Button("Atualizar", id="atualizar-btn", color="primary", className="w-100 mb-3",
                           disabled=(not arquivos)),

                html.Label("Variáveis a exibir:", className="fw-bold mt-2"),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Selecionar Todas", id="select-all-btn",
                                           color="outline-primary", size="sm", className="w-100 mb-1",
                                           disabled=(not arquivos)), width=6),
                        dbc.Col(dbc.Button("Desmarcar Todas", id="deselect-all-btn",
                                           color="outline-secondary", size="sm", className="w-100 mb-1",
                                           disabled=(not arquivos)), width=6),
                    ]
                ),
                dbc.Card(
                    dbc.CardBody([html.Div(id="vars-checkbox-container", children=[])]),
                    style={"maxHeight": "210px", "overflowY": "auto", "padding": "10px"},
                    className="mb-3"
                ),

                html.Label("Modo de visualização:", className="fw-bold"),
                dcc.RadioItems(
                    id="view-mode",
                    options=[{"label": " Sobreposto", "value": "overlay"},
                             {"label": " Múltiplos painéis", "value": "multiples"}],
                    value="overlay",
                    className="mb-3"
                ),

                html.Label("Suavização (janela):", className="fw-bold"),
                dcc.Slider(
                    id="smooth-window",
                    min=1, max=31, step=2, value=1,
                    marks={1: "Off", 7: "7", 15: "15", 31: "31"},
                    className="mb-3"
                ),

                html.Label("Filtro de profundidade (m):", className="fw-bold"),
                dcc.RangeSlider(
                    id="depth-range",
                    min=0, max=1000, value=[0, 1000], step=500,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-2"
                ),
                html.Div(id="depth-range-label", className="text-muted mt-1 small text-center"),

                html.Hr(),
                html.Div(id="campo-info", className="mb-2"),
                html.Div(id="bacia-info", className="mb-2"),
            ]
        ),
    ],
    className="mb-3 shadow"
)

app_layout = dbc.Container(
    [
        # ---- Cabeçalho com botão Sair ----
        dbc.Row(
            [
                dbc.Col(
                    html.H2("Análise de Dados de Perfilagem - Dissertação",
                            className="text-center my-3 text-primary"),
                    width=10
                ),
                dbc.Col(
                    html.A(
                        dbc.Button(
                            [html.I(className="bi bi-box-arrow-right me-1"), " Sair"],
                            color="outline-danger",
                            size="sm",
                            id="logout-btn",
                        ),
                        href="/logout",
                        style={"textDecoration": "none"}
                    ),
                    width=2,
                    className="d-flex align-items-center justify-content-end"
                ),
            ],
            className="align-items-center"
        ),

        dbc.Row(
            [
                dbc.Col(controls_card, width=3),
                dbc.Col(
                    [
                        dbc.Tabs(
                            id="tabs",
                            active_tab="series",
                            children=[
                                dbc.Tab(label="Séries de Perfilagem", tab_id="series"),
                                dbc.Tab(label="Correlação de Spearman", tab_id="correlacao"),
                                dbc.Tab(label="Remoções e Controle de Qualidade", tab_id="remocao"),
                                dbc.Tab(label="Min-Max", tab_id="minmax"),
                                dbc.Tab(label="Máximos Globais", tab_id="maximos"),
                            ],
                        ),
                        html.Div(id="tab-content", className="mt-2"),
                    ],
                    width=9
                ),
            ]
        ),
        html.Div(id="empty-data-banner", children=(empty_data_banner() if not arquivos else None)),
    ],
    fluid=True
)

# ============================================================
# Página de Login (HTML puro servido pelo Flask)
# ============================================================
LOGIN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Acesso ao Dashboard — Perfilagem de Poços</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      overflow: hidden;
    }
    /* Partículas decorativas */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        radial-gradient(circle at 20% 80%, rgba(32,162,255,.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(99,255,212,.12) 0%, transparent 50%);
      z-index: 0;
    }
    .card {
      position: relative;
      z-index: 1;
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 20px;
      padding: 48px 44px 44px;
      width: 100%;
      max-width: 420px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.5);
      animation: fadeUp .5s ease;
    }
    @keyframes fadeUp {
      from { opacity:0; transform: translateY(24px); }
      to   { opacity:1; transform: translateY(0); }
    }
    .icon {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 64px; height: 64px;
      background: linear-gradient(135deg, #20a2ff, #63ffd4);
      border-radius: 16px;
      margin: 0 auto 24px;
      font-size: 28px;
    }
    h1 {
      text-align: center;
      font-size: 1.35rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 4px;
      letter-spacing: -0.3px;
    }
    .subtitle {
      text-align: center;
      font-size: 0.85rem;
      color: rgba(255,255,255,0.55);
      margin-bottom: 32px;
    }
    label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      color: rgba(255,255,255,0.75);
      margin-bottom: 8px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .input-wrap {
      position: relative;
      margin-bottom: 20px;
    }
    .input-wrap input {
      width: 100%;
      padding: 14px 18px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 12px;
      font-size: 1rem;
      color: #fff;
      outline: none;
      transition: border-color .2s, background .2s;
    }
    .input-wrap input::placeholder { color: rgba(255,255,255,0.3); }
    .input-wrap input:focus {
      border-color: #20a2ff;
      background: rgba(32,162,255,0.1);
    }
    .toggle-pw {
      position: absolute;
      right: 14px; top: 50%;
      transform: translateY(-50%);
      cursor: pointer;
      font-size: 1.1rem;
      color: rgba(255,255,255,0.4);
      user-select: none;
      transition: color .2s;
    }
    .toggle-pw:hover { color: rgba(255,255,255,0.8); }
    .btn {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #20a2ff, #1a7cff);
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      cursor: pointer;
      letter-spacing: 0.3px;
      transition: opacity .2s, transform .1s;
    }
    .btn:hover { opacity: .9; }
    .btn:active { transform: scale(.98); }
    .error {
      background: rgba(255,70,70,0.15);
      border: 1px solid rgba(255,70,70,0.4);
      border-radius: 10px;
      color: #ff9898;
      font-size: 0.87rem;
      text-align: center;
      padding: 10px 14px;
      margin-bottom: 18px;
      animation: shake .3s ease;
    }
    @keyframes shake {
      0%,100% { transform: translateX(0); }
      25%      { transform: translateX(-6px); }
      75%      { transform: translateX(6px); }
    }
    .footer {
      text-align: center;
      font-size: 0.75rem;
      color: rgba(255,255,255,0.3);
      margin-top: 28px;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">🔬</div>
    <h1>Dashboard de Perfilagem</h1>
    <p class="subtitle">Dissertação — Rayza<br>Acesso restrito · insira a senha para continuar</p>

    {error_block}

    <form method="POST" action="/login">
      <label for="password">Senha de Acesso</label>
      <div class="input-wrap">
        <input id="password" name="password" type="password"
               placeholder="Digite a senha…" autocomplete="current-password" autofocus required/>
        <span class="toggle-pw" onclick="togglePw(this)">👁</span>
      </div>
      <button class="btn" type="submit">Entrar no Dashboard →</button>
    </form>

    <p class="footer">Acesso válido apenas neste navegador · sessão até fechar</p>
  </div>

  <script>
    function togglePw(el) {
      var inp = el.previousElementSibling;
      inp.type = inp.type === 'password' ? 'text' : 'password';
      el.textContent = inp.type === 'password' ? '👁' : '🙈';
    }
  </script>
</body>
</html>
"""

# ============================================================
# Rotas Flask: login / logout / proteção
# ============================================================
server = app.server  # o Flask subjacente do Dash

@server.route("/login", methods=["GET", "POST"])
def login():
    error_block = ""
    if request.method == "POST":
        senha = request.form.get("password", "")
        if senha == DASHBOARD_PASSWORD:
            session["authenticated"] = True
            return redirect("/")
        else:
            error_block = '<div class="error">⚠️ Senha incorreta. Tente novamente.</div>'
    
    try:
        page = LOGIN_PAGE_HTML.replace("{error_block}", error_block)
        return page
    except Exception as e:
        return f"Erro interno ao renderizar página de login: {e}", 500

@server.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@server.before_request
def require_login():
    """Bloqueia qualquer rota do Dash se o usuário não estiver autenticado."""
    # Rota atual limpa (sem query params e sem barras extras)
    path = request.path.rstrip("/")
    if not path: path = "/"
    
    # Lista de rotas públicas e assets
    if path == "/login" or path == "/favicon.ico" or path.startswith("/assets"):
        return None
        
    if not session.get("authenticated"):
        if path.startswith("/_dash"):
            return jsonify({"error": "Não autenticado"}), 401
        
        # Redireciona para /login
        return redirect("/login")

# ---- Define o layout do app (necessário ser definido após o server) ----
app.layout = app_layout

# ============================================================
# Diagnóstico do poço alvo (pós-QC) — antes vs depois MAD
# ============================================================
def diagnostico_poco_alvo(fp: str, threshold: float = OUTLIER_THRESHOLD):
    # diagnóstico é em pós-QC porque faz sentido explicar as alterações (MAD) após QC
    df = ler_excel(fp)
    depth_col = DEPTH_COL if DEPTH_COL in df.columns else None
    if depth_col:
        df[depth_col] = to_numeric(df[depth_col])

    keep = [c for c in VARS_OFICIAIS if c in df.columns]
    df = df[keep].copy()
    if depth_col:
        df = df[df[depth_col].notna()].copy()

    df_qc, _ = aplicar_qc(df)
    cols = [c for c in df_qc.columns if c != depth_col and c in VARS_PETROFISICAS]

    df_before = df_qc.copy()
    df_after = df_qc.copy()
    masks: Dict[str, np.ndarray] = {}

    for c in cols:
        s_clip, mask = mad_winsorize(df_after[c], threshold=threshold)
        df_after[c] = s_clip
        masks[c] = mask

    return df_before, df_after, depth_col, cols, masks

# ============================================================
# Callbacks
# ============================================================
@app.callback(
    Output({"type": "var-checkbox", "index": ALL}, "value"),
    Input("select-all-btn", "n_clicks"),
    Input("deselect-all-btn", "n_clicks"),
    State({"type": "var-checkbox", "index": ALL}, "id"),
    prevent_initial_call=True
)
def toggle_all_variables(n1, n2, checkbox_ids):
    if not checkbox_ids or not callback_context.triggered:
        raise PreventUpdate
    button_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if button_id == "select-all-btn":
        return [True] * len(checkbox_ids)
    if button_id == "deselect-all-btn":
        return [False] * len(checkbox_ids)
    raise PreventUpdate

@app.callback(
    Output("vars-checkbox-container", "children"),
    Output("depth-range", "min"),
    Output("depth-range", "max"),
    Output("depth-range", "value"),
    Output("depth-range", "marks"),
    Output("depth-range-label", "children"),
    Output("campo-info", "children"),
    Output("bacia-info", "children"),
    Input("atualizar-btn", "n_clicks"),
    State("arquivo-dropdown", "value"),
)
def preparar_controles(n_clicks, arquivo):
    if not arquivo or not n_clicks:
        raise PreventUpdate

    fp = os.path.join(DATA_PATH, arquivo)
    mt = file_mtime(fp)
    df_z, _, depth_col, cols, _ = ler_para_series_cached(fp, mt, OUTLIER_THRESHOLD)

    cols = [c for c in cols if c in VARS_PETROFISICAS]
    checkboxes = [
        html.Div([dbc.Checkbox(id={"type": "var-checkbox", "index": col}, value=True, label=col)],
                 className="mb-1")
        for col in cols
    ]

    if depth_col and depth_col in df_z.columns and not df_z.empty:
        prof = to_numeric(df_z[depth_col]).dropna()
        if len(prof) > 0:
            dmin = int(np.floor(float(prof.min())))
            dmax = int(np.ceil(float(prof.max())))
            if dmin == dmax:
                dmax = dmin + 1
            rng = [dmin, dmax]
            marks = gerar_marks_slider(dmin, dmax)
            label = f"Intervalo: {fmt_int_pt(dmin)} m – {fmt_int_pt(dmax)} m"
        else:
            dmin, dmax, rng = 0, 1000, [0, 1000]
            marks = {0: "0", 500: "500", 1000: "1k"}
            label = "Profundidade: dados inválidos"
    else:
        dmin, dmax, rng = 0, 1000, [0, 1000]
        marks = {0: "0", 500: "500", 1000: "1k"}
        label = "Sem coluna de profundidade"

    campo, bacia = obter_informacoes_poco(arquivo)
    campo_info = html.Div([html.Span("Campo: ", className="fw-bold"), html.Span(campo)])
    bacia_info = html.Div([html.Span("Bacia: ", className="fw-bold"), html.Span(bacia)])

    return checkboxes, dmin, dmax, rng, marks, label, campo_info, bacia_info

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("atualizar-btn", "n_clicks"),
    Input({"type": "var-checkbox", "index": ALL}, "value"),
    Input("view-mode", "value"),
    Input("smooth-window", "value"),
    Input("depth-range", "value"),
    State("arquivo-dropdown", "value"),
    State({"type": "var-checkbox", "index": ALL}, "id"),
)
def renderizar(active_tab, n_clicks, selected_values, view_mode, smooth_win,
               depth_rng, arquivo, checkbox_ids):
    if not arquivo:
        return dbc.Alert("Nenhum poço selecionado.", color="warning")
    if not n_clicks:
        return dbc.Alert("Selecione um poço e clique em Atualizar.", color="info")

    vars_sel: List[str] = []
    for i, v in enumerate(selected_values or []):
        if v and i < len(checkbox_ids):
            vars_sel.append(checkbox_ids[i]["index"])

    # ===== Máximos Globais (Bruto) =====
    if active_tab == "maximos":
        m_globais = get_maximos_globais()
        if not m_globais:
            return dbc.Alert("Sem dados suficientes para máximos globais.", color="info")

        df_max = pd.DataFrame([
            {
                "Variável": var,
                "Poço": info["poço"],
                "Máximo (Bruto)": f"{info['valor']:.4f}",
                "Profundidade (m)": fmt_float_pt(info["profundidade"], 1),
            }
            for var, info in m_globais.items()
        ])

        return html.Div([
            html.H4("Máximos Globais", className="text-primary mt-2"),
            dbc.Alert("Valores RAW podem conter sentinelas/fill. Use esta aba para auditoria do arquivo original.",
                      color="light", className="mt-2"),
            dbc.Table.from_dataframe(df_max, striped=True, bordered=True, hover=True,
                                     size="sm", className="mt-2")
        ])

    # ===== Remoções & QC (melhor estética) =====
    if active_tab == "remocao":
        r_global = get_resumo_global()
        tot = int(r_global.get("total_linhas", 0))
        null_total = int(r_global.get("null_to_nan_total", 0))
        clip_total = int(r_global.get("clip_fisico_total", 0))
        mad_total = int(r_global.get("mad_total", 0))
        phi_conv = int(r_global.get("phi_percent_conv", 0))
        rho_fill = int(r_global.get("rho_fill_to_nan", 0))

        cards = dbc.Row(
            [
                dbc.Col(card_metric(
                    "Nulos → NaN", fmt_int_pt(null_total),
                    "Sentinelas e valores absurdos convertidos para dado ausente (NaN).",
                    "#5A6472"), md=4),
                dbc.Col(card_metric(
                    "Clipping Físico", fmt_int_pt(clip_total),
                    f"Valores truncados ao intervalo físico de cada variável (winsorização por limites).",
                    "#1A8FC1"), md=4),
                dbc.Col(card_metric(
                    "Winsorização MAD", fmt_int_pt(mad_total),
                    f"Outliers estatísticos winsorisados pelo critério MAD (limiar = {OUTLIER_THRESHOLD}).",
                    "#E8691A"), md=4),
            ],
            className="g-3 mt-1"
        )

        resumo_badges = dbc.Row(
            [
                dbc.Col(dbc.Badge(
                    [html.Strong("Linhas analisadas: "), fmt_int_pt(tot)],
                    color="primary", pill=True, className="me-2 px-3 py-2"),
                    md="auto"),
                dbc.Col(dbc.Badge(
                    [html.Strong("PHI % → fração: "), fmt_int_pt(phi_conv)],
                    color="secondary", pill=True, className="me-2 px-3 py-2"),
                    md="auto"),
                dbc.Col(dbc.Badge(
                    [html.Strong("RHO fill value → NaN: "), fmt_int_pt(rho_fill)],
                    color="secondary", pill=True, className="me-2 px-3 py-2"),
                    md="auto"),
            ],
            className="g-2 mt-2",
            style={"flexWrap": "wrap"}
        )

        pv = r_global.get("por_variavel", {}) or {}
        df_stack = pd.DataFrame([
            {"Variável": v,
             "Nulos → NaN": int(pv.get(v, {}).get("null_to_nan", 0)),
             "Clipping Físico": int(pv.get(v, {}).get("clip_fisico", 0)),
             "Winsorização MAD": int(pv.get(v, {}).get("mad", 0))}
            for v in VARS_PETROFISICAS
        ])

        fig_stack = go.Figure()
        fig_stack.add_bar(x=df_stack["Variável"], y=df_stack["Nulos → NaN"],
                          name="Nulos → NaN", marker_color="#5A6472")
        fig_stack.add_bar(x=df_stack["Variável"], y=df_stack["Clipping Físico"],
                          name="Clipping Físico", marker_color="#1A8FC1")
        fig_stack.add_bar(x=df_stack["Variável"], y=df_stack["Winsorização MAD"],
                          name="Winsorização MAD", marker_color="#E8691A")
        fig_stack.update_layout(
            template=TEMPLATE,
            barmode="stack",
            title=dict(text="Intervenções de QC por variável (todos os poços)", x=0.5,
                       font=dict(size=TITLE_SIZE)),
            xaxis_title="Variável petrofísica",
            yaxis_title="Número de registros afetados",
            height=520,
            font=dict(size=FONT_BASE),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                        font=dict(size=LEGEND_SIZE),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            margin=dict(l=70, r=20, t=90, b=60),
        )

        # Tabela de limites — colunas separadas para Inferior e Superior
        df_lim = pd.DataFrame([
            {
                "Variável": v,
                "Limite Inferior": LIMITES_FISICOS[v][0],
                "Limite Superior": LIMITES_FISICOS[v][1],
                "Unidade": v.split("(")[-1].rstrip(")") if "(" in v else "—",
            }
            for v in VARS_PETROFISICAS
        ])
        lim_header = html.Thead(html.Tr([
            html.Th("Variável", style={"width": "40%"}),
            html.Th("Limite Inferior", style={"textAlign": "center"}),
            html.Th("Limite Superior", style={"textAlign": "center"}),
            html.Th("Unidade", style={"textAlign": "center"}),
        ]))
        lim_body = html.Tbody([
            html.Tr([
                html.Td(r["Variável"]),
                html.Td(
                    html.Span(str(r["Limite Inferior"]),
                              style={"color": "#2CA02C", "fontWeight": "600"}),
                    style={"textAlign": "center"}
                ),
                html.Td(
                    html.Span(str(r["Limite Superior"]),
                              style={"color": "#D62728", "fontWeight": "600"}),
                    style={"textAlign": "center"}
                ),
                html.Td(r["Unidade"], style={"textAlign": "center", "color": "#6c757d"}),
            ])
            for _, r in df_lim.iterrows()
        ])
        limites_tbl = dbc.Table(
            [lim_header, lim_body],
            bordered=True, striped=True, hover=True,
            size="sm", className="mt-2",
            style={"fontSize": "14px"}
        )

        # Ranking top 10 (opcional)
        por_poco = r_global.get("por_poco", {}) or {}
        if por_poco:
            df_rank = pd.DataFrame([{"Poço": p, **por_poco[p]} for p in por_poco]).sort_values("total", ascending=False)
            df_rank = df_rank.rename(columns={
                "linhas":      "Registros",
                "null_to_nan": "Nulos → NaN",
                "clip_fisico": "Clipping Físico",
                "mad":         "Winsor. MAD",
                "total":       "Total Intervenções",
            })
            df_rank["Registros"]           = df_rank["Registros"].map(fmt_int_pt)
            df_rank["Nulos → NaN"]         = df_rank["Nulos → NaN"].map(fmt_int_pt)
            df_rank["Clipping Físico"]     = df_rank["Clipping Físico"].map(fmt_int_pt)
            df_rank["Winsor. MAD"]         = df_rank["Winsor. MAD"].map(fmt_int_pt)
            df_rank["Total Intervenções"]  = df_rank["Total Intervenções"].map(fmt_int_pt)
            tabela_rank = dbc.Table.from_dataframe(df_rank.head(10), striped=True, bordered=True,
                                                   hover=True, size="sm", className="mt-2",
                                                   style={"fontSize": "13px"})
        else:
            tabela_rank = dbc.Alert("Sem dados para ranking.", color="info")

        return html.Div([
            cards,
            resumo_badges,
            html.Hr(className="mt-3"),
            html.H6(
                [html.Span("📐 ", style={"marginRight": "4px"}),
                 "Intervalos físicos aceitos por variável (clipping de QC)"],
                className="mt-2 text-secondary fw-semibold"
            ),
            dbc.Alert(
                "Valores fora do intervalo são truncados ao limite mais próximo (winsorização por limites físicos).",
                color="light", className="py-2 mb-1", style={"fontSize": "13px"}
            ),
            limites_tbl,
            html.Hr(),
            html.H6(
                [html.Span("📊 ", style={"marginRight": "4px"}),
                 "Intervenções de QC por variável petrofísica (todos os poços)"],
                className="mt-2 text-secondary fw-semibold"
            ),
            dcc.Graph(figure=fig_stack, config=GRAPH_CONFIG, style={"height": "520px"}),
            html.Hr(),
            html.H5(
                [html.Span("🏆 ", style={"marginRight": "4px"}),
                 "Top 10 poços com maior número de intervenções"],
                className="mt-2"
            ),
            tabela_rank,
        ])

    # ===== Min–Max (Bruto) =====
    if active_tab == "minmax":
        mm_raw = get_minmax_raw()
        if mm_raw is None or mm_raw.empty:
            return dbc.Alert("Sem dados para Min–Max (Bruto).", color="info")

        vars_disponiveis = [v for v in VARS_OFICIAIS if v in mm_raw["Variável"].unique()]
        vars_disponiveis = [v for v in vars_disponiveis if v != "PROF (m)"] + (
            ["PROF (m)"] if "PROF (m)" in vars_disponiveis else [])
        if not vars_disponiveis:
            return dbc.Alert("Sem variáveis disponíveis.", color="info")

        return html.Div([
            html.H4("Min-Max por poço", className="mt-2"),
            dbc.Alert("Esta aba usa os valores exatamente como estão no arquivo (pode conter sentinelas/fill).",
                      color="light"),
            dbc.Row(
                [
                    dbc.Col([
                        html.Label("Variável:", className="fw-bold"),
                        dcc.Dropdown(
                            id="minmax-var",
                            options=[{"label": v, "value": v} for v in vars_disponiveis],
                            value=vars_disponiveis[0],
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Filtro:", className="fw-bold"),
                        dcc.RadioItems(
                            id="minmax-filtro",
                            options=[{"label": " Todos os poços", "value": "all"},
                                     {"label": " Apenas com violação", "value": "viol"}],
                            value="all"
                        )
                    ], md=5),
                    dbc.Col([
                        html.Label("Ordenação:", className="fw-bold"),
                        dcc.Dropdown(
                            id="minmax-sort",
                            options=[{"label": "Por Max (crescente)", "value": "max_asc"},
                                     {"label": "Por Max (decrescente)", "value": "max_desc"},
                                     {"label": "Por Min (crescente)", "value": "min_asc"},
                                     {"label": "Por Min (decrescente)", "value": "min_desc"}],
                            value="max_asc",
                            clearable=False
                        )
                    ], md=3)
                ],
                className="g-2 mb-2"
            ),
            dcc.Graph(id="minmax-plot", config=GRAPH_CONFIG_MINMAX),
            html.Div(id="minmax-tabela-violacoes", className="mt-3"),
        ])

    # ===== Séries / Correlação =====
    fp = os.path.join(DATA_PATH, arquivo)
    mt = file_mtime(fp)
    df_z, corr, depth_col, cols, _ = ler_para_series_cached(fp, mt, OUTLIER_THRESHOLD)

    if not depth_col or depth_col not in df_z.columns or df_z.empty:
        return dbc.Alert("Arquivo sem profundidade válida (ou vazio após QC).", color="warning")

    if depth_rng:
        profz = to_numeric(df_z[depth_col])
        maskz = (profz >= depth_rng[0]) & (profz <= depth_rng[1])
        df_z = df_z.loc[maskz].reset_index(drop=True)

    if not vars_sel:
        vars_sel = [c for c in cols if c in VARS_PETROFISICAS]

    if active_tab == "series":
        if not vars_sel:
            return dbc.Alert("Selecione ao menos uma variável.", color="warning")

        x = df_z[depth_col].to_numpy()
        poco_nome = os.path.splitext(arquivo)[0]
        is_alvo = (poco_nome == POCO_ALVO_CLIP)
        palette = ["#3366CC", "#DC3912", "#109618", "#990099", "#0099C6", "#DD4477"]

        if view_mode == "overlay" or len(vars_sel) <= 3:
            fig = go.Figure()
            for i, c in enumerate(vars_sel):
                if c in df_z.columns:
                    y = suavizar(df_z[c].to_numpy(), smooth_win)
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode="lines",
                        name=f"{c} (Z)",
                        connectgaps=False,
                        line=dict(width=LINE_WIDTH, color=palette[i % len(palette)]),
                        hovertemplate=(f"<b>{c} (Z)</b><br>Profundidade: %{{x:.1f}} m"
                                       f"<br>Valor: %{{y:.3f}}<extra></extra>")
                    ))

            fig.update_layout(
                template=TEMPLATE,
                title=dict(text=f"Perfil de Poços – {poco_nome}", x=0.5, font=dict(size=TITLE_SIZE)),
                xaxis_title="Profundidade (m)",
                yaxis_title=" padronizado (intra-poço)",
                hovermode="x unified",
                height=H_MAIN,
                font=dict(size=FONT_BASE),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=LEGEND_SIZE)),
                margin=dict(l=65, r=15, t=70, b=60)
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=FONT_BASE))
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=FONT_BASE))

            children = [dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": f"{H_MAIN}px"})]

            if is_alvo:
                try:
                    df_bef, df_aft, dcol_b, _, masks = diagnostico_poco_alvo(fp, threshold=OUTLIER_THRESHOLD)
                    if dcol_b is None or dcol_b not in df_bef.columns:
                        raise ValueError("Sem coluna de profundidade no diagnóstico.")

                    if depth_rng:
                        profb = to_numeric(df_bef[dcol_b])
                        mask_diag = (profb >= depth_rng[0]) & (profb <= depth_rng[1])
                        df_bef = df_bef.loc[mask_diag].reset_index(drop=True)
                        df_aft = df_aft.loc[mask_diag].reset_index(drop=True)
                        masks = {k: (v[mask_diag.to_numpy()] if len(v) == len(mask_diag) else v) for k, v in masks.items()}

                    linhas = []
                    for v in vars_sel:
                        if v not in df_bef.columns or v not in df_aft.columns:
                            continue
                        vb = to_numeric(df_bef[v]).to_numpy(float)
                        va = to_numeric(df_aft[v]).to_numpy(float)
                        nb = int(np.isfinite(vb).sum())
                        na = int(np.isfinite(va).sum())
                        mb = float(np.nanmin(vb)) if nb else np.nan
                        xb = float(np.nanmax(vb)) if nb else np.nan
                        ma = float(np.nanmin(va)) if na else np.nan
                        xa = float(np.nanmax(va)) if na else np.nan
                        nclip = int(masks.get(v, np.zeros(len(vb), dtype=bool)).sum())
                        linhas.append((v, nb, mb, xb, na, ma, xa, nclip))

                    if linhas:
                        header = html.Thead(html.Tr([
                            html.Th("Variável"),
                            html.Th("n válidos (antes)"),
                            html.Th("mín (antes)"),
                            html.Th("máx (antes)"),
                            html.Th("n válidos (depois)"),
                            html.Th("mín (depois)"),
                            html.Th("máx (depois)"),
                            html.Th("# MAD winsor"),
                        ]))
                        body = []
                        for (v, nb, mb, xb, na, ma, xa, nclip) in linhas:
                            body.append(html.Tr([
                                html.Td(v),
                                html.Td(nb),
                                html.Td(f"{mb:.3g}" if np.isfinite(mb) else "—"),
                                html.Td(f"{xb:.3g}" if np.isfinite(xb) else "—"),
                                html.Td(na),
                                html.Td(f"{ma:.3g}" if np.isfinite(ma) else "—"),
                                html.Td(f"{xa:.3g}" if np.isfinite(xa) else "—"),
                                html.Td(nclip),
                            ]))
                        children.append(html.H5("Diagnóstico (pós-QC): antes vs. depois MAD — poço alvo", className="mt-3"))
                        children.append(dbc.Table([header, html.Tbody(body)],
                                                  bordered=True, striped=True, hover=True,
                                                  size="sm", className="mt-2"))
                except Exception as e:
                    children.append(dbc.Alert(f"Falha no diagnóstico do poço alvo: {e}", color="warning"))

            return html.Div(children)

        rows_n = len(vars_sel)
        fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        for i, c in enumerate(vars_sel, start=1):
            if c in df_z.columns:
                y = suavizar(df_z[c].to_numpy(), smooth_win)
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    name=f"{c} (Z)",
                    connectgaps=False,
                    line=dict(width=LINE_WIDTH, color=palette[(i - 1) % len(palette)]),
                ), row=i, col=1)
                fig.update_yaxes(title_text=f"{c} (Z)", row=i, col=1)

        h_multi = max(H_MAIN, 220 * rows_n + 160)
        fig.update_layout(
            template=TEMPLATE,
            title=dict(text=f"Perfil de Poços – {poco_nome}", x=0.5, font=dict(size=TITLE_SIZE)),
            height=h_multi,
            showlegend=False,
            font=dict(size=FONT_BASE),
            margin=dict(l=85, r=15, t=70, b=60)
        )
        fig.update_xaxes(title_text="Profundidade (m)", row=rows_n, col=1,
                         showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=FONT_BASE))
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=FONT_BASE))
        return dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": f"{h_multi}px"})

    if active_tab == "correlacao":
        if corr.empty or len(cols) < 2:
            return dbc.Alert("Dados insuficientes para correlação.", color="warning")

        use = [v for v in vars_sel if v in corr.index]
        if len(use) < 2:
            return dbc.Alert("Selecione ao menos 2 variáveis.", color="warning")

        # Manter unidades nos rótulos (mesmo com z-score adimensional) para facilitar leitura
        corr_f = corr.loc[use, use]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_f.values,
            x=list(corr_f.columns),
            y=list(corr_f.index),
colorscale="RdBu",
            zmin=-1, zmax=1, zmid=0,
            text=np.round(corr_f.values, 2),
            texttemplate="%{text}",
            hovertemplate="Variável (coluna): %{x}<br>Variável (linha): %{y}<br>Correlação de Spearman: %{z:.3f}<extra></extra>"
        ))
        poco_nome_corr = os.path.splitext(arquivo)[0]
        fig_corr.update_layout(
            template=TEMPLATE,
            title=dict(
                text=(
                    f"Correlação de Spearman — {poco_nome_corr}"
                    "<br><sup>Variáveis padronizadas por z-score intra-poço (adimensional)</sup>"
                ),
                x=0.5,
                font=dict(size=TITLE_SIZE),
            ),
            height=H_MAIN,
            font=dict(size=FONT_BASE),
            margin=dict(l=100, r=30, t=100, b=110),
            xaxis=dict(tickangle=-35, tickfont=dict(size=FONT_BASE)),
            yaxis=dict(tickfont=dict(size=FONT_BASE)),
        )
        return dcc.Graph(figure=fig_corr, config=GRAPH_CONFIG, style={"height": f"{H_MAIN}px"})

    return dbc.Alert("Selecione uma aba válida.", color="info")

# ============================================================
# Min–Max plot (Bruto) — legenda limites ajustada
# ============================================================
@app.callback(
    Output("minmax-tabela-violacoes", "children"),
    Input("minmax-var", "value"),
    Input("minmax-filtro", "value"), # Added this input to ensure table updates with filter
)
def render_minmax_table(var, filtro):
    mm_raw = get_minmax_raw()
    if mm_raw is None or mm_raw.empty or not var:
        return html.Div("Sem dados.")

    dfv = mm_raw[mm_raw["Variável"] == var].copy()
    if dfv.empty:
        return html.Div("Nenhum dado para a variável selecionada.")

    dfv["Min"] = dfv["Min"].astype(float)
    dfv["Max"] = dfv["Max"].astype(float)

    lim_lo, lim_hi = LIMITES_FISICOS.get(var, (None, None))
    if lim_lo is not None and lim_hi is not None:
        viol = (dfv["Min"] < lim_lo) | (dfv["Max"] > lim_hi)
    else:
        viol = pd.Series(False, index=dfv.index)

    dfv["Violação"] = viol
    if filtro == "viol":
        dfv = dfv[dfv["Violação"]].copy()

    # Bloco de violações (se existir limite)
    if lim_lo is not None and lim_hi is not None:
        df_viol = dfv[dfv["Violação"]].copy()
        nviol = int(df_viol.shape[0])
        if not df_viol.empty:
            df_show = df_viol[["Poço", "Min", "Max"]].copy()
            df_show = df_show.rename(columns={"Min": "Mínimo (Bruto)", "Max": "Máximo (Bruto)"})
            df_show["Mínimo (Bruto)"] = df_show["Mínimo (Bruto)"].map(lambda x: f"{x:.4g}".replace(".", ","))
            df_show["Máximo (Bruto)"] = df_show["Máximo (Bruto)"].map(lambda x: f"{x:.4g}".replace(".", ","))
            tbl = dbc.Table.from_dataframe(
                df_show,
                striped=True, bordered=True, hover=True, size="sm", className="mt-2"
            )
            bloco = html.Div([
                dbc.Alert(
                    [
                        html.Strong(f"{nviol} poço(s) "),
                        f"com valores fora do intervalo físico aceito — "
                        f"Limite inferior: {lim_lo}  |  Limite superior: {lim_hi}",
                    ],
                    color="warning", className="mb-2"
                ),
                tbl
            ])
        else:
            bloco = dbc.Alert(
                [html.Strong("✓ "), f"Nenhum poço com violação do intervalo [{lim_lo}, {lim_hi}]."],
                color="success"
            )
    else:
        bloco = dbc.Alert("Sem limites físicos definidos para esta variável.", color="info")
    
    return bloco


@app.callback(
    Output("minmax-plot", "figure"),
    Input("minmax-var", "value"),
    Input("minmax-filtro", "value"),
    Input("minmax-sort", "value"),
)
def plot_minmax(var, filtro, sort_mode):
    mm_raw = get_minmax_raw()
    if mm_raw is None or mm_raw.empty or not var:
        raise PreventUpdate

    dfv = mm_raw[mm_raw["Variável"] == var].copy()
    if dfv.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, title=dict(text=f"Min–Max — {var}", x=0.5), height=500)
        return fig

    dfv["Min"] = dfv["Min"].astype(float)
    dfv["Max"] = dfv["Max"].astype(float)

    lim_lo, lim_hi = LIMITES_FISICOS.get(var, (None, None))
    if lim_lo is not None and lim_hi is not None:
        viol = (dfv["Min"] < lim_lo) | (dfv["Max"] > lim_hi)
    else:
        viol = pd.Series(False, index=dfv.index)

    dfv["Violação"] = viol
    if filtro == "viol":
        dfv = dfv[dfv["Violação"]].copy()

    if sort_mode == "max_asc":
        dfv = dfv.sort_values(["Max", "Min"], ascending=[True, True])
    elif sort_mode == "max_desc":
        dfv = dfv.sort_values(["Max", "Min"], ascending=[False, True])
    elif sort_mode == "min_asc":
        dfv = dfv.sort_values(["Min", "Max"], ascending=[True, True])
    elif sort_mode == "min_desc":
        dfv = dfv.sort_values(["Min", "Max"], ascending=[False, True])

    if dfv.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, title=dict(text=f"Min–Max — {var}", x=0.5), height=500)
        return fig

    x_min = float(dfv["Min"].min())
    x_max = float(dfv["Max"].max())
    span = (x_max - x_min) if x_max > x_min else 1.0
    pad = 0.08 * span
    x_range = [x_min - pad, x_max + pad]

    fig_height = max(520, 28 * len(dfv) + 240)

    COR_OK   = "#3366CC"
    COR_VIOL = "#DC3912"

    fig = go.Figure()

    # Legenda via legendgroup: 1ª ocorrência de cada grupo mostra o nome
    legend_shown = {"ok": False, "viol": False}

    for _, r in dfv.iterrows():
        mn      = float(r["Min"])
        mx      = float(r["Max"])
        poco    = str(r["Poço"])          # garante string → eixo categórico
        is_viol = bool(r["Violação"])

        if is_viol:
            cor, grp, grp_name = COR_VIOL, "viol", "Fora do intervalo físico"
        else:
            cor, grp, grp_name = COR_OK,   "ok",   "Dentro do intervalo"

        show_leg = not legend_shown[grp]
        if show_leg:
            legend_shown[grp] = True

        fig.add_trace(go.Scatter(
            x=[mn, mx],
            y=[poco, poco],
            mode="lines+markers+text",
            text=[f"{mn:.4g}", f"{mx:.4g}"],
            textposition=["middle left", "middle right"],
            textfont=dict(size=16, color=cor),
            line=dict(width=5, color=cor),
            marker=dict(size=9, color=cor),
            name=grp_name,
            legendgroup=grp,
            showlegend=show_leg,
            hovertemplate=(
                f"<b>{poco}</b><br>"
                f"Mín: {mn:.4g}<br>"
                f"Máx: {mx:.4g}<extra></extra>"
            ),
        ))

    # ── Limites físicos ──────────────────────────────────────────────────
    # Anotações em y DIFERENTES (1.13 vs 1.05) → nunca se sobrepõem,
    # independentemente de quão próximos os limites estejam no eixo X.
    if lim_lo is not None:
        fig.add_vline(x=lim_lo, line_dash="dash", line_color="#2CA02C", line_width=1.8)
        fig.add_annotation(
            x=lim_lo, y=1.13, yref="paper",
            text=f"Lim. inferior: {lim_lo:.4g}",
            showarrow=False,
            xanchor="center",
            font=dict(size=11, color="#2CA02C"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#2CA02C", borderwidth=1, borderpad=3,
        )
    if lim_hi is not None:
        fig.add_vline(x=lim_hi, line_dash="dash", line_color="#D62728", line_width=1.8)
        fig.add_annotation(
            x=lim_hi, y=1.05, yref="paper",
            text=f"Lim. superior: {lim_hi:.4g}",
            showarrow=False,
            xanchor="center",
            font=dict(size=11, color="#D62728"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#D62728", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=var,
        yaxis_title="Poços",
        height=fig_height,
        font=dict(size=FONT_BASE),
        legend=dict(
            orientation="h", y=-0.10, x=0.5, xanchor="center",
            font=dict(size=LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.12)", borderwidth=1,
        ),
        margin=dict(l=170, r=30, t=120, b=90),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.07)",
        range=x_range, automargin=True,
    )
    fig.update_yaxes(
        type="category",          # ← ESSENCIAL: força eixo Y categórico (nomes dos poços)
        showgrid=True, gridcolor="rgba(0,0,0,0.04)",
        autorange="reversed", automargin=True,
    )

    return fig

# ============================================================
# Execução
# ============================================================
if __name__ == "__main__":
    logger.info("Dashboard iniciado. Acesse: http://localhost:8050")
    logger.info("Senha configurada via .env (DASHBOARD_PASSWORD)")
    app.run(debug=False, port=8050, host="0.0.0.0")