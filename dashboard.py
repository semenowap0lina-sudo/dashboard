#!/usr/bin/env python3
"""Сборка единого dashboard.html из готовых результатов BlocksNet.

Скрипт только читает сохранённые файлы (GeoJSON, CSV, JSON) и собирает из них
одну HTML-страницу: интерактивную карту, графики, таблицы всех показателей
и автоматические выводы. Он не обращается к OpenStreetMap и не пересчитывает
показатели BlocksNet.

Запуск:
    python dashboard.py
    python dashboard.py --data-dir notebooks/.ipynb_checkpoints/data/vasileostrovsky
    python dashboard.py --out dashboard.html --no-open

Ожидаемая структура каталога данных:
    <data-dir>/results/blocks_metrics.geojson   (обязательно)
    <data-dir>/results/summary.csv
    <data-dir>/results/checks.csv
    <data-dir>/results/morphotype_counts.csv
    <data-dir>/results/spacematrix_clusters.csv
    <data-dir>/results/provision_links.csv
    <data-dir>/results/accessibility_matrix.csv
    <data-dir>/results/manifest.json
    <data-dir>/processed/schools.geojson
    <data-dir>/raw/boundary.geojson

Карта собрана на Leaflet, графики — на Plotly. Обе библиотеки встраиваются
в HTML, поэтому файл открывается без интернета; из сети подгружаются только
тайлы подложки (OpenStreetMap или Carto).
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "dashboard"

DEFAULT_DATA_CANDIDATES = [
    ROOT / "data" / "vasileostrovsky",
    ROOT / "notebooks" / "data" / "vasileostrovsky",
    ROOT / "notebooks" / ".ipynb_checkpoints" / "data" / "vasileostrovsky",
    ROOT / "notebooks" / "cache" / "data" / "vasileostrovsky",
]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

LEAFLET_VERSION = "1.9.4"
LEAFLET_CSS_URL = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.css"
LEAFLET_JS_URL = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.js"

LAND_USE_RU = {
    "residential": "жилая",
    "business": "деловая",
    "recreation": "рекреация",
    "industrial": "промышленная",
    "transport": "транспорт",
    "special": "специальная",
    "agriculture": "сельхоз",
}

LAND_USE_ORDER = ["residential", "business", "recreation", "industrial", "transport", "special", "agriculture"]

LAND_USE_COLORS = {
    "жилая": "#8dd3c7",
    "деловая": "#fdb462",
    "рекреация": "#b3de69",
    "промышленная": "#fb8072",
    "транспорт": "#80b1d3",
    "специальная": "#bebada",
    "сельхоз": "#ffffb3",
    "не назначено": "#d9d9d9",
}

MORPHO_RU = {
    "low-rise low-density residential": "малоэтажная низкоплотная жилая",
    "low-rise low-density non-residential": "малоэтажная низкоплотная нежилая",
    "low-rise low-density mixed-use": "малоэтажная низкоплотная смешанная",
    "low-rise model": "малоэтажная модель Strelka",
    "low-rise non-residential": "малоэтажная нежилая",
    "individual residential": "индивидуальная жилая",
}

COLOR_STOPS = {
    "population": ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
    "fsi": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "gsi": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "mxi": ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8", "#081d58"],
    "l": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
    "osr": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "capacity": ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
    "provision_weak": ["#a50026", "#f46d43", "#fee08b", "#66bd63", "#006837"],
    "site_area": ["#ffffff", "#f0f0f0", "#d9d9d9", "#969696", "#252525"],
}

NUMERIC_COLUMNS = [
    "block_id",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "land_use_share",
    "site_area",
    "population",
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "capacity",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "osr",
    "share_living",
    "share_non_living",
    "spacematrix_cluster",
    "demand",
    "provision_strong",
    "provision_weak",
    "demand_within",
    "demand_without",
    "area_accessibility",
]

COLUMNS = [
    ("block_id", "Квартал", "int", "Идентификатор квартала"),
    ("land_use_ru", "Землепользование", "text", "Преобладающая категория земли"),
    ("land_use_share", "Доля категории", "pct", "Доля площади квартала под преобладающей категорией"),
    ("site_area", "Площадь, м²", "int", "Площадь квартала"),
    ("population", "Население, чел.", "int", "Оценка по жилой площади: 30 м² на человека"),
    ("capacity", "Ёмкость школ, мест", "int", "Суммарная ёмкость школ в квартале"),
    ("demand", "Спрос на школы, мест", "int", "Норматив 120 мест на 1000 жителей"),
    ("provision_weak", "Обеспеченность weak", "dec2", "Доля спроса с учётом конкуренции между кварталами, 0..1"),
    ("provision_strong", "Обеспеченность strong", "dec2", "Доля спроса в пределах радиуса 15 минут, 0..1"),
    ("demand_within", "Спрос в радиусе", "int", "Спрос, покрываемый в пределах 15 минут"),
    ("demand_without", "Спрос вне радиуса", "int", "Спрос, не покрываемый в пределах 15 минут"),
    (
        "area_accessibility",
        "Доступность территории, мин",
        "dec2",
        "Среднее время до других кварталов, взвешенное по площади",
    ),
    ("fsi", "FSI", "dec2", "Плотность застройки: м² этажей на м² земли"),
    ("gsi", "GSI", "dec2", "Доля квартала, занятая зданиями"),
    ("mxi", "MXI", "dec2", "Доля жилой площади в общей площади застройки"),
    ("l", "L, этажей", "dec1", "Средняя этажность"),
    ("osr", "OSR", "dec2", "Открытое пространство на м² застройки"),
    ("share_living", "Жилая площадь / пятно", "dec2", "Жилая площадь на м² пятна застройки"),
    ("share_non_living", "Нежилая площадь / пятно", "dec2", "Нежилая площадь на м² пятна застройки"),
    ("footprint_area", "Пятно застройки, м²", "int", "Суммарная площадь зданий в плане"),
    ("build_floor_area", "Площадь этажей, м²", "int", "Суммарная площадь этажей"),
    ("living_area", "Жилая площадь, м²", "int", "Оценка жилой площади"),
    ("non_living_area", "Нежилая площадь, м²", "int", "Оценка нежилой площади"),
    ("morphotype_spacematrix", "Морфотип Spacematrix", "text", "Кластер Spacematrix"),
    ("spacematrix_cluster", "Кластер Spacematrix", "int", "Номер кластера Spacematrix"),
    ("morphotype_strelka", "Морфотип Strelka", "text", "Тип по правилам Strelka"),
    ("residential", "Доля жилой", "dec2", "Доля площади квартала под жилой функцией"),
    ("business", "Доля деловой", "dec2", "Доля площади квартала под деловой функцией"),
    ("recreation", "Доля рекреации", "dec2", "Доля площади квартала под рекреацией"),
    ("industrial", "Доля промышленной", "dec2", "Доля площади квартала под промышленностью"),
    ("transport", "Доля транспорта", "dec2", "Доля площади квартала под транспортом"),
    ("special", "Доля специальной", "dec2", "Доля площади квартала под специальной функцией"),
    ("agriculture", "Доля сельхоза", "dec2", "Доля площади квартала под сельским хозяйством"),
]


# ---------------------------------------------------------------------------
# Пути и чтение данных
# ---------------------------------------------------------------------------


def find_data_dir(explicit: str | None) -> Path:
    """Найти каталог с готовыми результатами расчётов."""
    marker = Path("results") / "blocks_metrics.geojson"

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / marker).exists():
            raise SystemExit(f"В каталоге нет {marker}: {path}")
        return path

    for candidate in DEFAULT_DATA_CANDIDATES:
        if (candidate / marker).exists():
            return candidate

    for found in sorted(ROOT.rglob(str(marker))):
        if ".venv" not in found.parts:
            return found.parent.parent

    raise SystemExit(
        "Не найдены готовые результаты (results/blocks_metrics.geojson).\n"
        "Укажите каталог явно: python dashboard.py --data-dir путь/к/data/vasileostrovsky"
    )


def read_json(path: Path) -> dict[str, Any] | None:
    """Прочитать JSON, вернуть None, если файла нет."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> pd.DataFrame | None:
    """Прочитать CSV, вернуть None, если файла нет."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def round_coords(value: Any, digits: int = 6) -> Any:
    """Округлить координаты GeoJSON, чтобы уменьшить размер HTML."""
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)):
            return [round(float(item), digits) for item in value]
        return [round_coords(item, digits) for item in value]
    return value


def load_geojson(path: Path) -> dict[str, Any] | None:
    """Прочитать GeoJSON и округлить координаты."""
    data = read_json(path)
    if not data:
        return None
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        if "coordinates" in geometry:
            geometry["coordinates"] = round_coords(geometry["coordinates"])
    return data


def geojson_frame(geojson: dict[str, Any]) -> pd.DataFrame:
    """Собрать таблицу свойств из GeoJSON."""
    frame = pd.DataFrame([feature.get("properties") or {} for feature in geojson["features"]])
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["block_id"] = frame["block_id"].astype(int)
    return frame.sort_values("block_id").reset_index(drop=True)


def iter_coords(geometry: dict[str, Any]) -> list[list[float]]:
    """Плоский список координат геометрии."""
    coords = geometry.get("coordinates") or []
    stack: list[Any] = [coords]
    points: list[list[float]] = []
    while stack:
        item = stack.pop()
        if isinstance(item, list) and item and isinstance(item[0], (int, float)):
            points.append(item)
        elif isinstance(item, list):
            stack.extend(item)
    return points


def schools_geojson(path: Path) -> dict[str, Any] | None:
    """Точки-центроиды школ для слоя на карте."""
    geojson = load_geojson(path)
    if not geojson:
        return None
    features = []
    for feature in geojson["features"]:
        points = iter_coords(feature.get("geometry") or {})
        if not points:
            continue
        lon = sum(point[0] for point in points) / len(points)
        lat = sum(point[1] for point in points) / len(points)
        capacity = (feature.get("properties") or {}).get("capacity")
        features.append(
            {
                "type": "Feature",
                "properties": {"capacity": capacity},
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def download_text(url: str) -> str:
    """Скачать текст: requests (с сертификатами), иначе urllib, иначе curl."""
    try:
        import requests

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:
        pass

    try:
        import ssl

        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=20, context=context) as response:
            return response.read().decode("utf-8")
    except Exception:
        pass

    import subprocess

    return subprocess.check_output(["curl", "-fsSL", url], timeout=30).decode("utf-8")


def load_leaflet() -> tuple[str, str, bool]:
    """Вернуть CSS и JS Leaflet: из кэша, из сети или пустые строки."""
    cache = CACHE_DIR / f"leaflet-{LEAFLET_VERSION}"
    css_path = cache / "leaflet.css"
    js_path = cache / "leaflet.js"
    if css_path.exists() and js_path.exists():
        return css_path.read_text(encoding="utf-8"), js_path.read_text(encoding="utf-8"), True

    try:
        css = download_text(LEAFLET_CSS_URL)
        js = download_text(LEAFLET_JS_URL)
    except Exception:
        return "", "", False

    cache.mkdir(parents=True, exist_ok=True)
    css_path.write_text(css, encoding="utf-8")
    js_path.write_text(js, encoding="utf-8")
    return css, js, True


# ---------------------------------------------------------------------------
# Форматирование и таблицы
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 2, dash: str = "—") -> str:
    """Число в русском виде: пробелы в тысячах, запятая в дробях."""
    if value is None:
        return dash
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(number):
        return dash
    if digits == 0:
        return f"{number:,.0f}".replace(",", "\u202f")
    return f"{number:,.{digits}f}".replace(",", "\u202f").replace(".", ",")


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Доля 0..1 в проценты."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{fmt(float(value) * 100, digits)}\u00a0%"


def esc(value: Any) -> str:
    """Экранировать текст для HTML."""
    return html_lib.escape("" if value is None else str(value))


def cell_text(column: str, kind: str, value: Any) -> tuple[str, str | None]:
    """Вернуть (текст, значение для сортировки)."""
    if kind == "text":
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "—", None
        return str(value), None
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—", None
    number = float(value)
    if not math.isfinite(number):
        return "—", None
    if kind == "int":
        return fmt(number, 0), f"{number:.6f}"
    if kind == "pct":
        return fmt_pct(number), f"{number:.6f}"
    if kind == "dec1":
        return fmt(number, 1), f"{number:.6f}"
    if kind == "dec3":
        return fmt(number, 3), f"{number:.6f}"
    return fmt(number, 2), f"{number:.6f}"


def simple_table(frame: pd.DataFrame, table_id: str, numeric: set[str] | None = None) -> str:
    """Простая таблица из DataFrame с человеческим форматом чисел."""
    del numeric
    head = "".join(f"<th>{esc(column)}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                number = float(value)
                if not math.isfinite(number):
                    cells.append("<td>—</td>")
                    continue
                text = fmt(number, 0) if number.is_integer() else fmt(number, 2)
                cells.append(f'<td data-v="{number:.6f}">{text}</td>')
            else:
                cells.append(f"<td>{esc(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table class="data-table sortable" id="{table_id}">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def checks_table(checks: pd.DataFrame) -> str:
    """Таблица проверок с цветными значками."""
    head = "".join(f"<th>{esc(column)}</th>" for column in checks.columns)
    rows = []
    for _, row in checks.iterrows():
        cells = []
        for column in checks.columns:
            value = row[column]
            if column == "Результат":
                ok = str(value).strip().upper() == "OK"
                cls = "ok" if ok else "problem"
                cells.append(f'<td><span class="badge {cls}">{esc(value)}</span></td>')
            else:
                cells.append(f"<td>{esc(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table class="data-table" id="checks-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def big_table(frame: pd.DataFrame) -> str:
    """Полная таблица показателей с поиском, фильтрами и сортировкой."""
    head = "".join(f'<th title="{esc(title)}">{esc(header)}</th>' for _, header, _, title in COLUMNS)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        search_parts = []
        for column, _, kind, _ in COLUMNS:
            text, sort = cell_text(column, kind, row.get(column))
            attribute = f' data-v="{sort}"' if sort is not None else ""
            cells.append(f"<td{attribute}>{esc(text)}</td>")
            search_parts.append(str(text))
        flags = []
        if float(row.get("population") or 0) > 0:
            flags.append("pop")
        if float(row.get("capacity") or 0) > 0:
            flags.append("school")
        if float(row.get("fsi") or 0) > 0:
            flags.append("built")
        search = esc(" ".join(search_parts).lower())
        rows.append(f'<tr data-flags="{" ".join(flags)}" data-text="{search}">' + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap big"><table class="data-table sortable" id="metrics-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Данные карты
# ---------------------------------------------------------------------------


def number_or_none(value: Any) -> float | None:
    """NaN и бесконечность превратить в None (валидный JSON)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_map_payload(
    frame: pd.DataFrame,
    geojson: dict[str, Any],
    schools: dict[str, Any] | None,
    boundary: dict[str, Any] | None,
) -> dict[str, Any]:
    """GeoJSON кварталов с готовыми к показу свойствами."""
    by_id = {int(row.block_id): row for row in frame.itertuples()}
    features = []
    for feature in geojson["features"]:
        block_id = (feature.get("properties") or {}).get("block_id")
        if block_id is None:
            continue
        row = by_id.get(int(block_id))
        if row is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "block_id": int(row.block_id),
                    "population": number_or_none(row.population),
                    "fsi": number_or_none(row.fsi),
                    "gsi": number_or_none(row.gsi),
                    "mxi": number_or_none(row.mxi),
                    "l": number_or_none(row.l),
                    "osr": number_or_none(row.osr),
                    "capacity": number_or_none(row.capacity),
                    "provision_weak": number_or_none(row.provision_weak),
                    "site_area": number_or_none(row.site_area),
                    "area_km2": number_or_none(row.site_area / 1e6),
                    "land_use_ru": "не назначено" if pd.isna(row.land_use_ru) else str(row.land_use_ru),
                    "morphotype_ru": "—" if pd.isna(row.morphotype_ru) else str(row.morphotype_ru),
                },
                "geometry": feature["geometry"],
            }
        )
    return {
        "blocks": {"type": "FeatureCollection", "features": features},
        "schools": schools,
        "boundary": boundary,
    }


def build_map_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Описание показателей для раскраски карты."""
    specs = [
        ("population", "Население, чел.", "population", 0, 95, "чел."),
        ("fsi", "Плотность FSI", "fsi", 2, 100, "FSI"),
        ("gsi", "Застроенность GSI", "gsi", 2, 100, "GSI"),
        ("mxi", "Доля жилья MXI", "mxi", 2, 100, "MXI"),
        ("l", "Этажность L", "l", 1, 100, "этажи"),
        ("osr", "Открытое пространство OSR", "osr", 2, 95, "OSR"),
        ("capacity", "Ёмкость школ, мест", "capacity", 0, 95, "мест"),
        ("provision_weak", "Обеспеченность школами", "provision_weak", 2, 100, "0..1"),
        ("site_area", "Площадь квартала", "site_area", 0, 95, "м²"),
    ]

    metrics: dict[str, Any] = {}
    for key, label, column, digits, percentile, unit in specs:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if key == "provision_weak":
            low, high = 0.0, 1.0
        elif key == "mxi":
            low, high = 0.0, 1.0
        else:
            low = 0.0
            high = float(np.percentile(finite, percentile)) if len(finite) else 1.0
            high = max(high, 1e-9)

        clipped = percentile < 100 and len(finite) and float(np.percentile(finite, percentile)) < float(finite.max())
        metrics[key] = {
            "key": column,
            "label": label,
            "min": low,
            "max": high,
            "digits": digits,
            "unit": unit,
            "stops": COLOR_STOPS[key],
            "note": (
                f"Шкала цвета ограничена {percentile}-м перцентилем, чтобы выбросы не съедали цвет." if clipped else ""
            ),
        }

    metrics["land_use"] = {
        "key": "land_use_ru",
        "label": "Землепользование",
        "categorical": True,
        "colors": LAND_USE_COLORS,
        "note": "Серым показаны кварталы без назначенной категории.",
    }
    return metrics


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def style_figure(figure: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Единый стиль графиков."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=16, t=54, b=10),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color="#1a202c")),
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color="#2d3748"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(font=dict(size=12)),
    )
    figure.update_xaxes(gridcolor="#edf2f7", zeroline=False)
    figure.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return figure


def fig_html(figure: go.Figure, div_id: str) -> str:
    """HTML одного графика без повторного подключения plotly.js."""
    return pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
        div_id=div_id,
    )


def chart_population(frame: pd.DataFrame) -> go.Figure:
    """Топ-15 кварталов по населению."""
    top = frame[frame["population"] > 0].nlargest(15, "population").sort_values("population")
    figure = go.Figure(
        go.Bar(
            x=top["population"],
            y=[f"№{int(value)}" for value in top["block_id"]],
            orientation="h",
            marker_color="#dd6b55",
            text=[fmt(value, 0) for value in top["population"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Квартал %{y}: %{x:,.0f} чел.<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="человек")
    return style_figure(figure, "Топ-15 кварталов по населению")


def chart_fsi(frame: pd.DataFrame) -> go.Figure:
    """Распределение плотности застройки."""
    values = frame.loc[frame["fsi"] > 0, "fsi"]
    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=24,
            marker_color="#3d5a80",
            marker_line_width=1,
            marker_line_color="#ffffff",
            hovertemplate="FSI %{x}: %{y} кварталов<extra></extra>",
        )
    )
    if len(values):
        median = float(values.median())
        figure.add_vline(
            x=median,
            line_dash="dash",
            line_color="#c53030",
            annotation_text=f"медиана {fmt(median, 2)}",
            annotation_position="top",
        )
    figure.update_layout(xaxis_title="FSI, м² этажей на м² земли", yaxis_title="кварталов")
    return style_figure(figure, "Плотность застройки (застроенные кварталы)")


def chart_land_use(frame: pd.DataFrame) -> go.Figure:
    """Площадь и население по категориям землепользования."""
    grouped = (
        frame.dropna(subset=["land_use_ru"])
        .groupby("land_use_ru", as_index=False)
        .agg(area=("site_area", "sum"), population=("population", "sum"), blocks=("block_id", "count"))
        .sort_values("area")
    )
    figure = go.Figure(
        go.Bar(
            x=grouped["area"] / 1e6,
            y=grouped["land_use_ru"],
            orientation="h",
            marker_color="#6b9080",
            text=[f"{fmt(value, 2)} км²" for value in grouped["area"] / 1e6],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([grouped["population"], grouped["blocks"]], axis=1),
            hovertemplate=(
                "%{y}: %{x:.2f} км²<br>Население: %{customdata[0]:,.0f} чел."
                "<br>Кварталов: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(xaxis_title="км²")
    return style_figure(figure, "Землепользование: площадь по категориям")


def chart_morphotypes(frame: pd.DataFrame) -> go.Figure:
    """Морфотипы Spacematrix и Strelka."""
    space = frame["morphotype_ru_spacematrix"].dropna().value_counts()
    strelka = frame["morphotype_ru_strelka"].dropna().value_counts()
    labels = list(dict.fromkeys(list(space.index) + list(strelka.index)))
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=labels,
            x=[int(space.get(label, 0)) for label in labels],
            orientation="h",
            name="Spacematrix",
            marker_color="#4c78a8",
            hovertemplate="%{y}: %{x} кварталов<extra>Spacematrix</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=[int(strelka.get(label, 0)) for label in labels],
            orientation="h",
            name="Strelka",
            marker_color="#f58518",
            hovertemplate="%{y}: %{x} кварталов<extra>Strelka</extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        xaxis_title="кварталов",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_figure(figure, "Морфотипы кварталов")


def chart_travel(matrix: dict[str, Any] | None) -> go.Figure:
    """Распределение времени между кварталами (по готовой матрице)."""
    figure = go.Figure()
    if matrix and len(matrix["hist_x"]):
        figure.add_trace(
            go.Bar(
                x=matrix["hist_x"],
                y=matrix["hist_y"],
                marker_color="#7c6f9f",
                hovertemplate="%{x:.0f} мин: %{y} пар<extra></extra>",
            )
        )
        figure.add_vline(
            x=15,
            line_dash="dash",
            line_color="#2f855a",
            annotation_text="радиус 15 мин",
            annotation_position="top right",
        )
        figure.update_layout(xaxis_title="время в пути, минуты (показаны значения ≤ 60)", yaxis_title="пар кварталов")
        title = "Время между кварталами"
    else:
        figure.add_annotation(
            text="Матрица доступности не найдена",
            showarrow=False,
            font=dict(size=14, color="#718096"),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        title = "Время между кварталами"
    return style_figure(figure, title)


def chart_provision(frame: pd.DataFrame) -> go.Figure:
    """Спрос и ёмкость школ по кварталам."""
    subset = frame[(frame["demand"] > 0) | (frame["capacity"] > 0)]
    figure = go.Figure()
    if len(subset):
        figure.add_trace(
            go.Scatter(
                x=subset["demand"],
                y=subset["capacity"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=subset["population"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=dict(text="население", side="top"), thickness=10, len=0.7),
                    line=dict(width=0.5, color="#ffffff"),
                    opacity=0.9,
                ),
                text=[f"№{int(value)}" for value in subset["block_id"]],
                customdata=np.stack(
                    [
                        [fmt(value, 0) for value in subset["population"]],
                        [fmt(value, 2) for value in subset["provision_weak"]],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "Квартал %{text}<br>Спрос: %{x:.0f} мест<br>Ёмкость: %{y:.0f} мест"
                    "<br>Население: %{customdata[0]} чел.<br>Обеспеченность: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
        limit = float(max(subset["demand"].max(), subset["capacity"].max(), 1))
        figure.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line=dict(color="#a0aec0", dash="dot", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(xaxis_title="спрос, мест", yaxis_title="ёмкость, мест")
    return style_figure(figure, "Школы: спрос и ёмкость по кварталам")


def chart_capacity(frame: pd.DataFrame) -> go.Figure:
    """Топ кварталов по ёмкости школ."""
    top = frame[frame["capacity"] > 0].nlargest(12, "capacity").sort_values("capacity")
    figure = go.Figure(
        go.Bar(
            x=top["capacity"],
            y=[f"№{int(value)}" for value in top["block_id"]],
            orientation="h",
            marker_color="#6b46c1",
            text=[fmt(value, 0) for value in top["capacity"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Квартал %{y}: %{x:,.0f} мест<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="мест")
    return style_figure(figure, "Топ-12 кварталов по ёмкости школ")


# ---------------------------------------------------------------------------
# Матрица доступности
# ---------------------------------------------------------------------------


def matrix_stats(path: Path) -> dict[str, Any] | None:
    """Статистика готовой матрицы времени (без пересчёта)."""
    frame = read_csv(path)
    if frame is None:
        return None
    matrix = frame.set_index(frame.columns[0]).to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    diagonal = np.diag(matrix)
    off = matrix.copy()
    np.fill_diagonal(off, np.nan)
    finite_mask = np.isfinite(off)
    values = off[finite_mask]
    values = values[values >= 0]
    clipped = values[values <= 60]
    counts, edges = np.histogram(clipped, bins=np.arange(0, 61, 1))
    return {
        "shape": matrix.shape,
        "pairs": int(off.size),
        "unreachable": int(np.isinf(off).sum()),
        "unreachable_share": float(np.isinf(off).sum() / off.size),
        "finite": int(values.size),
        "zero_share": float((values == 0).mean()) if values.size else 0.0,
        "gt1000_share": float((values > 1000).mean()) if values.size else 0.0,
        "le15_share": float((values <= 15).mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else float("nan"),
        "diag_nonzero": int((diagonal != 0).sum()),
        "hist_x": (edges[:-1] + 0.5).tolist(),
        "hist_y": counts.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Выводы
# ---------------------------------------------------------------------------


def conclusion_items(
    frame: pd.DataFrame,
    checks: pd.DataFrame | None,
    matrix: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    schools_capacity_unique: int | None,
) -> list[tuple[str, str, str]]:
    """Автоматические выводы по готовым данным."""
    items: list[tuple[str, str, str]] = []

    count = len(frame)
    area = float(frame["site_area"].sum()) / 1e6
    population = float(frame["population"].sum())
    density = population / area if area else 0.0
    inhabited = int((frame["population"] > 0).sum())
    built = int((frame["fsi"] > 0).sum())
    assigned = int(frame["land_use"].notna().sum())
    capacity = float(frame["capacity"].sum())
    demand = float(frame["demand"].sum())
    provision_blocks = int(frame["provision_weak"].notna().sum())
    schools = int((manifest or {}).get("counts", {}).get("schools", 0))

    items.append(
        (
            "info",
            "Масштаб территории",
            f"{count} кварталов общей площадью {fmt(area, 2)} км². Оценка населения — "
            f"{fmt(population, 0)} человек, то есть около {fmt(density, 0)} чел./км².",
        )
    )

    items.append(
        (
            "warn",
            "Заселено меньше пятой части кварталов",
            f"Население больше нуля только в {inhabited} из {count} кварталов "
            f"({fmt_pct(inhabited / count)}). В среднем на заселённый квартал приходится "
            f"{fmt(population / inhabited if inhabited else 0, 0)} человек. Пустые кварталы — "
            "это парки, вода, промзоны и кварталы, куда не попали здания из OSM.",
        )
    )

    floor_single = frame["l"].dropna().nunique() <= 1
    area_equal = bool(np.allclose(frame["build_floor_area"], frame["footprint_area"], equal_nan=True))
    if floor_single or area_equal:
        items.append(
            (
                "bad",
                "Этажность не восстановлена — плотность занижена",
                f"Показатель L принимает только значения {', '.join(fmt(value, 0) for value in sorted(frame['l'].dropna().unique()))}, "
                "а площадь этажей совпадает с площадью пятна застройки. Значит, у зданий не подставилась "
                "этажность, и FSI = GSI, население и морфотипы посчитаны по одноэтажной застройке. "
                "Числа ниже — нижняя оценка.",
            )
        )

    top_land = (
        frame.dropna(subset=["land_use"]).groupby("land_use_ru")["site_area"].sum().sort_values(ascending=False).head(3)
    )
    top_text = ", ".join(f"{name} — {fmt(value / 1e6, 2)} км²" for name, value in top_land.items())
    items.append(
        (
            "info",
            "Землепользование назначено не всем кварталам",
            f"Категория есть у {assigned} из {count} кварталов ({fmt_pct(assigned / count)}). "
            f"Крупнейшие категории: {top_text}. У остальных преобладающая функция не определена.",
        )
    )

    items.append(
        (
            "info",
            "Морфотипы: только малоэтажные типы",
            f"Классифицировано {built} кварталов. Все они попали в малоэтажные низкоплотные типы — "
            "это следствие невосстановленной этажности, а не вывод о застройке Васильевского острова.",
        )
    )

    if schools and capacity:
        ratio = capacity / demand if demand else float("nan")
        imputed = (
            " Все школы получили одинаковую ёмкость 250 мест — реальных данных о вместимости в OSM нет."
            if schools_capacity_unique == 1
            else ""
        )
        items.append(
            (
                "warn",
                "Школы: ёмкость выше норматива, но это оценка",
                f"{schools} школ, суммарная ёмкость {fmt(capacity, 0)} мест при нормативном спросе "
                f"{fmt(demand, 0)} мест (120 на 1000 жителей), то есть в {fmt(ratio, 1)} раза больше норматива."
                f"{imputed} Обеспеченность рассчитана только для {provision_blocks} кварталов с населением; "
                "во всех них она равна 1.0.",
            )
        )

    if matrix:
        items.append(
            (
                "bad",
                "Матрица доступности непригодна для выводов",
                f"В матрице {matrix['shape'][0]}×{matrix['shape'][1]} {fmt_pct(matrix['unreachable_share'])} пар "
                f"недостижимы, {fmt_pct(matrix['zero_share'])} значений равны нулю, "
                f"{fmt_pct(matrix['gt1000_share'])} значений больше 1000 (это не минуты), а диагональ ненулевая "
                f"в {matrix['diag_nonzero']} случаях. Единицы измерения смешаны: часть значений в минутах, "
                "часть — в метрах или секундах. Матрицу нужно пересчитать.",
            )
        )

    if frame["area_accessibility"].notna().sum() == 0:
        items.append(
            (
                "bad",
                "Доступность территории не рассчитана",
                "Столбец area_accessibility пуст во всех кварталах: шаг расчёта доступности не дал результата. "
                "На карте и в таблице он показан как «—».",
            )
        )

    if checks is not None and "Результат" in checks.columns:
        problems = checks[checks["Результат"].astype(str).str.upper() != "OK"]
        if len(problems):
            names = "; ".join(problems["Проверка"].astype(str).tolist())
            items.append(
                (
                    "bad",
                    f"Проверки: {len(problems)} из {len(checks)} с замечаниями",
                    f"Проблемные проверки: {names}. Подробности — в таблице раздела «Проверки».",
                )
            )
        else:
            items.append(("good", "Все проверки пройдены", f"{len(checks)} проверок без замечаний."))

    items.append(
        (
            "info",
            "Что делать дальше",
            "1) Исправить дозаполнение этажности зданий и пересчитать население, FSI и морфотипы. "
            "2) Пересчитать матрицу доступности с весом time_min и проверить единицы. "
            "3) После этого заново запустить этот файл — он прочитает новые GeoJSON и пересоберёт дашборд "
            "без изменения кода.",
        )
    )
    return items


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


CSS = """
:root {
  --bg: #f5f6f8;
  --card: #ffffff;
  --ink: #1a202c;
  --muted: #64748b;
  --line: #e2e8f0;
  --accent: #2b6cb0;
  --good: #2f855a;
  --warn: #b7791f;
  --bad: #c53030;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 15px;
  line-height: 1.55;
}
a { color: var(--accent); }
header {
  background: var(--card);
  border-bottom: 1px solid var(--line);
  padding: 28px 32px 22px;
}
header h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: -0.02em; }
header p.sub { margin: 0 0 14px; color: var(--muted); max-width: 900px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  background: #edf2f7;
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 12.5px;
  color: #2d3748;
  white-space: nowrap;
}
nav {
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
  padding: 8px 32px;
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  font-size: 14px;
}
nav a { color: var(--muted); text-decoration: none; padding: 4px 0; border-bottom: 2px solid transparent; }
nav a:hover { color: var(--accent); border-bottom-color: var(--accent); }
main { max-width: 1400px; margin: 0 auto; padding: 26px 32px 60px; }
section { margin-bottom: 38px; scroll-margin-top: 64px; }
h2 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
p.section-note { margin: 0 0 16px; color: var(--muted); font-size: 14px; }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
.kpi .label { color: var(--muted); font-size: 13px; }
.kpi .value { font-size: 26px; font-weight: 650; letter-spacing: -0.02em; margin-top: 2px; }
.kpi .sub { color: var(--muted); font-size: 12.5px; }
.map-card { padding: 12px; }
.map-controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 4px 8px 12px; }
.map-controls label { font-size: 13px; color: var(--muted); }
select, input[type="search"] {
  font: inherit;
  font-size: 14px;
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
}
select:focus, input:focus { outline: 2px solid #bee3f8; border-color: var(--accent); }
.leaflet-map { height: 620px; width: 100%; border-radius: 10px; background: #eef1f4; }
.map-legend {
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.45;
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.12);
  min-width: 150px;
  max-width: 240px;
}
.legend-title { font-weight: 650; margin-bottom: 6px; }
.legend-bar { height: 10px; border-radius: 4px; margin: 6px 0 4px; }
.legend-scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 11.5px; }
.legend-note { color: var(--muted); font-size: 11px; margin-top: 6px; }
.legend-row { display: flex; align-items: center; gap: 7px; margin-top: 3px; }
.legend-row .swatch { width: 13px; height: 13px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.12); }
.leaflet-tooltip {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(16, 24, 40, 0.16);
  font-size: 12.5px;
  line-height: 1.45;
  padding: 8px 10px;
  color: var(--ink);
}
.leaflet-tooltip-top:before { border-top-color: #ffffff; }
.caption { color: var(--muted); font-size: 13px; padding: 6px 8px 2px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }
.chart { padding: 6px 8px 4px; }
.table-card { padding: 18px 20px; margin-bottom: 16px; }
.table-card h3 { font-size: 16px; margin: 0 0 10px; }
.table-wrap { overflow: auto; max-height: 460px; border: 1px solid var(--line); border-radius: 10px; }
.table-wrap.big { max-height: 72vh; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data-table th, table.data-table td {
  padding: 7px 10px;
  border-bottom: 1px solid #edf2f7;
  text-align: left;
  white-space: nowrap;
}
table.data-table th {
  position: sticky;
  top: 0;
  background: #f7fafc;
  z-index: 2;
  font-weight: 600;
  color: #2d3748;
  border-bottom: 1px solid var(--line);
}
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th:hover { background: #edf2f7; }
table.data-table td:first-child, table.data-table th:first-child {
  position: sticky;
  left: 0;
  background: #fff;
  z-index: 1;
}
table.data-table th:first-child { z-index: 3; background: #f7fafc; }
table.data-table tbody tr:hover td { background: #f7fafc; }
table.data-table td { font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge.ok { background: #c6f6d5; color: #22543d; }
.badge.problem { background: #fed7d7; color: #822727; }
.filters { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
.filters .count { color: var(--muted); font-size: 13px; }
button {
  font: inherit;
  font-size: 13.5px;
  padding: 7px 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}
button:hover { background: #f7fafc; border-color: #cbd5e0; }
.conclusions { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.conclusions li {
  background: var(--card);
  border: 1px solid var(--line);
  border-left: 4px solid var(--muted);
  border-radius: 12px;
  padding: 14px 18px;
}
.conclusions li.good { border-left-color: var(--good); }
.conclusions li.warn { border-left-color: var(--warn); }
.conclusions li.bad { border-left-color: var(--bad); }
.conclusions li.info { border-left-color: var(--accent); }
.conclusions .title { font-weight: 650; margin-bottom: 2px; }
.conclusions .text { color: #2d3748; font-size: 14px; }
.limits { margin: 0; padding-left: 20px; color: #2d3748; }
.limits li { margin-bottom: 6px; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--accent); font-size: 14px; }
.kv { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.kv td { padding: 6px 10px; border-bottom: 1px solid #edf2f7; vertical-align: top; }
.kv td:first-child { color: var(--muted); width: 38%; }
footer { color: var(--muted); font-size: 12.5px; text-align: center; padding: 24px; }
.alert {
  background: #fffaf0;
  border: 1px solid #fbd38d;
  color: #744210;
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 14px;
  margin-bottom: 18px;
}
@media (max-width: 720px) {
  header, nav, main { padding-left: 16px; padding-right: 16px; }
  .charts { grid-template-columns: 1fr; }
  .leaflet-map { height: 460px; }
}
@media print {
  nav, .filters, button { display: none; }
  body { background: #fff; }
  .card, .conclusions li { box-shadow: none; }
  .table-wrap { max-height: none; overflow: visible; }
}
"""

MAP_JS = r"""
(function () {
  if (!window.L || !window.MAP_DATA) { return; }

  const data = window.MAP_DATA;
  const metrics = window.MAP_METRICS;
  const state = { metric: 'population' };

  const map = L.map('blocks-map', { zoomControl: true, preferCanvas: false });
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '© OpenStreetMap'
  });
  const light = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 20, attribution: '© OpenStreetMap, © CARTO'
  });
  const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 20, attribution: '© OpenStreetMap, © CARTO'
  });
  const empty = L.layerGroup();
  const basemaps = { osm: osm, light: light, dark: dark, empty: empty };
  osm.addTo(map);

  function hexToRgb(hex) {
    const value = hex.replace('#', '');
    return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
  }

  function mixColor(from, to, t) {
    const a = hexToRgb(from);
    const b = hexToRgb(to);
    const channel = (index) => Math.round(a[index] + (b[index] - a[index]) * t);
    return 'rgb(' + channel(0) + ',' + channel(1) + ',' + channel(2) + ')';
  }

  function scaleColor(stops, t) {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (stops.length - 1);
    const index = Math.min(Math.floor(scaled), stops.length - 2);
    return mixColor(stops[index], stops[index + 1], scaled - index);
  }

  const numberFormat = (value, digits) => {
    if (value === null || value === undefined || Number.isNaN(value)) { return '—'; }
    return new Intl.NumberFormat('ru-RU', {
      minimumFractionDigits: 0,
      maximumFractionDigits: digits
    }).format(value);
  };

  function styleFor(feature) {
    const metric = metrics[state.metric];
    const properties = feature.properties;
    const base = { color: '#ffffff', weight: 0.5, opacity: 0.9, fillOpacity: 0.75 };
    if (metric.categorical) {
      const color = metric.colors[properties[metric.key]] || '#cbd5e0';
      return Object.assign(base, { fillColor: color });
    }
    const value = properties[metric.key];
    if (value === null || value === undefined || Number.isNaN(value)) {
      return Object.assign(base, { fillColor: '#cbd5e0', fillOpacity: 0.35 });
    }
    const span = metric.max - metric.min || 1;
    const t = (value - metric.min) / span;
    return Object.assign(base, { fillColor: scaleColor(metric.stops, t) });
  }

  function tooltip(properties) {
    return '<b>Квартал ' + properties.block_id + '</b><br>' +
      'Население: ' + numberFormat(properties.population, 0) + ' чел.<br>' +
      'Площадь: ' + numberFormat(properties.area_km2, 3) + ' км²<br>' +
      'FSI: ' + numberFormat(properties.fsi, 2) + '<br>' +
      'Этажность L: ' + numberFormat(properties.l, 1) + '<br>' +
      'Ёмкость школ: ' + numberFormat(properties.capacity, 0) + ' мест<br>' +
      'Обеспеченность: ' + numberFormat(properties.provision_weak, 2) + '<br>' +
      'Землепользование: ' + properties.land_use_ru + '<br>' +
      'Морфотип: ' + properties.morphotype_ru;
  }

  const boundaryLayer = data.boundary
    ? L.geoJSON(data.boundary, {
        style: { color: '#1a202c', weight: 2.5, fill: false, interactive: false }
      })
    : null;

  const blocksLayer = L.geoJSON(data.blocks, {
    style: styleFor,
    onEachFeature: function (feature, layer) {
      layer.bindTooltip(tooltip(feature.properties), { sticky: true, opacity: 1 });
      layer.on('mouseover', function () { this.setStyle({ weight: 2, color: '#1a202c' }); });
      layer.on('mouseout', function () { blocksLayer.resetStyle(this); });
    }
  });

  if (boundaryLayer) { boundaryLayer.addTo(map); }
  blocksLayer.addTo(map);

  let schoolsLayer = null;
  if (data.schools && data.schools.features.length) {
    schoolsLayer = L.geoJSON(data.schools, {
      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, {
          radius: 5, weight: 1.5, color: '#ffffff', fillColor: '#c53030', fillOpacity: 0.95
        });
      },
      onEachFeature: function (feature, layer) {
        layer.bindTooltip('<b>Школа</b><br>Ёмкость: ' +
          numberFormat(feature.properties.capacity, 0) + ' мест', { sticky: true });
      }
    }).addTo(map);
  }

  map.fitBounds(blocksLayer.getBounds(), { padding: [18, 18] });

  const overlays = {};
  if (schoolsLayer) { overlays['Школы'] = schoolsLayer; }
  if (boundaryLayer) { overlays['Граница района'] = boundaryLayer; }
  if (Object.keys(overlays).length) {
    L.control.layers(null, overlays, { position: 'topright', collapsed: false }).addTo(map);
  }

  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = function () { return L.DomUtil.create('div', 'map-legend'); };
  legend.addTo(map);

  function renderLegend() {
    const metric = metrics[state.metric];
    const element = document.querySelector('.map-legend');
    if (!element) { return; }
    if (metric.categorical) {
      let html = '<div class="legend-title">' + metric.label + '</div>';
      Object.keys(metric.colors).forEach(function (label) {
        html += '<div class="legend-row"><span class="swatch" style="background:' +
          metric.colors[label] + '"></span>' + label + '</div>';
      });
      element.innerHTML = html;
      return;
    }
    const gradient = metric.stops.map(function (color, index) {
      return color + ' ' + (index * 100 / (metric.stops.length - 1)) + '%';
    }).join(',');
    element.innerHTML =
      '<div class="legend-title">' + metric.label + '</div>' +
      '<div class="legend-bar" style="background:linear-gradient(90deg,' + gradient + ')"></div>' +
      '<div class="legend-scale"><span>' + numberFormat(metric.min, metric.digits) +
      '</span><span>' + numberFormat(metric.max, metric.digits) + '</span></div>' +
      (metric.note ? '<div class="legend-note">' + metric.note + '</div>' : '');
  }

  function applyMetric(key) {
    state.metric = key;
    blocksLayer.setStyle(styleFor);
    renderLegend();
    const note = document.getElementById('map-note');
    if (note) { note.textContent = metrics[key].note || ''; }
  }

  document.getElementById('metric-select').addEventListener('change', function (event) {
    applyMetric(event.target.value);
  });

  document.getElementById('style-select').addEventListener('change', function (event) {
    Object.keys(basemaps).forEach(function (key) { map.removeLayer(basemaps[key]); });
    basemaps[event.target.value].addTo(map);
  });

  renderLegend();

  const search = document.getElementById('metrics-search');
  const filter = document.getElementById('metrics-filter');
  const rowsShown = document.getElementById('rows-shown');
  const metricsTable = document.getElementById('metrics-table');

  function applyFilter() {
    if (!metricsTable) { return; }
    const query = (search.value || '').toLowerCase().trim();
    const mode = filter.value;
    let shown = 0;
    Array.prototype.forEach.call(metricsTable.tBodies[0].rows, function (row) {
      const matchesText = !query || (row.dataset.text || '').indexOf(query) !== -1;
      const flags = (row.dataset.flags || '').split(' ');
      const matchesMode = mode === 'all' || flags.indexOf(mode) !== -1;
      const visible = matchesText && matchesMode;
      row.hidden = !visible;
      if (visible) { shown += 1; }
    });
    rowsShown.textContent = 'Показано ' + shown + ' из ' + metricsTable.tBodies[0].rows.length;
  }

  if (metricsTable) {
    search.addEventListener('input', applyFilter);
    filter.addEventListener('change', applyFilter);
    applyFilter();
  }

  function cellValue(row, index) {
    const cell = row.cells[index];
    if (!cell) { return ''; }
    if (cell.dataset.v !== undefined) { return parseFloat(cell.dataset.v); }
    return cell.textContent.trim();
  }

  document.querySelectorAll('table.sortable').forEach(function (table) {
    table.querySelectorAll('thead th').forEach(function (th) {
      th.addEventListener('click', function () {
        const tbody = table.tBodies[0];
        const index = Array.prototype.indexOf.call(th.parentNode.children, th);
        const direction = th.dataset.dir === 'asc' ? -1 : 1;
        Array.prototype.forEach.call(th.parentNode.children, function (other) {
          delete other.dataset.dir;
        });
        th.dataset.dir = direction === 1 ? 'asc' : 'desc';
        const rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) {
          const left = cellValue(a, index);
          const right = cellValue(b, index);
          if (typeof left === 'number' && typeof right === 'number') {
            return (left - right) * direction;
          }
          return String(left).localeCompare(String(right), 'ru') * direction;
        });
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });
  });

  const download = document.getElementById('download-csv');
  if (download) {
    download.addEventListener('click', function () {
      const blob = new Blob([window.METRICS_CSV], { type: 'text/csv;charset=utf-8' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'blocks_metrics.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    });
  }

  const printButton = document.getElementById('print-page');
  if (printButton) {
    printButton.addEventListener('click', function () { window.print(); });
  }
})();
"""


def kpi(label: str, value: str, sub: str = "") -> str:
    """Карточка с ключевым числом."""
    return (
        f'<div class="card kpi"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
    )


def build_html(
    data_dir: Path,
    frame: pd.DataFrame,
    manifest: dict[str, Any] | None,
    summary: pd.DataFrame | None,
    checks: pd.DataFrame | None,
    morphotypes: pd.DataFrame | None,
    clusters: pd.DataFrame | None,
    links: pd.DataFrame | None,
    map_data: dict[str, Any],
    map_metrics: dict[str, Any],
    charts: dict[str, str],
    conclusions: list[tuple[str, str, str]],
    metrics_csv: str,
    leaflet: tuple[str, str, bool],
) -> str:
    """Собрать итоговую HTML-страницу."""
    count = len(frame)
    area = float(frame["site_area"].sum()) / 1e6
    population = float(frame["population"].sum())
    capacity = float(frame["capacity"].sum())
    demand = float(frame["demand"].sum())
    inhabited = int((frame["population"] > 0).sum())
    built = int((frame["fsi"] > 0).sum())
    schools = int((manifest or {}).get("counts", {}).get("schools", 0)) or int((frame["capacity"] > 0).sum())
    problems = int((checks["Результат"].astype(str).str.upper() != "OK").sum()) if checks is not None else 0

    territory = (manifest or {}).get("territory", "Василеостровский район, Санкт-Петербург")
    created = (manifest or {}).get("created_at", "")
    if created:
        try:
            created = datetime.fromisoformat(created).strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pass
    blocksnet = (manifest or {}).get("blocksnet_version", "—")
    crs = (manifest or {}).get("crs_epsg", "—")
    mode = (manifest or {}).get("graph_mode", "—")
    speed = (manifest or {}).get("walk_speed_kmh", "—")

    chips = [
        f"Территория: {territory}",
        f"Расчёты: {created}" if created else "",
        f"BlocksNet {blocksnet}" if blocksnet != "—" else "",
        f"EPSG:{crs}" if crs != "—" else "",
        f"Граф: {mode}, {speed} км/ч" if mode != "—" else "",
    ]
    chips_html = "".join(f'<span class="chip">{esc(chip)}</span>' for chip in chips if chip)

    kpis = "".join(
        [
            kpi("Кварталов", fmt(count, 0), f"застроено {fmt(built, 0)}"),
            kpi("Площадь", f"{fmt(area, 2)} км²", f"медиана {fmt(frame['site_area'].median() / 1e4, 1)} га"),
            kpi("Население (оценка)", fmt(population, 0), f"{fmt(inhabited, 0)} кварталов с жителями"),
            kpi("Школ", fmt(schools, 0), f"{fmt(capacity, 0)} мест"),
            kpi("Нормативный спрос", f"{fmt(demand, 0)} мест", "120 на 1000 жителей"),
            kpi("Плотность населения", f"{fmt(population / area if area else 0, 0)}", "чел./км²"),
        ]
    )

    alert = ""
    if problems:
        alert = (
            f'<div class="alert">⚠ {problems} проверка(и) не пройдена. '
            'Подробности — в разделе <a href="#checks">Проверки</a> и в выводах.</div>'
        )

    conclusion_html = "".join(
        f'<li class="{level}"><div class="title">{esc(title)}</div><div class="text">{esc(text)}</div></li>'
        for level, title, text in conclusions
    )

    summary_note = (
        '<p class="caption">Значения взяты из summary.csv без изменений. '
        "Строка про среднее время унаследовала ошибку матрицы доступности — см. «Качество данных».</p>"
    )
    summary_html = (
        simple_table(summary, "summary-table") if summary is not None else "<p>Файл summary.csv не найден.</p>"
    )

    morph_html = simple_table(morphotypes, "morphotypes-table") if morphotypes is not None else "<p>Файл не найден.</p>"
    clusters_html = simple_table(clusters, "clusters-table") if clusters is not None else "<p>Файл не найден.</p>"
    links_html = simple_table(links, "links-table") if links is not None else "<p>Файл не найден.</p>"
    checks_html = checks_table(checks) if checks is not None else "<p>Файл checks.csv не найден.</p>"

    manifest_rows = []
    if manifest:
        flat = {
            "Территория": manifest.get("territory"),
            "Дата расчёта": created,
            "Версия BlocksNet": manifest.get("blocksnet_version"),
            "OSMnx": manifest.get("osmnx_version"),
            "IduEdu": manifest.get("iduedu_version"),
            "CRS": f"EPSG:{manifest.get('crs_epsg')}",
            "Источник графа": manifest.get("graph_source"),
            "Тип графа": manifest.get("graph_mode"),
            "Скорость, км/ч": manifest.get("walk_speed_kmh"),
            "Буфер графа, м": manifest.get("graph_buffer_m"),
            "Единица времени": manifest.get("time_unit"),
            "Норматив школ": (
                f"{manifest.get('school_norm', {}).get('demand_per_1000')} мест на 1000 жителей, "
                f"радиус {manifest.get('school_norm', {}).get('accessibility_min')} мин"
            ),
            "Объектов": ", ".join(f"{key}: {value}" for key, value in (manifest.get("counts") or {}).items()),
        }
        for key, value in flat.items():
            if value not in (None, ""):
                manifest_rows.append(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>")

    files = (manifest or {}).get("files") or []
    files_html = ""
    if files:
        files_html = "<details><summary>Файлы расчёта (" + str(len(files)) + ")</summary><ul>"
        files_html += "".join(f"<li>{esc(name)}</li>" for name in files)
        files_html += "</ul></details>"

    manifest_html = (
        f'<table class="kv">{"".join(manifest_rows)}</table>{files_html}'
        if manifest_rows
        else "<p>Файл manifest.json не найден.</p>"
    )

    limits = [
        "Данные OpenStreetMap неполные: часть зданий и объектов отсутствует, у части нет этажности.",
        "Население — расчёт по жилой площади (30 м² на человека), а не перепись.",
        "Ёмкость школ в этих данных дозаполнена нормативом 250 мест, потому что в OSM её нет.",
        "Квартал — грубая единица: внутри него могут быть и школа, и пустырь, а показатель один.",
        "Обеспеченность нормативная, а не фактическая: она не знает о второй смене и переполненных классах.",
        "Учитывается только пешая доступность 5 км/ч, без общественного транспорта.",
        "Результат зависит от качества границ района, тегов OSM и настроек BlocksNet.",
    ]
    limits_html = "".join(f"<li>{esc(item)}</li>" for item in limits)

    sources = (
        f"<code>{esc(data_dir.name)}/results/blocks_metrics.geojson</code>, "
        "summary.csv, checks.csv, morphotype_counts.csv, spacematrix_clusters.csv, "
        "provision_links.csv, accessibility_matrix.csv, manifest.json, processed/schools.geojson, raw/boundary.geojson"
    )

    chart_cards = "".join(
        f'<div class="card chart">{charts[key]}</div>'
        for key in ["population", "fsi", "land_use", "morphotypes", "travel", "provision", "capacity"]
        if key in charts
    )

    metric_options = "".join(
        f'<option value="{esc(key)}">{esc(value["label"])}</option>' for key, value in map_metrics.items()
    )

    map_caption = (
        "Переключатель меняет показатель, которым раскрашены кварталы. Наведите курсор на квартал — "
        "появятся все основные числа. Слой школ и границу района можно отключить в правом верхнем углу. "
        "Подложка подгружается из интернета, сами данные встроены в файл."
    )

    leaflet_css, leaflet_js, leaflet_inline = leaflet
    if leaflet_inline:
        leaflet_head = f"<style>{leaflet_css}</style><script>{leaflet_js}</script>"
    else:
        leaflet_head = f'<link rel="stylesheet" href="{LEAFLET_CSS_URL}">' f'<script src="{LEAFLET_JS_URL}"></script>'

    data_script = (
        "window.MAP_DATA = "
        + json.dumps(map_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        + ";"
        "window.MAP_METRICS = " + json.dumps(map_metrics, ensure_ascii=False).replace("</", "<\\/") + ";"
        "window.METRICS_CSV = " + json.dumps(metrics_csv, ensure_ascii=False).replace("</", "<\\/") + ";"
    )

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(territory)} — дашборд BlocksNet</title>",
        f"<style>{CSS}</style>",
        f"<script>{get_plotlyjs()}</script>",
        leaflet_head,
        "</head>",
        "<body>",
        "<header>",
        "<h1>Василеостровский район: показатели кварталов</h1>",
        '<p class="sub">Один файл со всеми результатами расчёта: интерактивная карта, распределения, '
        "полная таблица показателей и автоматические выводы. Страница только читает готовые GeoJSON и CSV — "
        "расчёты не повторяются.</p>",
        f'<div class="chips">{chips_html}</div>',
        "</header>",
        "<nav>",
        '<a href="#map">Карта</a>',
        '<a href="#charts">Графики</a>',
        '<a href="#tables">Таблицы</a>',
        '<a href="#conclusions">Выводы</a>',
        '<a href="#checks">Проверки</a>',
        '<a href="#passport">Паспорт</a>',
        "</nav>",
        "<main>",
        f'<section id="overview"><div class="kpis">{kpis}</div></section>',
        f"{alert}",
        '<section id="map">',
        "<h2>Карта кварталов</h2>",
        '<p class="section-note">Все 615 кварталов с показателями. Цвет — выбранный показатель.</p>',
        '<div class="card map-card">',
        '<div class="map-controls">',
        '<label for="metric-select">Показатель:</label>',
        f'<select id="metric-select">{metric_options}</select>',
        '<label for="style-select">Подложка:</label>',
        '<select id="style-select">'
        '<option value="osm">OpenStreetMap</option>'
        '<option value="light">Carto Light</option>'
        '<option value="dark">Carto Dark</option>'
        '<option value="empty">Без подложки</option>'
        "</select>",
        "</div>",
        '<div id="blocks-map" class="leaflet-map"></div>',
        f'<div class="caption">{esc(map_caption)}</div>',
        '<div class="caption" id="map-note"></div>',
        "</div>",
        "</section>",
        '<section id="charts">',
        "<h2>Графики</h2>",
        '<p class="section-note">Распределения и структура показателей по готовым данным.</p>',
        f'<div class="charts">{chart_cards}</div>',
        "</section>",
        '<section id="tables">',
        "<h2>Таблицы</h2>",
        '<p class="section-note">Все сохранённые показатели. Заголовки сортируются кликом; '
        "полную таблицу можно искать и фильтровать.</p>",
        '<div class="card table-card">',
        "<h3>Сводка</h3>",
        summary_html,
        summary_note,
        "</div>",
        '<div class="card table-card">',
        "<h3>Морфотипы</h3>",
        morph_html,
        "</div>",
        '<div class="card table-card">',
        "<h3>Кластеры Spacematrix</h3>",
        clusters_html,
        "</div>",
        '<div class="card table-card">',
        "<h3>Связи обеспеченности школами</h3>",
        links_html,
        '<p class="caption">Пары «квартал спроса → квартал ёмкости» из provision_links.csv.</p>',
        "</div>",
        '<div class="card table-card" id="checks">',
        "<h3>Проверки</h3>",
        checks_html,
        "</div>",
        '<div class="card table-card">',
        "<h3>Все показатели по кварталам</h3>",
        '<div class="filters">',
        '<input type="search" id="metrics-search" placeholder="Поиск: номер квартала, категория, морфотип…" size="38">',
        '<select id="metrics-filter">'
        '<option value="all">Все кварталы</option>'
        '<option value="pop">С населением</option>'
        '<option value="school">Со школами</option>'
        '<option value="built">Застроенные (FSI > 0)</option>'
        "</select>",
        '<span class="count" id="rows-shown"></span>',
        '<button id="download-csv">Скачать CSV</button>',
        '<button id="print-page">Печать / PDF</button>',
        "</div>",
        big_table(frame),
        "</div>",
        "</section>",
        '<section id="conclusions">',
        "<h2>Выводы</h2>",
        '<p class="section-note">Сформулированы автоматически по числам из файлов — без ручных правок.</p>',
        f'<ul class="conclusions">{conclusion_html}</ul>',
        "</section>",
        '<section id="quality">',
        "<h2>Ограничения исследования</h2>",
        f'<div class="card"><ul class="limits">{limits_html}</ul></div>',
        "</section>",
        '<section id="passport">',
        "<h2>Паспорт запуска</h2>",
        '<p class="section-note">Версии, параметры и состав файлов из manifest.json.</p>',
        f'<div class="card">{manifest_html}</div>',
        "</section>",
        "</main>",
        f"<footer>Собрано dashboard.py · {datetime.now().strftime('%d.%m.%Y %H:%M')} · "
        f"источник: {sources}</footer>",
        f"<script>{data_script}</script>",
        f"<script>{MAP_JS}</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Точка входа."""
    parser = argparse.ArgumentParser(description="Собрать dashboard.html из готовых результатов BlocksNet.")
    parser.add_argument("--data-dir", help="каталог с raw/, processed/, results/")
    parser.add_argument("--out", default=str(ROOT / "dashboard.html"), help="путь к итоговому HTML")
    parser.add_argument("--no-open", action="store_true", help="не открывать браузер")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    results = data_dir / "results"

    metrics_path = results / "blocks_metrics.geojson"
    geojson = load_geojson(metrics_path)
    if not geojson:
        raise SystemExit(f"Не удалось прочитать {metrics_path}")
    frame = geojson_frame(geojson)

    frame["land_use_ru"] = frame["land_use"].map(lambda value: LAND_USE_RU.get(value, value))
    frame["morphotype_ru"] = frame["morphotype_spacematrix"].map(lambda value: MORPHO_RU.get(value, value))
    frame["morphotype_ru_spacematrix"] = frame["morphotype_ru"]
    frame["morphotype_ru_strelka"] = frame["morphotype_strelka"].map(lambda value: MORPHO_RU.get(value, value))

    manifest = read_json(results / "manifest.json")
    summary = read_csv(results / "summary.csv")
    checks = read_csv(results / "checks.csv")
    morphotypes = read_csv(results / "morphotype_counts.csv")
    clusters = read_csv(results / "spacematrix_clusters.csv")
    links = read_csv(results / "provision_links.csv")
    matrix = matrix_stats(results / "accessibility_matrix.csv")

    schools = schools_geojson(data_dir / "processed" / "schools.geojson")
    boundary = load_geojson(data_dir / "raw" / "boundary.geojson")

    capacities = {(feature.get("properties") or {}).get("capacity") for feature in (schools or {}).get("features", [])}
    schools_capacity_unique = len(capacities) if capacities else None

    map_data = build_map_payload(frame, geojson, schools, boundary)
    map_metrics = build_map_metrics(frame)

    charts = {
        "population": fig_html(chart_population(frame), "chart-population"),
        "fsi": fig_html(chart_fsi(frame), "chart-fsi"),
        "land_use": fig_html(chart_land_use(frame), "chart-land-use"),
        "morphotypes": fig_html(chart_morphotypes(frame), "chart-morphotypes"),
        "travel": fig_html(chart_travel(matrix), "chart-travel"),
        "provision": fig_html(chart_provision(frame), "chart-provision"),
        "capacity": fig_html(chart_capacity(frame), "chart-capacity"),
    }

    conclusions = conclusion_items(frame, checks, matrix, manifest, schools_capacity_unique)

    csv_path = results / "blocks_metrics.csv"
    metrics_csv = csv_path.read_text(encoding="utf-8") if csv_path.exists() else frame.to_csv(index=False)

    leaflet = load_leaflet()

    html = build_html(
        data_dir,
        frame,
        manifest,
        summary,
        checks,
        morphotypes,
        clusters,
        links,
        map_data,
        map_metrics,
        charts,
        conclusions,
        metrics_csv,
        leaflet,
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(f"Данные:      {data_dir}")
    print(f"Кварталов:   {len(frame)}")
    print(f"Графиков:    {len(charts)}")
    print(f"Leaflet:     {'встроен' if leaflet[2] else 'из CDN (нет кэша)'}")
    print(f"Файл:        {out} ({size_mb:.1f} МБ)")
    print("Готово.")

    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
