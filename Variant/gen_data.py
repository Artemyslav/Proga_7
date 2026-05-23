"""
Генератор тестовых данных для лабораторной работы.
Поддерживает числовые, временные и категориальные поля.
"""
import numpy as np
import csv
import json
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def parse_variant_fields(field_strings: list[str]) -> list[dict]:
    fields = []
    pattern = re.compile(r"^(\w+)\s*\(([\w\d]+)\)")
    for fs in field_strings:
        match = pattern.match(fs.strip())
        if match:
            fields.append({"name": match.group(1), "dtype": match.group(2)})
        else:
            parts = fs.split(":")[0].strip().split()
            fields.append({"name": parts[0], "dtype": parts[1].strip("()")})
    return fields

def compute_age_group(ages: np.ndarray) -> np.ndarray:
    """Преобразует возраст в возрастную группу."""
    groups = []
    for age in ages:
        if age < 18:
            groups.append('дети')
        elif age < 60:
            groups.append('взрослые')
        else:
            groups.append('пожилые')
    return np.array(groups, dtype=object)

def generate_field_data(name: str, dtype: str, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Генерация столбца данных на основе имени и типа."""
    n_lower = name.lower()
    dtype_np = np.dtype(dtype) if dtype not in ['datetime', 'category', 'string'] else None

    # --- Временные метки (целочисленные UNIX) ---
    if any(k in n_lower for k in ["ts", "time", "timestamp"]):
        return np.sort(rng.integers(1_600_000_000, 1_750_000_000, n_rows)).astype(np.int32)

    # --- Дата (тип datetime) ---
    if dtype == 'datetime':
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        delta_days = (end_date - start_date).days
        random_days = rng.integers(0, delta_days, n_rows)
        dates = [start_date + timedelta(days=int(d)) for d in random_days]
        return np.array([d.strftime("%Y-%m-%d") for d in dates], dtype=object)

    # --- Категориальные текстовые поля (НО НЕ patient_age_group) ---
    if dtype == 'category' or dtype == 'string':
        # patient_age_group вычисляется ПОСТ-ФАКТУМ, не генерируем случайно
        if 'patient_age_group' in n_lower:
            # Временно заполняем заглушкой, потом перезапишем
            return np.array(['взрослые'] * n_rows, dtype=object)
        
        if 'lab_type' in n_lower:
            categories = ['государственная', 'частная', 'исследовательская']
        elif 'region' in n_lower:
            categories = ['Север', 'Юг', 'Центр', 'Запад', 'Восток']
        elif 'test_reason' in n_lower:
            categories = ['профилактика', 'диагностика', 'контроль лечения']
        else:
            categories = ['A', 'B', 'C', 'D', 'E']
        return rng.choice(categories, n_rows)

    # --- Числовые поля ---
    
    # Возраст (НОВЫЙ числовой признак)
    if 'age' in n_lower:
        # Бета-распределение с пиком в районе 30-50 лет, диапазон 0-100
        ages = rng.beta(a=2, b=3, size=n_rows) * 100
        return ages.astype(np.int16)
    
    # Идентификаторы и коды
    if any(k in n_lower for k in ["id", "code", "type", "flag", "status", "zone", "err", "alarm", "fault", "sleep", "wcode", "pay_type", "art", "fall"]):
        limit = 500 if any(x in n_lower for x in ["user", "patient", "student", "acc", "hotel", "airline"]) else 100
        return rng.integers(0, limit, n_rows).astype(dtype_np)

    # Проценты, доли, рейтинги
    if any(k in n_lower for k in ["hum", "occ", "load", "batt", "fill", "cap", "eff", "mois", "succ", "ctr", "comp", "leak", "churn", "err_r", "viol_r", "cong", "drops", "fid", "qerr"]):
        if "ph" in n_lower:
            return rng.uniform(6.5, 8.5, n_rows).astype(np.float32)
        if "rating" in n_lower or n_lower.startswith("rat"):
            return rng.uniform(1.0, 5.0, n_rows).astype(np.float32)
        if any(k in n_lower for k in ["succ", "ctr", "comp", "leak", "churn", "err_r", "viol_r", "fid", "qerr"]):
            return rng.uniform(0.05, 0.95, n_rows).astype(np.float32)
        return rng.uniform(5.0, 95.0, n_rows).astype(np.float32)

    # Цены, тарифы, объёмы, медицинские показатели
    if any(k in n_lower for k in ["price", "fare", "rate", "amt", "rev", "mrr", "royal", "avg_bill", "cpc", "gain", "power", "bit", "latency", "ping", "alt", "km", "dist", "area", "weight", "mass", "flow", "volt", "cur", "tvol", "bpm", "glu", "hgb", "wbc", "plt", "eos", "hist", "ige", "symp", "iop", "thick", "acu", "dp", "ery", "mel", "hours", "proc_t", "door_t", "vib", "wait", "cpu", "pos_err", "comf", "uv", "ozone", "pm25", "aqi", "temp", "press", "strain", "wind", "sp", "vel", "drag", "reverb", "db", "laeq", "la10", "la90", "freq", "wave", "intensity", "coh", "noise_db", "bright", "depth", "fps", "layer", "tnoz", "gps", "lat", "long"]):
        if any(k in n_lower for k in ["temp", "press", "volt", "alt", "strain", "wind", "sp", "vel"]):
            return rng.normal(20.0, 10.0, n_rows).astype(np.float32)
        if any(k in n_lower for k in ["glu", "hgb", "wbc", "plt"]):
            if 'glu' in n_lower:
                return np.clip(rng.normal(5.5, 2.0, n_rows), 2.0, 15.0).astype(np.float32)
            if 'hgb' in n_lower:
                return np.clip(rng.normal(140, 20, n_rows), 60, 200).astype(np.float32)
            if 'wbc' in n_lower:
                return np.clip(rng.normal(7, 3, n_rows), 1, 20).astype(np.float32)
            if 'plt' in n_lower:
                return np.clip(rng.normal(250, 70, n_rows), 50, 500).astype(np.float32)
        return np.abs(rng.normal(50.0, 25.0, n_rows)).astype(np.float32)

    # Счётчики, количества
    if any(k in n_lower for k in ["qty", "items", "steps", "likes", "orders", "cycl", "veh", "tix", "shows", "nights", "players", "turns", "gates", "imgs", "unlocks", "dl", "plays", "covers", "pick", "entry", "evt", "awak", "attempts", "crash", "fall"]):
        return rng.integers(1, 150, n_rows).astype(dtype_np)

    if "float" in dtype:
        return rng.normal(0, 10, n_rows).astype(dtype_np)
    return rng.integers(0, 50, n_rows).astype(dtype_np)

def inject_anomalies(data: dict, fields: list[dict], n_rows: int, rng: np.random.Generator):
    """~3% пропусков, ~3% отрицательных значений и ~3% выбросов в float-полях."""
    for f in fields:
        if f["dtype"] in ['datetime', 'category', 'string']:
            continue
        if "float" not in f["dtype"]:
            continue
        arr = data[f["name"]]
        # Пропуски (NaN) ~3%
        nan_mask = rng.random(n_rows) < 0.03
        arr[nan_mask] = np.nan
        # Отрицательные значения ~3%
        neg_mask = rng.random(n_rows) < 0.03
        valid_neg = ~np.isnan(arr) & neg_mask
        if valid_neg.any():
            arr[valid_neg] = -np.abs(arr[valid_neg])
        # Выбросы ~3%
        out_mask = rng.random(n_rows) < 0.03
        valid_out = ~np.isnan(arr) & out_mask
        if valid_out.any():
            lower, upper = np.nanpercentile(arr, [5, 95])
            span = upper - lower if upper > lower else 10.0
            arr[valid_out] = upper + span * rng.uniform(1.5, 4.0, valid_out.sum())

def main():
    parser = argparse.ArgumentParser(description="Генератор CSV данных для лабораторной работы")
    parser.add_argument("-variant", type=int, required=True, help="Номер варианта (1-60)")
    parser.add_argument("-rows", type=int, default=2_000_000, help="Количество строк")
    parser.add_argument("-seed", type=int, default=77, help="Сид генератора")
    parser.add_argument("-output", type=str, default=None, help="Имя выходного файла CSV")
    parser.add_argument("-config", type=str, default="variant.json", help="Путь к variant.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Файл конфигурации не найден: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not (1 <= args.variant <= len(config["variants"])):
        print(f"Вариант должен быть от 1 до {len(config['variants'])}")
        sys.exit(1)

    variant = config["variants"][args.variant - 1]
    output_file = args.output or f"data_variant_{args.variant}.csv"
    rng = np.random.default_rng(args.seed)
    fields = parse_variant_fields(variant["fields"])
    n_rows = args.rows

    print(f"Генерация данных для Варианта {variant['id']}: {variant['title']}")
    print(f"Строк: {n_rows:,} | Seed: {args.seed} | Полей: {len(fields)}")

    data = {}
    for f in fields:
        data[f["name"]] = generate_field_data(f["name"], f["dtype"], n_rows, rng)

    # ПОСТ-ОБРАБОТКА: вычисляем возрастную группу на основе возраста
    if 'age' in data and 'patient_age_group' in data:
        data['patient_age_group'] = compute_age_group(data['age'])
        print("  → Вычислена возрастная группа на основе возраста")

    inject_anomalies(data, fields, n_rows, rng)

    print("Запись в CSV...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f["name"] for f in fields])
        chunk_size = 50_000
        for i in range(0, n_rows, chunk_size):
            end = min(i + chunk_size, n_rows)
            rows = zip(*(data[f["name"]][i:end] for f in fields))
            writer.writerows(rows)

    size_mb = os.path.getsize(output_file) / 1024**2
    print(f"Готово: {output_file} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()