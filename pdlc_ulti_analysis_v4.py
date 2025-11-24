# PDLC Değer Zinciri + Maliyet Entegre Arayüzü
# v4 – pdlc_cost_app_v5'in Genel Parametreler + Özet&Break-even yapısı ile uyumlu
# Atama oranı (%) tüm stream'lerde aktif

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# -------------------------------------------------------------------
# Sabit listeler – Porter aktiviteleri ve default VCA satırları
# -------------------------------------------------------------------

VALUE_CHAIN_ACTIVITIES = [
    "Inbound Logistics",
    "Operations",
    "Outbound Logistics",
    "Marketing & Sales",
    "Service",
    "Firm Infrastructure",
    "HR Management",
    "Technology Development",
    "Procurement",
]

# Orijinal VCA default satırları
DEFAULT_PRIMARY = [
    ["Inbound Logistics", "Tedarikçi seçimi & kalifikasyonu", 4, 3, 3, 3],
    ["Inbound Logistics", "Mal kabul & karantina", 3, 3, 2, 2],
    ["Inbound Logistics", "IQC testleri (AQL/SPC)", 4, 2, 3, 4],
    ["Operations", "Dope hazırlama", 4, 3, 3, 3],
    ["Operations", "R2R kaplama & UV kürleme", 5, 2, 5, 4],
    ["Operations", "Rulo kalite kontrolü", 4, 3, 2, 3],
    ["Operations", "Kesim & boyutlandırma", 3, 3, 3, 2],
    ["Operations", "Busbar & kablolama", 4, 2, 3, 4],
    ["Operations", "Laminasyon", 5, 3, 5, 4],
    ["Operations", "Modül EOL test", 4, 3, 2, 4],
    ["Outbound Logistics", "Paketleme & palletizasyon", 3, 3, 3, 3],
    [
        "Outbound Logistics",
        "Sevkiyat planlama & taşıyıcı yönetimi",
        3,
        2,
        3,
        3,
    ],
    [
        "Marketing & Sales",
        "Proje geliştirme & mimar ilişkileri",
        4,
        2,
        2,
        3,
    ],
    ["Marketing & Sales", "Tekliflendirme & fiyatlama", 4, 3, 2, 2],
    ["Service", "Montaj partner desteği", 4, 2, 3, 4],
    ["Service", "Devreye alma", 4, 3, 2, 3],
    ["Service", "RMA & 8D/CAPA", 5, 3, 3, 5],
]

DEFAULT_SUPPORT = [
    [
        "Firm Infrastructure",
        "Kalite yönetim sistemi & dokümantasyon",
        4,
        3,
        2,
        4,
    ],
    ["Firm Infrastructure", "Strateji & iş planı", 5, 3, 1, 4],
    ["HR Management", "Operatör eğitim planı", 4, 2, 2, 3],
    [
        "HR Management",
        "Mühendis / Ar-Ge yetkinlik yönetimi",
        4,
        3,
        2,
        3,
    ],
    [
        "Technology Development",
        "Reçete geliştirme (PDLC formülasyon)",
        5,
        3,
        3,
        4,
    ],
    [
        "Technology Development",
        "Proses penceresi optimizasyonu",
        5,
        2,
        3,
        4,
    ],
    [
        "Technology Development",
        "Dayanım & yaşlandırma testleri",
        4,
        3,
        2,
        4,
    ],
    [
        "Procurement",
        "Stratejik satınalma & dual sourcing",
        4,
        3,
        3,
        4,
    ],
    [
        "Procurement",
        "Tedarikçi performans yönetimi",
        4,
        2,
        2,
        3,
    ],
]

# -------------------------------------------------------------------
# Değer zinciri tarafı – init & hesap fonksiyonları
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PDLC Value Chain Analyzer", layout="wide")
    st.title("PDLC Değer Zinciri – Porter Value Chain Analiz Arayüzü")

    # Yardım / metodoloji açıklaması
    with st.expander("Yardım / Metodoloji – Tanımlar ve 1–5 Ölçekleri", expanded=False):
        st.markdown(
            """
**Bu çalışmada kullanılan ana kavramlar**

- **Değer Potansiyeli (1–5)**  
  Sürecin, müşteriye değer ve rekabet avantajı yaratma potansiyeli.

- **Mevcut Performans (1–5)**  
  Bugünkü olgunluk ve performans seviyesi.

- **Maliyet Etkisi (1–5)**  
  Sürecin toplam maliyet yapısı içindeki ağırlığı (OPEX / CAPEX / fire vb.).

- **Risk / Kayıp (1–5)**  
  Süreçte sorun çıktığında kalite, müşteri, güvenlik veya finansal etki düzeyi.

- **Değer Açığı = Değer Potansiyeli − Mevcut Performans**  

- **Öncelik (Risk Bazlı) = Değer Açığı × Risk / Kayıp**  

- **Öncelik (Maliyet Bazlı) = Değer Açığı × Maliyet Etkisi**  

Bu sürümde **Yıllık Hacim (m²)** kolonunu; nihai PDLC cam çıktısı, R2R fire oranı ve proses aşamalarına göre
model otomatik hesaplamaktadır. Kullanıcıdan sadece **nihai m²** ve **proses parametreleri** alınır.
"""
        )

    init_session_state()
    df_base = st.session_state.df

@st.cache_data
def get_default_value_chain_df() -> pd.DataFrame:
    """Primary + Support için varsayılan VCA tablosu."""
    prim = pd.DataFrame(
        DEFAULT_PRIMARY,
        columns=[
            "Kategori",
            "Alt Süreç",
            "Değer Potansiyeli",
            "Mevcut Performans",
            "Maliyet Etkisi",
            "Risk / Kayıp",
        ],
    )
    prim["Grup"] = "Primary"

    supp = pd.DataFrame(
        DEFAULT_SUPPORT,
        columns=[
            "Kategori",
            "Alt Süreç",
            "Değer Potansiyeli",
            "Mevcut Performans",
            "Maliyet Etkisi",
            "Risk / Kayıp",
        ],
    )
    supp["Grup"] = "Support"

    df = pd.concat([prim, supp], ignore_index=True)
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """VCA satırları için Değer Açığı ve Öncelik skorlarını hesaplar."""
    df = df.copy()
    df["Değer Açığı"] = df["Değer Potansiyeli"] - df["Mevcut Performans"]
    df["Öncelik_Risk"] = df["Değer Açığı"] * df["Risk / Kayıp"]
    df["Öncelik_Maliyet"] = df["Değer Açığı"] * df["Maliyet Etkisi"]
    return df


def summarize_value_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Satır bazlı VCA tablosundan, Porter aktivitesi bazında stratejik özet üretir.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Değer Zinciri Aktivitesi",
                "Stratejik_Skor",
                "Mevcut_Performans",
                "Değer_Açığı",
                "Öncelik_Risk",
                "Öncelik_Maliyet",
            ]
        )

    df_derived = compute_derived(df)

    grouped = (
        df_derived.groupby("Kategori")
        .agg(
            Stratejik_Skor=("Değer Potansiyeli", "mean"),
            Mevcut_Performans=("Mevcut Performans", "mean"),
            Değer_Açığı=("Değer Açığı", "mean"),
            Öncelik_Risk=("Öncelik_Risk", "sum"),
            Öncelik_Maliyet=("Öncelik_Maliyet", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"Kategori": "Değer Zinciri Aktivitesi"})
    return grouped


# -------------------------------------------------------------------
# Maliyet tarafı – init fonksiyonları
# -------------------------------------------------------------------


def init_cost_table_upstream() -> pd.DataFrame:
    """Varsayılan upstream (hammadde / giriş) kalemleri."""
    data = {
        "Tip": ["malzeme", "malzeme"],
        "Kalem": ["ITO kaplı PET", "PDLC karışımı"],
        "Değer Zinciri Aktivitesi": ["Inbound Logistics", "Inbound Logistics"],
        "Dağıtım hedefi": ["per_m2", "per_m2"],
        "Birim": ["m²", "g"],
        "Ürün birim miktar kullanımı": [2.0, 21.0],  # 1 m² PDLC için
        "Birim maliyet (USD)": [0.0, 0.0],
        "Fire oranı": [0.05, 0.05],
        "Sabit/Değişken": ["Değişken", "Değişken"],
        "Atama oranı (%)": [100.0, 100.0],
    }
    return pd.DataFrame(data)


def init_cost_table_midstream() -> pd.DataFrame:
    """Varsayılan midstream (proses, enerji, işçilik) kalemleri."""
    data = {
        "Tip": ["işçilik", "enerji"],
        "Kalem": ["Operatör işçilik (lot)", "Hat enerji tüketimi"],
        "Değer Zinciri Aktivitesi": ["Operations", "Operations"],
        "Dağıtım hedefi": ["per_lot", "per_saat"],
        "Birim": ["kişi-saat", "USD/saat"],
        "Ürün birim miktar kullanımı": [8.0, 1.0],
        "Birim maliyet (USD)": [0.0, 0.0],
        "Fire oranı": [0.0, 0.0],
        "Sabit/Değişken": ["Sabit", "Değişken"],
        "Atama oranı (%)": [100.0, 100.0],
    }
    return pd.DataFrame(data)


def init_cost_table_downstream(
    paste_per_part_g: float,
    foil_length_per_part_m: float,
    cable_length_per_part_m: float,
    part_area_m2: float,
    pvb_m2_per_m2_pdlc: float,
    band_m_per_m2_pdlc: float,
) -> pd.DataFrame:
    """
    Downstream (laminasyon, bağlantı elemanları) için 1 m² PDLC'ye göre türetilmiş default kalemler.
    """
    paste_per_m2 = paste_per_part_g / part_area_m2 if part_area_m2 > 0 else 0.0
    foil_per_m2 = (
        foil_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
    )
    cable_per_m2 = (
        cable_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
    )

    data = {
        "Tip": ["malzeme", "malzeme", "malzeme", "malzeme", "malzeme"],
        "Kalem": [
            "İletken pasta",
            "İletken folyo",
            "Güç aktarım kablosu",
            "PVB ara katmanı",
            "Laminasyon bandı",
        ],
        "Değer Zinciri Aktivitesi": [
            "Outbound Logistics",
            "Outbound Logistics",
            "Outbound Logistics",
            "Outbound Logistics",
            "Outbound Logistics",
        ],
        "Dağıtım hedefi": ["per_m2"] * 5,
        "Birim": ["g", "m", "m", "m²", "m"],
        "Ürün birim miktar kullanımı": [
            paste_per_m2,
            foil_per_m2,
            cable_per_m2,
            pvb_m2_per_m2_pdlc,
            band_m_per_m2_pdlc,
        ],
        "Birim maliyet (USD)": [0.0] * 5,
        "Fire oranı": [0.05] * 5,
        "Sabit/Değişken": ["Değişken"] * 5,
        "Atama oranı (%)": [100.0] * 5,
    }
    return pd.DataFrame(data)


def init_cost_table_support() -> pd.DataFrame:
    """Varsayılan destek birimleri / overhead kalemleri."""
    data = {
        "Tip": ["diğer"],
        "Kalem": ["Genel yönetim & ofis giderleri"],
        "Değer Zinciri Aktivitesi": ["Firm Infrastructure"],
        "Dağıtım hedefi": ["per_yıl"],
        "Birim": ["USD/yıl"],
        "Ürün birim miktar kullanımı": [1.0],
        "Birim maliyet (USD)": [0.0],  # yıllık toplam tutar
        "Fire oranı": [0.0],
        "Sabit/Değişken": ["Sabit"],
        "Atama oranı (%)": [50.0],  # varsayılan: bu giderin %50'si PDLC'ye yazılsın
    }
    return pd.DataFrame(data)


@st.cache_data
def get_default_upstream_df() -> pd.DataFrame:
    return init_cost_table_upstream()


@st.cache_data
def get_default_midstream_df() -> pd.DataFrame:
    return init_cost_table_midstream()


@st.cache_data
def get_default_support_df() -> pd.DataFrame:
    return init_cost_table_support()


def ensure_cost_session_state(downstream_df_init: pd.DataFrame | None = None):
    """
    Session state içinde upstream/midstream/downstream/support tabloları yoksa default olarak yaratır.
    """
    if "upstream_df" not in st.session_state:
        st.session_state["upstream_df"] = get_default_upstream_df().copy()

    if "midstream_df" not in st.session_state:
        st.session_state["midstream_df"] = get_default_midstream_df().copy()

    if "support_df" not in st.session_state:
        st.session_state["support_df"] = get_default_support_df().copy()

    if "downstream_df" not in st.session_state:
        if downstream_df_init is None:
            st.session_state["downstream_df"] = pd.DataFrame(
                columns=[
                    "Tip",
                    "Kalem",
                    "Değer Zinciri Aktivitesi",
                    "Dağıtım hedefi",
                    "Birim",
                    "Ürün birim miktar kullanımı",
                    "Birim maliyet (USD)",
                    "Fire oranı",
                    "Sabit/Değişken",
                    "Atama oranı (%)",
                ]
            )
        else:
            st.session_state["downstream_df"] = downstream_df_init.copy()


# -------------------------------------------------------------------
# Parametre defaultları (session_state üzerinden)
# -------------------------------------------------------------------


def init_param_defaults():
    defaults = {
        "roll_length_m": 300.0,
        "roll_width_m": 1.5,
        "scrap_each_end_m": 4.0,  # her uç için fire (m)
        "part_length_m": 1.5,
        "part_width_m": 1.0,
        "paste_per_part_g": 0.5,
        "foil_length_per_part_m": 0.40,
        "cable_length_per_part_m": 0.90,
        "pvb_m2_per_m2_pdlc": 1.0,
        "band_m_per_m2_pdlc": 0.5,
        "lot_duration_hours": 6.0,
        "production_input_mode": "Yıllık üretim alanı (m²)",
        "annual_good_m2": 0.0,
        "annual_lot_count": 0.0,
        "selling_price_per_m2": 100.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------------------------------------------------------------
# Maliyet tarafı – hesap fonksiyonları
# -------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cost DF içinde beklenen kolonların varlığını ve temel default değerlerini garanti eder.
    Var olan diğer kolonlar korunur.
    """
    required = [
        "Tip",
        "Kalem",
        "Değer Zinciri Aktivitesi",
        "Dağıtım hedefi",
        "Birim",
        "Ürün birim miktar kullanımı",
        "Birim maliyet (USD)",
        "Fire oranı",
        "Sabit/Değişken",
        "Atama oranı (%)",
    ]

    for c in required:
        if c not in df.columns:
            if c in [
                "Ürün birim miktar kullanımı",
                "Birim maliyet (USD)",
                "Fire oranı",
            ]:
                df[c] = 0.0
            elif c == "Sabit/Değişken":
                df[c] = "Değişken"
            elif c == "Değer Zinciri Aktivitesi":
                df[c] = ""
            elif c == "Atama oranı (%)":
                df[c] = 100.0
            else:
                df[c] = ""
    return df


def calculate_section_costs(
    df: pd.DataFrame,
    annual_lot_count: float,
    annual_good_m2: float,
    lot_duration_hours: float,
    is_support: bool = False,
):
    """
    Tek bir stream (upstream/midstream/downstream/support) için:
      - satır bazlı brüt yıllık maliyet
      - atama oranı ile çarpılmış yıllık maliyet
      - toplam sabit/değişken yıllık maliyet
    döner.
    """
    df = ensure_columns(df.copy())

    # Sayısal kolonlar
    for col in ["Ürün birim miktar kullanımı", "Birim maliyet (USD)", "Fire oranı"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Atama oranı (%)"] = pd.to_numeric(
        df["Atama oranı (%)"], errors="coerce"
    ).fillna(100.0)

    annual_costs_raw = []
    allocated_costs = []

    for _, row in df.iterrows():
        dist = str(row["Dağıtım hedefi"]).strip()
        qty = row["Ürün birim miktar kullanımı"]
        unit_cost = row["Birim maliyet (USD)"]
        scrap_rate = row["Fire oranı"]

        # Yıllık miktar (dağıtım hedefine göre)
        if dist == "per_lot":
            annual_qty = qty * annual_lot_count
        elif dist == "per_m2":
            annual_qty = qty * annual_good_m2
        elif dist == "per_yıl":
            annual_qty = qty
        elif dist == "per_ay":
            annual_qty = qty * 12.0
        elif dist == "per_saat":
            annual_qty = qty * lot_duration_hours * annual_lot_count
        elif dist == "per_kişi_saat":
            annual_qty = qty * lot_duration_hours * annual_lot_count
        else:
            annual_qty = 0.0

        annual_qty = annual_qty * (1.0 + scrap_rate)
        annual_cost_raw = annual_qty * unit_cost
        annual_costs_raw.append(annual_cost_raw)

        alloc_rate = row["Atama oranı (%)"] / 100.0
        allocated_costs.append(annual_cost_raw * alloc_rate)

    df["Brüt yıllık maliyet (USD)"] = annual_costs_raw
    df["Yıllık maliyet (USD)"] = allocated_costs  # atanmış kısım

    cost_for_break_even = np.array(allocated_costs)
    total_annual_cost = float(np.sum(cost_for_break_even))

    # Sabit / değişken ayrımı (atanmış maliyet üzerinden)
    fixed_mask = (
        df["Sabit/Değişken"].fillna("Değişken").str.lower() == "sabit"
    )
    fixed_cost_annual = float(cost_for_break_even[fixed_mask].sum())
    variable_cost_annual = float(cost_for_break_even[~fixed_mask].sum())

    return df, total_annual_cost, fixed_cost_annual, variable_cost_annual


def aggregate_costs_by_activity(
    upstream_df_calc: pd.DataFrame,
    midstream_df_calc: pd.DataFrame,
    downstream_df_calc: pd.DataFrame,
    support_df_calc: pd.DataFrame,
    annual_good_m2: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tüm stream'lerden gelen satır bazlı cost DF'lerini birleştirir ve
    Porter aktivitesi bazında yıllık maliyet + m² başına maliyet hesaplar.
    'Yıllık maliyet (USD)' kolonunu atanmış maliyet olarak kullanır.
    """
    # Detaylı tablo: stream ismi ile beraber concat
    up_export = upstream_df_calc.copy()
    up_export["Seviye"] = "Upstream"

    mid_export = midstream_df_calc.copy()
    mid_export["Seviye"] = "Midstream"

    down_export = downstream_df_calc.copy()
    down_export["Seviye"] = "Downstream"

    sup_export = support_df_calc.copy()
    sup_export["Seviye"] = "Support"

    df_cost_detailed = pd.concat(
        [up_export, mid_export, down_export, sup_export],
        ignore_index=True,
    )

    # Aktivite bazında özet
    rows = []
    for df_section, seviye in [
        (upstream_df_calc, "Upstream"),
        (midstream_df_calc, "Midstream"),
        (downstream_df_calc, "Downstream"),
        (support_df_calc, "Support"),
    ]:
        if df_section is None or df_section.empty:
            continue

        df_tmp = ensure_columns(df_section.copy())

        for _, row in df_tmp.iterrows():
            act = str(row.get("Değer Zinciri Aktivitesi", "")).strip()
            if not act:
                act = "Unassigned"
            annual_cost = float(row.get("Yıllık maliyet (USD)", 0.0))
            is_fixed = (
                str(row.get("Sabit/Değişken", "Değişken")).lower() == "sabit"
            )
            rows.append(
                {
                    "Değer Zinciri Aktivitesi": act,
                    "Yıllık Maliyet (USD)": annual_cost,
                    "Sabit": annual_cost if is_fixed else 0.0,
                    "Değişken": 0.0 if is_fixed else annual_cost,
                }
            )

    if not rows:
        df_cost_by_activity = pd.DataFrame(
            columns=[
                "Değer Zinciri Aktivitesi",
                "Yıllık Maliyet (USD)",
                "m2 Başına Maliyet (USD/m2)",
                "Sabit Oran",
                "Değişken Oran",
            ]
        )
        return df_cost_detailed, df_cost_by_activity

    df_agg = pd.DataFrame(rows)
    grouped = df_agg.groupby("Değer Zinciri Aktivitesi").sum(numeric_only=True)

    grouped["Toplam Maliyet (USD)"] = grouped["Yıllık Maliyet (USD)"]
    if annual_good_m2 > 0:
        grouped["m2 Başına Maliyet (USD/m2)"] = (
            grouped["Yıllık Maliyet (USD)"] / annual_good_m2
        )
    else:
        grouped["m2 Başına Maliyet (USD/m2)"] = 0.0

    grouped["Sabit Oran"] = grouped["Sabit"] / grouped["Yıllık Maliyet (USD)"].replace(
        0, np.nan
    )
    grouped["Değişken Oran"] = grouped["Değişken"] / grouped[
        "Yıllık Maliyet (USD)"
    ].replace(0, np.nan)

    df_cost_by_activity = grouped.reset_index().fillna(0.0)

    return df_cost_detailed, df_cost_by_activity


def calculate_total_cost(
    upstream_total_annual: float,
    midstream_total_annual: float,
    downstream_total_annual: float,
    support_total_annual: float,
) -> float:
    """Tüm stream'lerin yıllık toplam (atanmış) maliyetinin hesaplanması."""
    return (
        upstream_total_annual
        + midstream_total_annual
        + downstream_total_annual
        + support_total_annual
    )


def calculate_cost_per_m2(total_annual_cost: float, annual_good_m2: float) -> float:
    """Yıllık toplam maliyet → 1 m² başına maliyet."""
    if annual_good_m2 <= 0:
        return 0.0
    return total_annual_cost / annual_good_m2


def calculate_break_even_point(
    total_fixed_annual: float,
    variable_cost_per_m2: float,
    selling_price_per_m2: float,
) -> float | None:
    """
    Basit başabaş noktası: Q* = Sabit yıllık maliyet / (Satış fiyatı - Değişken maliyet).
    """
    margin = selling_price_per_m2 - variable_cost_per_m2
    if margin <= 0:
        return None
    return total_fixed_annual / margin


# -------------------------------------------------------------------
# Görselleştirme fonksiyonları
# -------------------------------------------------------------------


def plot_value_chain_heatmap(df_vc_cost: pd.DataFrame):
    """
    Değer zinciri aktiviteleri için:
      - Stratejik Skor
      - m2 Başına Maliyet
      - Toplam Maliyet
    kolonlarını renk skalalı heatmap olarak gösterir.
    """
    if df_vc_cost.empty:
        st.info(
            "Henüz maliyet/veri girişi yapılmadı. Önce maliyet ve VCA girdilerini doldurun."
        )
        return

    metrics = [
        "Stratejik_Skor",
        "m2 Başına Maliyet (USD/m2)",
        "Toplam Maliyet (USD)",
    ]
    df_plot = df_vc_cost.copy()
    for m in metrics:
        if m not in df_plot.columns:
            df_plot[m] = 0.0
        df_plot[m] = pd.to_numeric(df_plot[m], errors="coerce").fillna(0.0)

    data = df_plot[metrics].values.astype(float)

    norm_data = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin > 1e-9:
            norm_data[:, j] = (col - cmin) / (cmax - cmin)
        else:
            norm_data[:, j] = 0.0

    fig, ax = plt.subplots(
        figsize=(6, max(4, 0.5 * len(df_plot["Değer Zinciri Aktivitesi"])))
    )
    im = ax.imshow(norm_data, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(len(df_plot)))
    ax.set_yticklabels(df_plot["Değer Zinciri Aktivitesi"])

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        ["Stratejik Skor", "Maliyet/m² (USD)", "Toplam Maliyet (USD)"],
        rotation=45,
        ha="right",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalize skor (0–1)")

    ax.set_title("Değer Zinciri Isı Haritası – Strateji + Maliyet")

    st.pyplot(fig)


# -------------------------------------------------------------------
# Sayfa: Maliyet & Break-even
# -------------------------------------------------------------------


def render_cost_page(
    annual_good_m2: float,
    annual_lot_count: float,
    lot_duration_hours: float,
    roll_length_m: float,
    roll_width_m: float,
    part_length_m: float,
    part_width_m: float,
    paste_per_part_g: float,
    foil_length_per_part_m: float,
    cable_length_per_part_m: float,
    pvb_m2_per_m2_pdlc: float,
    band_m_per_m2_pdlc: float,
):
    """Maliyet & break-even sayfasının gövdesi (tabs + grafikler)."""

    # Rulo metrikleri (fire dahil)
    scrap_each_end_m = st.session_state.get("scrap_each_end_m", 0.0)
    total_scrap_m = 2.0 * scrap_each_end_m
    net_length_m = max(roll_length_m - total_scrap_m, 0.0)
    gross_area_m2 = roll_length_m * roll_width_m
    net_m2_per_lot = net_length_m * roll_width_m

    part_area_m2 = part_length_m * part_width_m

    tabs = st.tabs(
        [
            "Genel Parametreler",
            "Upstream",
            "Midstream",
            "Downstream",
            "Destek Birimleri",
            "Özet & Break-even",
        ]
    )

    # --- TAB 0: Genel Parametreler ---
    with tabs[0]:
        st.subheader("Genel Proses Özeti")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rulo boyu (m)", f"{roll_length_m:.1f}")
            st.metric("Rulo genişliği (m)", f"{roll_width_m:.2f}")
        with col_b:
            st.metric("Toplam fire (m)", f"{total_scrap_m:.1f}")
            st.metric("Net uzunluk (m)", f"{net_length_m:.1f}")
        with col_c:
            st.metric("Brüt alan (m²)", f"{gross_area_m2:.1f}")
            st.metric("Net alan / lot (m²)", f"{net_m2_per_lot:.1f}")

        st.write("### 1 m² PDLC için temel varsayımlar")
        st.write(
            "- 2 m² ITO kaplı PET kullanılır.\n"
            "- 21 g PDLC karışımı (LC + polimer + solvent vb.) kullanılır.\n"
            "- Örnek parça: "
            f"{part_length_m:.2f} × {part_width_m:.2f} m (alan ≈ {part_area_m2:.2f} m²)."
        )
        st.write(
            "Bu varsayımlar maliyet tablolarındaki satır tanımlarında referans olarak kullanılabilir."
        )

    upstream_df = st.session_state["upstream_df"]
    midstream_df = st.session_state["midstream_df"]
    downstream_df = st.session_state["downstream_df"]
    support_df = st.session_state["support_df"]

    # --- TAB 1: Upstream ---
    with tabs[1]:
        st.subheader("Upstream Maliyetler (Hammadde ve girişler)")

        upstream_df_input = st.data_editor(
            upstream_df,
            key="upstream_editor",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Tip": st.column_config.SelectboxColumn(
                    "Tip",
                    options=[
                        "malzeme",
                        "işçilik",
                        "enerji",
                        "bakım",
                        "amortisman",
                        "diğer",
                    ],
                ),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi",
                    options=VALUE_CHAIN_ACTIVITIES,
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_lot",
                        "per_m2",
                        "per_yıl",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Ürün birim miktar kullanımı": st.column_config.NumberColumn(
                    "Ürün birim miktar kullanımı"
                ),
                "Birim maliyet (USD)": st.column_config.NumberColumn(
                    "Birim maliyet (USD)", format="%.4f"
                ),
                "Fire oranı": st.column_config.NumberColumn(
                    "Fire oranı",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Sabit/Değişken": st.column_config.SelectboxColumn(
                    "Sabit/Değişken",
                    options=["Değişken", "Sabit"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                    help="Bu kalemin yüzde kaçının PDLC ürününe/hatta atanacağı.",
                ),
            },
        )
        st.session_state["upstream_df"] = upstream_df_input

        (
            upstream_df_calc,
            upstream_total_annual,
            upstream_fixed_annual,
            upstream_var_annual,
        ) = calculate_section_costs(
            upstream_df_input,
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )

        st.write("#### Hesaplanmış yıllık maliyetler (Upstream)")
        st.dataframe(
            upstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Upstream (atanmış) toplam yıllık maliyet: {upstream_total_annual:,.2f} USD"
        )

    # --- TAB 2: Midstream ---
    with tabs[2]:
        st.subheader("Midstream Maliyetler (Proses, enerji, işçilik)")

        midstream_df_input = st.data_editor(
            midstream_df,
            key="midstream_editor",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Tip": st.column_config.SelectboxColumn(
                    "Tip",
                    options=[
                        "malzeme",
                        "işçilik",
                        "enerji",
                        "bakım",
                        "amortisman",
                        "diğer",
                    ],
                ),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi",
                    options=VALUE_CHAIN_ACTIVITIES,
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_lot",
                        "per_m2",
                        "per_yıl",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Ürün birim miktar kullanımı": st.column_config.NumberColumn(
                    "Ürün birim miktar kullanımı"
                ),
                "Birim maliyet (USD)": st.column_config.NumberColumn(
                    "Birim maliyet (USD)", format="%.4f"
                ),
                "Fire oranı": st.column_config.NumberColumn(
                    "Fire oranı",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Sabit/Değişken": st.column_config.SelectboxColumn(
                    "Sabit/Değişken",
                    options=["Değişken", "Sabit"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                    help="Bu kalemin yüzde kaçının PDLC ürününe/hatta atanacağı.",
                ),
            },
        )
        st.session_state["midstream_df"] = midstream_df_input

        (
            midstream_df_calc,
            midstream_total_annual,
            midstream_fixed_annual,
            midstream_var_annual,
        ) = calculate_section_costs(
            midstream_df_input,
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )

        st.write("#### Hesaplanmış yıllık maliyetler (Midstream)")
        st.dataframe(
            midstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Midstream (atanmış) toplam yıllık maliyet: {midstream_total_annual:,.2f} USD"
        )

    # --- TAB 3: Downstream ---
    with tabs[3]:
        st.subheader("Downstream Maliyetler (Laminasyon ve çıkış)")

        downstream_df_input = st.data_editor(
            downstream_df,
            key="downstream_editor",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Tip": st.column_config.SelectboxColumn(
                    "Tip",
                    options=[
                        "malzeme",
                        "işçilik",
                        "enerji",
                        "bakım",
                        "amortisman",
                        "diğer",
                    ],
                ),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi",
                    options=VALUE_CHAIN_ACTIVITIES,
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_lot",
                        "per_m2",
                        "per_yıl",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Ürün birim miktar kullanımı": st.column_config.NumberColumn(
                    "Ürün birim miktar kullanımı"
                ),
                "Birim maliyet (USD)": st.column_config.NumberColumn(
                    "Birim maliyet (USD)", format="%.4f"
                ),
                "Fire oranı": st.column_config.NumberColumn(
                    "Fire oranı",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Sabit/Değişken": st.column_config.SelectboxColumn(
                    "Sabit/Değişken",
                    options=["Değişken", "Sabit"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                    help="Bu kalemin yüzde kaçının PDLC ürününe/hatta atanacağı.",
                ),
            },
        )
        st.session_state["downstream_df"] = downstream_df_input

        (
            downstream_df_calc,
            downstream_total_annual,
            downstream_fixed_annual,
            downstream_var_annual,
        ) = calculate_section_costs(
            downstream_df_input,
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )

        st.write("#### Hesaplanmış yıllık maliyetler (Downstream)")
        st.dataframe(
            downstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Downstream (atanmış) toplam yıllık maliyet: {downstream_total_annual:,.2f} USD"
        )

    # --- TAB 4: Destek Birimleri ---
    with tabs[4]:
        st.subheader("Destek Birimleri (Overhead & Genel Giderler)")

        support_df_input = st.data_editor(
            support_df,
            key="support_editor",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Tip": st.column_config.SelectboxColumn(
                    "Tip",
                    options=[
                        "malzeme",
                        "işçilik",
                        "enerji",
                        "bakım",
                        "amortisman",
                        "diğer",
                    ],
                ),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi",
                    options=VALUE_CHAIN_ACTIVITIES,
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_lot",
                        "per_m2",
                        "per_yıl",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Ürün birim miktar kullanımı": st.column_config.NumberColumn(
                    "Ürün birim miktar kullanımı"
                ),
                "Birim maliyet (USD)": st.column_config.NumberColumn(
                    "Birim maliyet (USD)", format="%.4f"
                ),
                "Fire oranı": st.column_config.NumberColumn(
                    "Fire oranı",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Sabit/Değişken": st.column_config.SelectboxColumn(
                    "Sabit/Değişken",
                    options=["Değişken", "Sabit"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                    help="Bu kalemin yüzde kaçının PDLC hattına/ürününe atanacağı.",
                ),
            },
        )
        st.session_state["support_df"] = support_df_input

        (
            support_df_calc,
            support_total_annual,
            support_fixed_annual,
            support_var_annual,
        ) = calculate_section_costs(
            support_df_input,
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=True,
        )

        st.write("#### Hesaplanmış yıllık maliyetler (Destek birimleri)")
        st.dataframe(
            support_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Destek birimleri (atanmış) toplam yıllık maliyet: {support_total_annual:,.2f} USD"
        )

    # --- TAB 5: Özet & Break-even ---
    with tabs[5]:
        st.subheader("Özet Maliyetler ve Break-even Analizi")

        # Toplam yıllık maliyetler
        total_annual_cost = calculate_total_cost(
            upstream_total_annual,
            midstream_total_annual,
            downstream_total_annual,
            support_total_annual,
        )

        # m² başına maliyetler
        if annual_good_m2 > 0:
            upstream_cost_per_m2 = upstream_total_annual / annual_good_m2
            midstream_cost_per_m2 = midstream_total_annual / annual_good_m2
            downstream_cost_per_m2 = downstream_total_annual / annual_good_m2
            support_cost_per_m2 = support_total_annual / annual_good_m2
            total_cost_per_m2 = total_annual_cost / annual_good_m2
        else:
            upstream_cost_per_m2 = midstream_cost_per_m2 = 0.0
            downstream_cost_per_m2 = support_cost_per_m2 = 0.0
            total_cost_per_m2 = 0.0

        # Metrikler
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            "Upstream maliyet / m² (USD)", f"{upstream_cost_per_m2:,.2f}"
        )
        col2.metric(
            "Midstream maliyet / m² (USD)", f"{midstream_cost_per_m2:,.2f}"
        )
        col3.metric(
            "Downstream maliyet / m² (USD)", f"{downstream_cost_per_m2:,.2f}"
        )
        col4.metric(
            "Destek maliyet / m² (USD)", f"{support_cost_per_m2:,.2f}"
        )
        col5.metric("Toplam maliyet / m² (USD)", f"{total_cost_per_m2:,.2f}")

        # Sabit / değişken toplam
        total_fixed_annual = (
            upstream_fixed_annual
            + midstream_fixed_annual
            + downstream_fixed_annual
            + support_fixed_annual
        )
        total_variable_annual = (
            upstream_var_annual
            + midstream_var_annual
            + downstream_var_annual
            + support_var_annual
        )

        if annual_good_m2 > 0:
            variable_cost_per_m2 = total_variable_annual / annual_good_m2
        else:
            variable_cost_per_m2 = 0.0

        st.write("---")
        st.write("### Seviye Bazlı Özet Tablo")

        summary_rows = [
            {
                "Seviye": "Upstream",
                "Yıllık maliyet (USD)": upstream_total_annual,
                "Maliyet / m² (USD)": upstream_cost_per_m2,
            },
            {
                "Seviye": "Midstream",
                "Yıllık maliyet (USD)": midstream_total_annual,
                "Maliyet / m² (USD)": midstream_cost_per_m2,
            },
            {
                "Seviye": "Downstream",
                "Yıllık maliyet (USD)": downstream_total_annual,
                "Maliyet / m² (USD)": downstream_cost_per_m2,
            },
            {
                "Seviye": "Destek birimleri",
                "Yıllık maliyet (USD)": support_total_annual,
                "Maliyet / m² (USD)": support_cost_per_m2,
            },
            {
                "Seviye": "Toplam",
                "Yıllık maliyet (USD)": total_annual_cost,
                "Maliyet / m² (USD)": total_cost_per_m2,
            },
        ]
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.write("---")
        st.write("### Break-even Parametreleri")

        # Satış fiyatı bu sekmede giriliyor (v5'teki gibi)
        default_price = st.session_state.get("selling_price_per_m2", 0.0)
        if default_price <= 0:
            default_price = max(total_cost_per_m2 * 1.2, 50.0)

        selling_price_per_m2 = st.number_input(
            "Satış fiyatı (USD/m²)",
            min_value=0.0,
            value=float(default_price),
            step=1.0,
        )
        st.session_state["selling_price_per_m2"] = selling_price_per_m2

        contribution_margin_per_m2 = (
            selling_price_per_m2 - variable_cost_per_m2
        )

        # Net alan / lot (fire sonrası)
        scrap_each_end_m = st.session_state.get("scrap_each_end_m", 0.0)
        total_scrap_m = 2.0 * scrap_each_end_m
        net_length_m = max(roll_length_m - total_scrap_m, 0.0)
        net_m2_per_lot = net_length_m * roll_width_m

        be_qty = calculate_break_even_point(
            total_fixed_annual, variable_cost_per_m2, selling_price_per_m2
        )
        if be_qty is None:
            break_even_m2 = np.nan
            break_even_lot = np.nan
        else:
            break_even_m2 = be_qty
            break_even_lot = (
                break_even_m2 / net_m2_per_lot if net_m2_per_lot > 0 else np.nan
            )

        col_be1, col_be2, col_be3 = st.columns(3)
        col_be1.metric(
            "Katkı marjı / m² (USD)", f"{contribution_margin_per_m2:,.2f}"
        )
        col_be2.metric(
            "Break-even üretim (m²)",
            "Tanımsız"
            if np.isnan(break_even_m2)
            else f"{break_even_m2:,.0f}",
        )
        col_be3.metric(
            "Break-even lot sayısı",
            "Tanımsız"
            if np.isnan(break_even_lot)
            else f"{break_even_lot:,.1f}",
        )

        st.write("---")
        st.write("### Senaryo Analizi ve Grafik")

        max_default = max(
            annual_good_m2 * 2,
            (break_even_m2 if not np.isnan(break_even_m2) else 0.0) * 2,
            net_m2_per_lot * 10,
            1.0,
        )

        scenario_max_m2 = st.number_input(
            "Senaryodaki maksimum üretim (m²)",
            min_value=0.0,
            value=float(max_default),
            step=max(100.0, max_default / 10.0),
        )

        volumes = np.linspace(0, scenario_max_m2, 50)
        revenues = volumes * selling_price_per_m2
        total_costs = total_fixed_annual + volumes * variable_cost_per_m2
        profits = revenues - total_costs

        chart_df = pd.DataFrame(
            {
                "Üretim hacmi (m²)": volumes,
                "Gelir (USD)": revenues,
                "Toplam maliyet (USD)": total_costs,
                "Kâr/Zarar (USD)": profits,
            }
        ).set_index("Üretim hacmi (m²)")

        st.line_chart(chart_df)

        if not np.isnan(break_even_m2):
            st.caption(
                f"Break-even noktası yaklaşık olarak {break_even_m2:,.0f} m² üretim seviyesindedir."
            )
        else:
            st.caption(
                "Break-even noktası hesaplanamıyor: katkı marjı ≤ 0 veya sabit maliyet = 0."
            )

        st.write("---")
        st.write("### Detaylı maliyet tablosu (CSV export)")

        # Detaylı DF'leri tekrar hesaplayıp export edelim
        (
            upstream_df_calc_full,
            _,
            _,
            _,
        ) = calculate_section_costs(
            st.session_state["upstream_df"],
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )
        (
            midstream_df_calc_full,
            _,
            _,
            _,
        ) = calculate_section_costs(
            st.session_state["midstream_df"],
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )
        (
            downstream_df_calc_full,
            _,
            _,
            _,
        ) = calculate_section_costs(
            st.session_state["downstream_df"],
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=False,
        )
        (
            support_df_calc_full,
            _,
            _,
            _,
        ) = calculate_section_costs(
            st.session_state["support_df"],
            annual_lot_count,
            annual_good_m2,
            lot_duration_hours,
            is_support=True,
        )

        df_cost_detailed, _df_cost_by_activity = aggregate_costs_by_activity(
            upstream_df_calc_full,
            midstream_df_calc_full,
            downstream_df_calc_full,
            support_df_calc_full,
            annual_good_m2,
        )

        st.dataframe(df_cost_detailed, use_container_width=True, hide_index=True)

        csv_bytes = df_cost_detailed.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Detaylı maliyet tablosunu CSV olarak indir",
            csv_bytes,
            file_name="pdlc_cost_detailed.csv",
            mime="text/csv",
        )

        st.write("---")
        st.write("### Maliyet Tipi Bazında Özet (malzeme / işçilik / enerji / bakım / amortisman / diğer)")

        if not df_cost_detailed.empty:
            # Yıllık maliyet kolonu numerik olsun, NaN'ler 0'a çekilsin
            df_tmp = df_cost_detailed.copy()
            df_tmp["Yıllık maliyet (USD)"] = pd.to_numeric(
                df_tmp["Yıllık maliyet (USD)"], errors="coerce"
            ).fillna(0.0)

            # Tip bazında yıllık maliyet toplamları
            df_by_type = (
                df_tmp.groupby("Tip", as_index=False)["Yıllık maliyet (USD)"]
                .sum()
            )

            # Sıfır ve negatif maliyetleri ele
            df_by_type = df_by_type[df_by_type["Yıllık maliyet (USD)"] > 0]

            if not df_by_type.empty:
                df_by_type = df_by_type.sort_values(
                    "Yıllık maliyet (USD)", ascending=False
                )

                # Tablo
                st.dataframe(
                    df_by_type, use_container_width=True, hide_index=True
                )

                # Pie chart
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(
                    df_by_type["Yıllık maliyet (USD)"],
                    labels=df_by_type["Tip"],
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title("Toplam Yıllık Maliyet – Tip Bazında Dağılım")
                ax.axis("equal")  # Tam daire

                st.pyplot(fig)
            else:
                st.info(
                    "Maliyet tipi bazında pozitif yıllık maliyet içeren kayıt bulunamadı."
                )
        else:
            st.info("Maliyet tipi dağılımı için yeterli veri bulunamadı.")

# -------------------------------------------------------------------
# Sayfa: Değer Zinciri + Maliyet Entegrasyonu
# -------------------------------------------------------------------


def render_value_chain_page(df_cost_by_activity: pd.DataFrame):
    """Değer zinciri girdileri + maliyet entegrasyonu sayfası."""

    st.subheader("Değer Zinciri – Porter Aktiviteleri ve Skor Girişi")

    df_vca = st.session_state["vca_df"]

    df_vca_input = st.data_editor(
        df_vca,
        key="vca_editor",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Grup": st.column_config.SelectboxColumn(
                "Grup",
                options=["Primary", "Support"],
            ),
            "Kategori": st.column_config.SelectboxColumn(
                "Değer Zinciri Aktivitesi",
                options=VALUE_CHAIN_ACTIVITIES,
            ),
            "Değer Potansiyeli": st.column_config.NumberColumn(
                "Değer Potansiyeli", min_value=1, max_value=5, step=1
            ),
            "Mevcut Performans": st.column_config.NumberColumn(
                "Mevcut Performans", min_value=1, max_value=5, step=1
            ),
            "Maliyet Etkisi": st.column_config.NumberColumn(
                "Maliyet Etkisi", min_value=1, max_value=5, step=1
            ),
            "Risk / Kayıp": st.column_config.NumberColumn(
                "Risk / Kayıp", min_value=1, max_value=5, step=1
            ),
        },
    )
    st.session_state["vca_df"] = df_vca_input

    df_vc_summary = summarize_value_chain(df_vca_input)

    st.markdown("### Aktivite Bazlı Stratejik Özet")
    st.dataframe(df_vc_summary, use_container_width=True, hide_index=True)

    if df_cost_by_activity is None or df_cost_by_activity.empty:
        st.info(
            "Henüz maliyet hesapları yapılmamış görünüyor. "
            "Önce 'Maliyet & Break-even' sayfasında maliyet girdilerini doldurun."
        )
        return

    df_merge = df_vc_summary.merge(
        df_cost_by_activity[
            [
                "Değer Zinciri Aktivitesi",
                "Yıllık Maliyet (USD)",
                "Toplam Maliyet (USD)",
                "m2 Başına Maliyet (USD/m2)",
                "Sabit Oran",
                "Değişken Oran",
            ]
        ],
        on="Değer Zinciri Aktivitesi",
        how="left",
    ).fillna(0.0)

    st.markdown("### Aktivite Bazlı Stratejik + Maliyet Tablosu")
    st.dataframe(df_merge, use_container_width=True, hide_index=True)

    st.markdown("### Değer Zinciri Isı Haritası")
    plot_value_chain_heatmap(df_merge)


# -------------------------------------------------------------------
# Sayfa: Özet / Dashboard
# -------------------------------------------------------------------


def render_dashboard_page(
    df_vc_summary: pd.DataFrame, df_cost_by_activity: pd.DataFrame
):
    """En yüksek maliyet / stratejik skor ve öncelikli alanların özet dashboard'u."""

    st.subheader("Özet Dashboard – Stratejik + Maliyet Perspektifi")

    if df_cost_by_activity.empty or df_vc_summary.empty:
        st.info(
            "Dashboard için yeterli veri yok. Önce maliyet girdilerini ve değer zinciri skorlarını doldurun."
        )
        return

    df_merge = df_vc_summary.merge(
        df_cost_by_activity,
        on="Değer Zinciri Aktivitesi",
        how="left",
    ).fillna(0.0)

    st.markdown("### En Yüksek Toplam Maliyete Sahip İlk 3 Aktivite")
    top_cost = df_merge.sort_values(
        "Toplam Maliyet (USD)", ascending=False
    ).head(3)
    st.dataframe(top_cost, use_container_width=True, hide_index=True)

    st.markdown("### En Yüksek Stratejik Skora Sahip Aktiviteler")
    top_strategic = df_merge.sort_values(
        "Stratejik_Skor", ascending=False
    ).head(5)
    st.dataframe(top_strategic, use_container_width=True, hide_index=True)

    st.markdown(
        "### Öncelikli İyileştirme Alanları (Yüksek Stratejik Skor + Yüksek Maliyet)"
    )

    df_prior = df_merge.copy()

    def _norm(col: str):
        vals = df_prior[col].astype(float).values
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-9:
            return np.zeros_like(vals)
        return (vals - vmin) / (vmax - vmin)

    df_prior["norm_strategic"] = _norm("Stratejik_Skor")
    df_prior["norm_cost"] = _norm("Toplam Maliyet (USD)")
    df_prior["priority_score"] = df_prior["norm_strategic"] * df_prior["norm_cost"]

    top_priority = df_prior.sort_values("priority_score", ascending=False).head(5)
    st.dataframe(
        top_priority[
            [
                "Değer Zinciri Aktivitesi",
                "Stratejik_Skor",
                "Toplam Maliyet (USD)",
                "m2 Başına Maliyet (USD/m2)",
                "priority_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "> **Yorum:** `priority_score`, normalize edilmiş stratejik skor ile normalize edilmiş "
        "toplam maliyetin çarpımıdır. Sağ üst köşedeki (yüksek strateji × yüksek maliyet) "
        "aktiviteler, tipik olarak *öncelikli iyileştirme* veya *verimlilik* projeleri için "
        "en anlamlı adaylardır."
    )


# -------------------------------------------------------------------
# main()
# -------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="PDLC Değer Zinciri + Maliyet Entegre Arayüzü",
        layout="wide",
    )
    st.title("PDLC Değer Zinciri ve Maliyet Entegrasyonu – Porter + Break-even")

    init_param_defaults()

    if "vca_df" not in st.session_state:
        st.session_state["vca_df"] = get_default_value_chain_df().copy()

    ensure_cost_session_state(downstream_df_init=None)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Genel Parametreler")

        st.subheader("R2R Lot Tanımı")
        roll_length_m = st.number_input(
            "Rulo boyu (m)",
            min_value=1.0,
            value=float(st.session_state["roll_length_m"]),
            step=1.0,
        )
        st.session_state["roll_length_m"] = roll_length_m

        roll_width_m = st.number_input(
            "Rulo genişliği (m)",
            min_value=0.1,
            value=float(st.session_state["roll_width_m"]),
            step=0.1,
        )
        st.session_state["roll_width_m"] = roll_width_m

        scrap_each_end_m = st.number_input(
            "Baş/son fire uzunluğu (m, her bir uç için)",
            min_value=0.0,
            value=float(st.session_state["scrap_each_end_m"]),
            step=0.5,
        )
        st.session_state["scrap_each_end_m"] = scrap_each_end_m

        total_scrap_m = 2.0 * scrap_each_end_m
        net_length_m = max(roll_length_m - total_scrap_m, 0.0)
        gross_area_m2 = roll_length_m * roll_width_m
        net_m2_per_lot = net_length_m * roll_width_m

        st.caption(
            f"Brüt alan ≈ {gross_area_m2:.1f} m², net alan (fire sonrası) ≈ {net_m2_per_lot:.1f} m²"
        )

        st.subheader("Parça / Laminasyon varsayımları")
        part_length_m = st.number_input(
            "Örnek parça uzunluğu (m)",
            min_value=0.1,
            value=float(st.session_state["part_length_m"]),
            step=0.1,
        )
        st.session_state["part_length_m"] = part_length_m

        part_width_m = st.number_input(
            "Örnek parça genişliği (m)",
            min_value=0.1,
            value=float(st.session_state["part_width_m"]),
            step=0.1,
        )
        st.session_state["part_width_m"] = part_width_m

        part_area_m2 = part_length_m * part_width_m

        paste_per_part_g = st.number_input(
            "Örnek parça başına iletken pasta (g)",
            min_value=0.0,
            value=float(st.session_state["paste_per_part_g"]),
            step=0.1,
        )
        st.session_state["paste_per_part_g"] = paste_per_part_g

        foil_length_per_part_m = st.number_input(
            "Örnek parça başına iletken folyo uzunluğu (m)",
            min_value=0.0,
            value=float(st.session_state["foil_length_per_part_m"]),
            step=0.05,
        )
        st.session_state["foil_length_per_part_m"] = foil_length_per_part_m

        cable_length_per_part_m = st.number_input(
            "Örnek parça başına kablo uzunluğu (m)",
            min_value=0.0,
            value=float(st.session_state["cable_length_per_part_m"]),
            step=0.1,
        )
        st.session_state["cable_length_per_part_m"] = cable_length_per_part_m

        paste_per_m2 = (
            paste_per_part_g / part_area_m2 if part_area_m2 > 0 else 0.0
        )
        foil_per_m2 = (
            foil_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
        )
        cable_per_m2 = (
            cable_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
        )

        st.caption(
            f"1 m² PDLC için yaklaşık:\n"
            f"- {paste_per_m2:.3f} g iletken pasta\n"
            f"- {foil_per_m2:.3f} m iletken folyo\n"
            f"- {cable_per_m2:.3f} m kablo"
        )

        st.subheader("Laminasyon varsayımları")
        pvb_m2_per_m2_pdlc = st.number_input(
            "1 m² PDLC başına PVB tüketimi (m²)",
            min_value=0.0,
            value=float(st.session_state["pvb_m2_per_m2_pdlc"]),
            step=0.1,
        )
        st.session_state["pvb_m2_per_m2_pdlc"] = pvb_m2_per_m2_pdlc

        band_m_per_m2_pdlc = st.number_input(
            "1 m² PDLC başına laminasyon bandı (m)",
            min_value=0.0,
            value=float(st.session_state["band_m_per_m2_pdlc"]),
            step=0.05,
        )
        st.session_state["band_m_per_m2_pdlc"] = band_m_per_m2_pdlc

        st.subheader("Zaman / Üretim Hedefi")
        lot_duration_hours = st.number_input(
            "1 lot üretim süresi (saat)",
            min_value=0.1,
            value=float(st.session_state["lot_duration_hours"]),
            step=0.5,
        )
        st.session_state["lot_duration_hours"] = lot_duration_hours

        production_input_mode = st.selectbox(
            "Üretim girişi modu",
            options=["Yıllık üretim alanı (m²)", "Yıllık lot sayısı"],
            index=0
            if st.session_state["production_input_mode"]
            == "Yıllık üretim alanı (m²)"
            else 1,
        )
        st.session_state["production_input_mode"] = production_input_mode

        if production_input_mode == "Yıllık üretim alanı (m²)":
            default_annual_m2 = (
                st.session_state["annual_good_m2"]
                if st.session_state["annual_good_m2"] > 0
                else max(net_m2_per_lot * 100, 1.0)
            )
            annual_good_m2 = st.number_input(
                "Yıllık iyi ürün alanı hedefi (m²)",
                min_value=0.0,
                value=float(default_annual_m2),
                step=float(net_m2_per_lot) if net_m2_per_lot > 0 else 100.0,
            )
            annual_lot_count = (
                annual_good_m2 / net_m2_per_lot if net_m2_per_lot > 0 else 0.0
            )
        else:
            default_annual_lots = (
                st.session_state["annual_lot_count"]
                if st.session_state["annual_lot_count"] > 0
                else 100.0
            )
            annual_lot_count = st.number_input(
                "Yıllık lot sayısı",
                min_value=0.0,
                value=float(default_annual_lots),
                step=1.0,
            )
            annual_good_m2 = annual_lot_count * net_m2_per_lot

        st.session_state["annual_good_m2"] = annual_good_m2
        st.session_state["annual_lot_count"] = annual_lot_count

        st.subheader("Navigasyon")
        page = st.radio(
            "Sayfa seçin",
            [
                "Maliyet & Break-even",
                "Değer Zinciri + Maliyet",
                "Özet / Dashboard",
            ],
        )

        # Konfigürasyon save/load (JSON)
        st.subheader("Konfigürasyon kaydet / yükle")
        with st.expander("Konfigürasyon yönetimi", expanded=False):
            st.markdown(
                "- Geçerli parametreler ve tabloları JSON olarak indirebilir,\n"
                "- Daha önce kaydedilmiş bir JSON'u yükleyerek aynı senaryoya geri dönebilirsin."
            )

            uploaded_cfg = st.file_uploader(
                "Konfigürasyon dosyası (JSON) yükle",
                type=["json"],
            )
            if uploaded_cfg is not None:
                if st.button("Yükle ve uygula", key="load_config_button"):
                    try:
                        cfg = json.load(uploaded_cfg)
                        params = cfg.get("params", {})

                        for p in [
                            "roll_length_m",
                            "roll_width_m",
                            "scrap_each_end_m",
                            "part_length_m",
                            "part_width_m",
                            "paste_per_part_g",
                            "foil_length_per_part_m",
                            "cable_length_per_part_m",
                            "pvb_m2_per_m2_pdlc",
                            "band_m_per_m2_pdlc",
                            "lot_duration_hours",
                            "production_input_mode",
                            "annual_good_m2",
                            "annual_lot_count",
                            "selling_price_per_m2",
                        ]:
                            if p in params:
                                st.session_state[p] = params[p]

                        mapping = {
                            "vca_df": "vca_df",
                            "upstream_df": "upstream_df",
                            "midstream_df": "midstream_df",
                            "downstream_df": "downstream_df",
                            "support_df": "support_df",
                        }
                        for key_cfg, key_state in mapping.items():
                            if key_cfg in cfg:
                                st.session_state[key_state] = pd.DataFrame(
                                    cfg[key_cfg]
                                )

                        st.success(
                            "Konfigürasyon yüklendi. Arayüz güncellendi."
                        )
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Konfigürasyon okunurken hata: {e}")

            cfg_out = {
                "params": {
                    "roll_length_m": st.session_state.get("roll_length_m"),
                    "roll_width_m": st.session_state.get("roll_width_m"),
                    "scrap_each_end_m": st.session_state.get("scrap_each_end_m"),
                    "part_length_m": st.session_state.get("part_length_m"),
                    "part_width_m": st.session_state.get("part_width_m"),
                    "paste_per_part_g": st.session_state.get("paste_per_part_g"),
                    "foil_length_per_part_m": st.session_state.get(
                        "foil_length_per_part_m"
                    ),
                    "cable_length_per_part_m": st.session_state.get(
                        "cable_length_per_part_m"
                    ),
                    "pvb_m2_per_m2_pdlc": st.session_state.get(
                        "pvb_m2_per_m2_pdlc"
                    ),
                    "band_m_per_m2_pdlc": st.session_state.get(
                        "band_m_per_m2_pdlc"
                    ),
                    "lot_duration_hours": st.session_state.get(
                        "lot_duration_hours"
                    ),
                    "production_input_mode": st.session_state.get(
                        "production_input_mode"
                    ),
                    "annual_good_m2": st.session_state.get("annual_good_m2"),
                    "annual_lot_count": st.session_state.get(
                        "annual_lot_count"
                    ),
                    "selling_price_per_m2": st.session_state.get(
                        "selling_price_per_m2"
                    ),
                },
                "vca_df": st.session_state["vca_df"].to_dict(orient="list"),
                "upstream_df": st.session_state["upstream_df"].to_dict(
                    orient="list"
                ),
                "midstream_df": st.session_state["midstream_df"].to_dict(
                    orient="list"
                ),
                "downstream_df": st.session_state["downstream_df"].to_dict(
                    orient="list"
                ),
                "support_df": st.session_state["support_df"].to_dict(
                    orient="list"
                ),
            }
            cfg_json = json.dumps(cfg_out, ensure_ascii=False, indent=2)
            st.download_button(
                "Geçerli konfigürasyonu indir (JSON)",
                data=cfg_json.encode("utf-8"),
                file_name="pdlc_value_chain_config.json",
                mime="application/json",
            )

    # Sidebar sonrası: downstream default tablosu (geometriye göre)
    part_area_m2 = st.session_state["part_length_m"] * st.session_state["part_width_m"]
    downstream_df_init = init_cost_table_downstream(
        st.session_state["paste_per_part_g"],
        st.session_state["foil_length_per_part_m"],
        st.session_state["cable_length_per_part_m"],
        part_area_m2 if part_area_m2 > 0 else 1.0,
        st.session_state["pvb_m2_per_m2_pdlc"],
        st.session_state["band_m_per_m2_pdlc"],
    )
    if st.session_state["downstream_df"].empty:
        st.session_state["downstream_df"] = downstream_df_init.copy()

    upstream_df_base = st.session_state["upstream_df"]
    midstream_df_base = st.session_state["midstream_df"]
    downstream_df_base = st.session_state["downstream_df"]
    support_df_base = st.session_state["support_df"]

    annual_good_m2 = st.session_state["annual_good_m2"]
    annual_lot_count = st.session_state["annual_lot_count"]
    lot_duration_hours = st.session_state["lot_duration_hours"]

    upstream_df_calc_base, up_tot, up_fix, up_var = calculate_section_costs(
        upstream_df_base,
        annual_lot_count,
        annual_good_m2,
        lot_duration_hours,
        is_support=False,
    )
    midstream_df_calc_base, mid_tot, mid_fix, mid_var = calculate_section_costs(
        midstream_df_base,
        annual_lot_count,
        annual_good_m2,
        lot_duration_hours,
        is_support=False,
    )
    downstream_df_calc_base, down_tot, down_fix, down_var = calculate_section_costs(
        downstream_df_base,
        annual_lot_count,
        annual_good_m2,
        lot_duration_hours,
        is_support=False,
    )
    support_df_calc_base, sup_tot, sup_fix, sup_var = calculate_section_costs(
        support_df_base,
        annual_lot_count,
        annual_good_m2,
        lot_duration_hours,
        is_support=True,
    )

    df_cost_detailed, df_cost_by_activity = aggregate_costs_by_activity(
        upstream_df_calc_base,
        midstream_df_calc_base,
        downstream_df_calc_base,
        support_df_calc_base,
        annual_good_m2,
    )

    df_vc_summary = summarize_value_chain(st.session_state["vca_df"])

    if page == "Maliyet & Break-even":
        render_cost_page(
            annual_good_m2,
            annual_lot_count,
            lot_duration_hours,
            st.session_state["roll_length_m"],
            st.session_state["roll_width_m"],
            st.session_state["part_length_m"],
            st.session_state["part_width_m"],
            st.session_state["paste_per_part_g"],
            st.session_state["foil_length_per_part_m"],
            st.session_state["cable_length_per_part_m"],
            st.session_state["pvb_m2_per_m2_pdlc"],
            st.session_state["band_m_per_m2_pdlc"],
        )
    elif page == "Değer Zinciri + Maliyet":
        render_value_chain_page(df_cost_by_activity)
    else:
        render_dashboard_page(df_vc_summary, df_cost_by_activity)


if __name__ == "__main__":
    main()
