# PDLC Değer Zinciri + Maliyet Entegre Arayüzü
# v4 – pdlc_cost_app_v5'in Genel Parametreler + Özet&Break-even yapısı ile uyumlu
# Atama oranı (%) tüm stream'lerde aktif
# v6 – CAPEX (yatırım) modülü eklendi ve break-even analizine entegre edildi.
# v6+ – JSON konfigürasyon kaydet / yükle eklendi (parametreler + tüm maliyet tabloları + CAPEX + VCA)

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
    ["Inbound Logistics", "IQC (Giriş kalite kontrol) testleri", 4, 3, 3, 2],
    ["Operations", "PDLC karışım hazırlanması", 5, 4, 4, 4],
    [
        "Operations",
        "PET film kesimi & yüzey temizliği",
        4,
        4,
        3,
        3,
    ],
    [
        "Operations",
        "R2R kaplama hattı hazırlığı (setup, reçete yükleme)",
        4,
        4,
        4,
        3,
    ],
    ["Operations", "PDLC R2R kaplama", 5, 5, 5, 5],
    ["Operations", "Kuruma / UV kürleme", 4, 4, 4, 4],
    ["Operations", "Rulo muayenesi (görsel, optik, elektriksel)", 4, 4, 4, 3],
    ["Operations", "Rulo ebatlama & parça kesimi", 4, 5, 5, 4],
    ["Operations", "Elektriksel bağlantıların yapılması", 4, 4, 4, 4],
    ["Operations", "Ara kontroller – işlevsel testler", 4, 4, 3, 3],
    ["Operations", "Laminasyon için ara stok yönetimi", 3, 3, 3, 2],
    ["Outbound Logistics", "Son ürün paketleme", 3, 3, 3, 2],
    ["Outbound Logistics", "Sevkiyat planlama", 3, 3, 3, 2],
    ["Outbound Logistics", "Depolama & stok takibi", 3, 3, 3, 2],
    ["Marketing & Sales", "PDLC ürün konumlandırma & segment analizi", 4, 3, 4, 4],
    ["Marketing & Sales", "Müşteri ziyaretleri & demo sunumlar", 3, 3, 4, 4],
    ["Marketing & Sales", "Fiyatlandırma & teklif yönetimi", 4, 3, 4, 4],
    ["Marketing & Sales", "Sözleşme yönetimi", 3, 3, 3, 3],
    ["Service", "Satış sonrası teknik destek", 3, 3, 4, 4],
    ["Service", "Garantili ürün takibi & arıza analizi", 3, 3, 3, 3],
    ["Service", "Saha montaj desteği (müşteriyle birlikte)", 3, 3, 3, 3],
]

DEFAULT_SUPPORT = [
    ["Firm Infrastructure", "Yönetim & strateji", 4, 4, 4, 4],
    ["Firm Infrastructure", "Finans & muhasebe", 3, 3, 3, 3],
    ["Firm Infrastructure", "Hukuk & sözleşme yönetimi", 3, 3, 3, 3],
    ["Firm Infrastructure", "Kalite yönetim sistemi & dokümantasyon", 4, 4, 4, 4],
    ["HR Management", "Mavi yaka işe alım & eğitim", 3, 3, 3, 3],
    ["HR Management", "Beyaz yaka işe alım & eğitim", 3, 3, 3, 3],
    ["HR Management", "Performans değerlendirme & ödüllendirme", 3, 3, 3, 3],
    [
        "Technology Development",
        "Proses geliştirme (yeni reçeteler, verim artışı)",
        5,
        4,
        5,
        5,
    ],
    ["Technology Development", "Yeni ürün geliştirme (otomotiv/arch)", 5, 4, 5, 5],
    ["Technology Development", "Patent & FTO analizleri", 4, 3, 4, 4],
    ["Procurement", "Hammadde tedarik stratejisi", 4, 4, 4, 4],
    ["Procurement", "Tedarikçi sözleşme & fiyat müzakereleri", 4, 4, 4, 4],
    ["Procurement", "Alternatif tedarikçi geliştirme", 4, 4, 4, 4],
]


# -------------------------------------------------------------------
# Maliyet tablolarının defaultları
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
    """Varsayılan midstream (proses) kalemleri."""
    data = {
        "Tip": ["enerji", "işçilik", "enerji"],
        "Kalem": [
            "R2R kaplama hattı elektrik tüketimi",
            "R2R operatör işçilik (2 kişi)",
            "HVAC & temiz oda enerji gideri",
        ],
        "Değer Zinciri Aktivitesi": [
            "Operations",
            "Operations",
            "Operations",
        ],
        "Dağıtım hedefi": ["per_saat", "per_saat", "per_saat"],
        "Birim": ["kWh/saat", "kişi_saat", "kWh/saat"],
        "Ürün birim miktar kullanımı": [0.0, 0.0, 0.0],
        "Birim maliyet (USD)": [0.0, 0.0, 0.0],
        "Fire oranı": [0.0, 0.0, 0.0],
        "Sabit/Değişken": ["Değişken", "Sabit", "Sabit"],
        "Atama oranı (%)": [100.0, 100.0, 100.0],
    }
    return pd.DataFrame(data)


def init_cost_table_support() -> pd.DataFrame:
    """Varsayılan destek birimleri kalemleri."""
    data = {
        "Tip": ["personel", "personel", "personel", "genel_gider"],
        "Kalem": [
            "Mühendislik & Ar-Ge ekibi",
            "Kalite & süreç mühendisleri",
            "Satış & iş geliştirme",
            "Genel yönetim giderleri",
        ],
        "Değer Zinciri Aktivitesi": [
            "Technology Development",
            "Technology Development",
            "Marketing & Sales",
            "Firm Infrastructure",
        ],
        "Dağıtım hedefi": ["per_yil", "per_yil", "per_yil", "per_yil"],
        "Birim": ["yıl", "yıl", "yıl", "yıl"],
        "Ürün birim miktar kullanımı": [1.0, 1.0, 1.0, 1.0],
        "Birim maliyet (USD)": [0.0, 0.0, 0.0, 0.0],
        "Fire oranı": [0.0, 0.0, 0.0, 0.0],
        "Sabit/Değişken": ["Sabit", "Sabit", "Sabit", "Sabit"],
        "Atama oranı (%)": [100.0, 100.0, 100.0, 100.0],
    }
    return pd.DataFrame(data)


def init_cost_table_downstream(
    part_area_m2: float,
    paste_per_part_g: float,
    foil_length_per_part_m: float,
    cable_length_per_part_m: float,
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
        "Tip": [
            "malzeme",
            "malzeme",
            "malzeme",
            "malzeme",
            "malzeme",
        ],
        "Kalem": [
            "PVB ara katman",
            "Busbar / bakır band",
            "İletken pasta",
            "Folyo",
            "Kablo",
        ],
        "Değer Zinciri Aktivitesi": [
            "Operations",
            "Operations",
            "Operations",
            "Operations",
            "Operations",
        ],
        "Dağıtım hedefi": [
            "per_m2",
            "per_m2",
            "per_m2",
            "per_m2",
            "per_m2",
        ],
        "Birim": ["m²", "m", "g", "m", "m"],
        "Ürün birim miktar kullanımı": [
            pvb_m2_per_m2_pdlc,
            band_m_per_m2_pdlc,
            paste_per_m2,
            foil_per_m2,
            cable_per_m2,
        ],
        "Birim maliyet (USD)": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Fire oranı": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Sabit/Değişken": [
            "Değişken",
            "Değişken",
            "Değişken",
            "Değişken",
            "Değişken",
        ],
        "Atama oranı (%)": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------------
# Parametre defaultları
# -------------------------------------------------------------------


def init_param_defaults() -> dict:
    """
    Genel parametreler için başlangıç değerleri.
    """
    return {
        "annual_good_m2": 100_000.0,
        "annual_lot_count": 300.0,
        "lot_duration_hours": 8.0,
        "roll_length_m": 300.0,
        "roll_width_m": 1.6,
        "scrap_each_end_m": 4.0,
        "part_length_m": 1.6,
        "part_width_m": 0.6,
        "paste_per_part_g": 0.5,
        "foil_length_per_part_m": 0.5,
        "cable_length_per_part_m": 0.5,
        "pvb_m2_per_m2_pdlc": 1.0,
        "band_m_per_m2_pdlc": 4.0,
    }


# -------------------------------------------------------------------
# CAPEX tablosu – yeni eklenen bölüm
# -------------------------------------------------------------------


def init_capex_table() -> pd.DataFrame:
    """
    CAPEX (yatırım) kalemleri için varsayılan tablo.

    Kullanıcı burada ekipman, bina, kalıp/tooling vb. yatırımları girer.
    "Yıllık CAPEX (USD)" kolonu arayüzde hesaplanır:
    Toplam yatırım / Ekonomik ömür * (Atama oranı / 100).
    """
    data = {
        "CAPEX kalemi": [
            "R2R kaplama ekipmanı",
            "Laminasyon presi",
        ],
        "Kategori": [
            "Ekipman",
            "Ekipman",
        ],
        "Toplam yatırım (USD)": [
            0.0,
            0.0,
        ],
        "Ekonomik ömür (yıl)": [
            10.0,
            10.0,
        ],
        "Atama oranı (%)": [
            100.0,
            100.0,
        ],
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------------
# VCA tablo defaultları
# -------------------------------------------------------------------


def get_default_value_chain_df() -> pd.DataFrame:
    """
    Porter değer zinciri için varsayılan tablo: Primary + Support aktiviteler.
    Kullanıcı bu tabloyu arayüz üzerinden düzenleyebilir.
    """
    primary = pd.DataFrame(
        DEFAULT_PRIMARY,
        columns=[
            "Aktivite",
            "Alt Aktivite",
            "Stratejik Önem (1-5)",
            "Şirket Yetkinliği (1-5)",
            "Maliyet Etkisi (1-5)",
            "Farklılaştırma Potansiyeli (1-5)",
        ],
    )
    support = pd.DataFrame(
        DEFAULT_SUPPORT,
        columns=[
            "Aktivite",
            "Alt Aktivite",
            "Stratejik Önem (1-5)",
            "Şirket Yetkinliği (1-5)",
            "Maliyet Etkisi (1-5)",
            "Farklılaştırma Potansiyeli (1-5)",
        ],
    )
    return pd.concat([primary, support], ignore_index=True)


# -------------------------------------------------------------------
# Yardımcı fonksiyonlar – maliyet hesapları
# -------------------------------------------------------------------


def ensure_cost_session_state(downstream_df_init: pd.DataFrame | None = None):
    """
    Session state içinde upstream/midstream/downstream/support tabloları yoksa default olarak yaratır.
    Ayrıca CAPEX tablosu da burada initialize edilir.
    """
    if "upstream_df" not in st.session_state:
        st.session_state["upstream_df"] = init_cost_table_upstream().copy()

    if "midstream_df" not in st.session_state:
        st.session_state["midstream_df"] = init_cost_table_midstream().copy()

    if "support_df" not in st.session_state:
        st.session_state["support_df"] = init_cost_table_support().copy()

    # Yeni: CAPEX tablosu
    if "capex_df" not in st.session_state:
        st.session_state["capex_df"] = init_capex_table().copy()

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

    # Eski kaydedilmiş tablolar ile yeni kolonlar arasında uyumsuzluk olmaması için
    for key in ["upstream_df", "midstream_df", "downstream_df", "support_df"]:
        df = st.session_state[key]
        required_cols = [
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
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        # Kolon sırasını normalize edelim
        st.session_state[key] = df[required_cols]


def calculate_section_costs(
    df_input: pd.DataFrame,
    annual_good_m2: float,
    annual_lot_count: float,
    lot_duration_hours: float,
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Bir maliyet tablosu (upstream/midstream/downstream/support) için:
    - Satır bazında yıllık maliyet
    - Toplam yıllık maliyet
    - Sabit / değişken yıllık maliyet ayrımı
    döndürür.
    """
    df = df_input.copy()

    # Eksik kolonlar varsa doldur
    required_cols = [
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
    for c in required_cols:
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

    # Fire oranını 0-1 aralığına çek
    df["Fire oranı"] = df["Fire oranı"].fillna(0.0).clip(lower=0.0, upper=1.0)

    # Atama oranı 0-1 aralığı
    df["Atama oranı (%)"] = (
        df["Atama oranı (%)"].fillna(100.0).clip(lower=0.0, upper=100.0) / 100.0
    )

    # Dağıtım hedefine göre yıllık tüketim katsayıları
    factor = np.zeros(len(df))

    # per_m2: ürün m² üzerinden
    mask_m2 = df["Dağıtım hedefi"] == "per_m2"
    factor[mask_m2] = annual_good_m2

    # per_lot: lot sayısı üzerinden
    mask_lot = df["Dağıtım hedefi"] == "per_lot"
    factor[mask_lot] = annual_lot_count

    # per_yil: zaten yıllık
    mask_year = df["Dağıtım hedefi"] == "per_yil"
    factor[mask_year] = 1.0

    # per_ay: 12 ay
    mask_month = df["Dağıtım hedefi"] == "per_ay"
    factor[mask_month] = 12.0

    # per_saat: toplam hat çalışmasından
    mask_hour = df["Dağıtım hedefi"] == "per_saat"
    factor[mask_hour] = annual_lot_count * lot_duration_hours

    # per_kişi_saat: kişi_saat mantığı ile (örn. operatör sayısı vs.)
    mask_person_hour = df["Dağıtım hedefi"] == "per_kişi_saat"
    factor[mask_person_hour] = annual_lot_count * lot_duration_hours

    # Yıllık miktar = ürün birim miktar kullanımı * factor * (1 + fire)
    df["Yıllık miktar (dağıtılmış)"] = (
        df["Ürün birim miktar kullanımı"].fillna(0.0)
        * factor
        * (1.0 + df["Fire oranı"])
    )

    # Yıllık maliyet (USD) = Yıllık miktar * Birim maliyet * Atama oranı
    df["Yıllık maliyet (USD)"] = (
        df["Yıllık miktar (dağıtılmış)"]
        * df["Birim maliyet (USD)"].fillna(0.0)
        * df["Atama oranı (%)"]
    )

    df["Yıllık maliyet (USD)"] = df["Yıllık maliyet (USD)"].fillna(0.0)

    total_annual = float(df["Yıllık maliyet (USD)"].sum())

    # Sabit / değişken ayrımı
    mask_fixed = df["Sabit/Değişken"].str.lower() == "sabit"
    fixed_annual = float(df.loc[mask_fixed, "Yıllık maliyet (USD)"].sum())
    var_annual = float(df.loc[~mask_fixed, "Yıllık maliyet (USD)"].sum())

    return df, total_annual, fixed_annual, var_annual


def calculate_total_cost(
    upstream_total_annual: float,
    midstream_total_annual: float,
    downstream_total_annual: float,
    support_total_annual: float,
    capex_total_annual: float = 0.0,
) -> float:
    """
    Tüm stream'lerin yıllık toplam (atanmış) maliyetinin hesaplanması.
    CAPEX (yatırım) yıllık yükü de bu toplama isteğe bağlı olarak eklenir.
    """
    return (
        upstream_total_annual
        + midstream_total_annual
        + downstream_total_annual
        + support_total_annual
        + capex_total_annual
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
    Klasik break-even analizi:
    Sabit maliyetler / (Birim fiyat - değişken birim maliyet).
    """
    margin = selling_price_per_m2 - variable_cost_per_m2
    if margin <= 0:
        return None
    return total_fixed_annual / margin


def get_default_upstream_df() -> pd.DataFrame:
    return init_cost_table_upstream()


def get_default_midstream_df() -> pd.DataFrame:
    return init_cost_table_midstream()


def get_default_support_df() -> pd.DataFrame:
    return init_cost_table_support()


# -------------------------------------------------------------------
# Porter VCA hesapları
# -------------------------------------------------------------------


def compute_vca_scores(df_vca: pd.DataFrame) -> pd.DataFrame:
    """
    Değer zinciri tablosundaki skorları normalleştirip, öncelik skoru hesaplar.
    """
    df = df_vca.copy()

    score_cols = [
        "Stratejik Önem (1-5)",
        "Şirket Yetkinliği (1-5)",
        "Maliyet Etkisi (1-5)",
        "Farklılaştırma Potansiyeli (1-5)",
    ]
    for c in score_cols:
        df[c] = df[c].fillna(0.0).astype(float)

    # Basit bir normalizasyon örneği: 1-5'i 0-1'e
    for c in score_cols:
        df[f"norm_{c}"] = (df[c] - 1.0) / 4.0

    # Örnek öncelik metriği: (stratejik önem * maliyet etkisi * farklılaştırma)
    df["priority_score"] = (
        df["norm_Stratejik Önem (1-5)"]
        * df["norm_Maliyet Etkisi (1-5)"]
        * df["norm_Farklılaştırma Potansiyeli (1-5)"]
    )

    return df


def map_costs_to_vca(
    df_cost_upstream: pd.DataFrame,
    df_cost_midstream: pd.DataFrame,
    df_cost_downstream: pd.DataFrame,
    df_cost_support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maliyet tablolarını değer zinciri aktiviteleri (Değer Zinciri Aktivitesi) ile eşler.
    Dönüş:
    - df_cost_detailed: satır bazında tüm maliyetler + aktivite
    - df_cost_by_activity: aktivite bazında toplam yıllık maliyet
    """
    df_all = []
    for df in [
        df_cost_upstream,
        df_cost_midstream,
        df_cost_downstream,
        df_cost_support,
    ]:
        if "Değer Zinciri Aktivitesi" not in df.columns:
            df["Değer Zinciri Aktivitesi"] = ""
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)
    df_all["Yıllık maliyet (USD)"] = df_all["Yıllık maliyet (USD)"].fillna(0.0)

    df_cost_detailed = df_all.copy()

    df_cost_by_activity = (
        df_all.groupby("Değer Zinciri Aktivitesi")["Yıllık maliyet (USD)"]
        .sum()
        .reset_index()
    )

    return df_cost_detailed, df_cost_by_activity


# -------------------------------------------------------------------
# Ana sayfa: Maliyet & Break-even sayfası
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

    # CAPEX yıllık yükü ve m² başına CAPEX için başlangıç değerleri.
    capex_total_annual = 0.0
    capex_cost_per_m2 = 0.0

    tabs = st.tabs(
        [
            "Genel Parametreler",
            "Upstream",
            "Midstream",
            "Downstream",
            "Destek Birimleri",
            "CAPEX / Yatırım",
            "Özet & Break-even",
        ]
    )

    # --- TAB 0: Genel Parametreler ---
    with tabs[0]:
        st.subheader("Genel Proses Özeti")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rulo uzunluğu (m)", f"{roll_length_m:,.1f}")
            st.metric("Rulo genişliği (m)", f"{roll_width_m:,.2f}")
            st.metric("Brüt rulo alanı (m²)", f"{gross_area_m2:,.1f}")

        with col_b:
            st.metric("Fire (her iki uç, m)", f"{total_scrap_m:,.1f}")
            st.metric("Net kullanılabilir uzunluk (m)", f"{net_length_m:,.1f}")
            st.metric("Net m² / lot", f"{net_m2_per_lot:,.1f}")

        with col_c:
            st.metric("Parça boyu (m)", f"{part_length_m:,.2f}")
            st.metric("Parça eni (m)", f"{part_width_m:,.2f}")
            st.metric("Parça alanı (m²)", f"{part_area_m2:,.2f}")

        st.write("---")
        st.write("### Laminasyon & Downstream Parametreleri (m²'ye dönüştürme)")
        st.caption(
            "Bu parametreler downstream default tablosuna da yansıtılır (PVB, busbar, pasta, folyo, kablo kullanımları)."
        )

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            pvb_m2_per_m2_pdlc = st.number_input(
                "1 m² PDLC başına PVB alanı (m²)",
                min_value=0.0,
                value=float(pvb_m2_per_m2_pdlc),
                step=0.1,
                key="pvb_m2_per_m2_pdlc_tab",
            )
            st.session_state["pvb_m2_per_m2_pdlc"] = pvb_m2_per_m2_pdlc

        with col_d2:
            band_m_per_m2_pdlc = st.number_input(
                "1 m² PDLC başına busbar band uzunluğu (m)",
                min_value=0.0,
                value=float(band_m_per_m2_pdlc),
                step=0.1,
                key="band_m_per_m2_pdlc_tab",
            )
            st.session_state["band_m_per_m2_pdlc"] = band_m_per_m2_pdlc

        with col_d3:
            paste_per_part_g = st.number_input(
                "Örnek parça başına iletken pasta (g)",
                min_value=0.0,
                value=float(paste_per_part_g),
                step=0.1,
                key="paste_per_part_g_tab",
            )
            st.session_state["paste_per_part_g"] = paste_per_part_g

        foil_length_per_part_m = st.number_input(
            "Parça başı folyo uzunluğu (m)",
            min_value=0.0,
            value=float(foil_length_per_part_m),
            step=0.05,
            key="foil_length_per_part_m_tab",
        )
        st.session_state["foil_length_per_part_m"] = foil_length_per_part_m

        cable_length_per_part_m = st.number_input(
            "Parça başı kablo uzunluğu (m)",
            min_value=0.0,
            value=float(cable_length_per_part_m),
            step=0.1,
            key="cable_length_per_part_m_tab",
        )
        st.session_state["cable_length_per_part_m"] = cable_length_per_part_m

        paste_per_m2 = paste_per_part_g / part_area_m2 if part_area_m2 > 0 else 0.0
        foil_per_m2 = (
            foil_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
        )
        cable_per_m2 = (
            cable_length_per_part_m / part_area_m2 if part_area_m2 > 0 else 0.0
        )

        st.write("#### 1 m² PDLC için türetilmiş downstream kullanımları")
        col_d4, col_d5, col_d6 = st.columns(3)
        with col_d4:
            st.metric("PVB (m²/m²)", f"{pvb_m2_per_m2_pdlc:,.2f}")
        with col_d5:
            st.metric("Busbar band (m/m²)", f"{band_m_per_m2_pdlc:,.2f}")
        with col_d6:
            st.metric("İletken pasta (g/m²)", f"{paste_per_m2:,.2f}")

        col_d7, col_d8 = st.columns(2)
        with col_d7:
            st.metric("Folyo (m/m²)", f"{foil_per_m2:,.2f}")
        with col_d8:
            st.metric("Kablo (m/m²)", f"{cable_per_m2:,.2f}")

        # Bu parametrelerden downstream default tablosunu yeniden hesaplayalım
        downstream_default = init_cost_table_downstream(
            part_area_m2,
            paste_per_part_g,
            foil_length_per_part_m,
            cable_length_per_part_m,
            pvb_m2_per_m2_pdlc,
            band_m_per_m2_pdlc,
        )
        st.session_state["downstream_default_calc"] = downstream_default

    # --- TAB 1: Upstream ---
    with tabs[1]:
        st.subheader("Upstream Maliyetler (Hammadde & Giriş)")

        upstream_df = st.session_state.get("upstream_df", init_cost_table_upstream())
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
                        "enerji",
                        "işçilik",
                        "bakım",
                        "genel_gider",
                        "personel",
                    ],
                ),
                "Kalem": st.column_config.TextColumn("Kalem"),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi", options=VALUE_CHAIN_ACTIVITIES
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_m2",
                        "per_lot",
                        "per_yil",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Birim": st.column_config.TextColumn("Birim"),
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
                    options=["Sabit", "Değişken"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
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
            upstream_df_input, annual_good_m2, annual_lot_count, lot_duration_hours
        )

        st.write("##### Hesaplanmış Upstream Maliyet Tablosu")
        st.dataframe(
            upstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Upstream (atanmış) toplam yıllık maliyet: {upstream_total_annual:,.2f} USD"
        )

    # --- TAB 2: Midstream ---
    with tabs[2]:
        st.subheader("Midstream Maliyetler (Proses)")

        midstream_df = st.session_state.get(
            "midstream_df", init_cost_table_midstream()
        )
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
                        "enerji",
                        "işçilik",
                        "bakım",
                        "genel_gider",
                        "personel",
                    ],
                ),
                "Kalem": st.column_config.TextColumn("Kalem"),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi", options=VALUE_CHAIN_ACTIVITIES
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_m2",
                        "per_lot",
                        "per_yil",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Birim": st.column_config.TextColumn("Birim"),
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
                    options=["Sabit", "Değişken"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
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
            midstream_df_input, annual_good_m2, annual_lot_count, lot_duration_hours
        )

        st.write("##### Hesaplanmış Midstream Maliyet Tablosu")
        st.dataframe(
            midstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Midstream (atanmış) toplam yıllık maliyet: {midstream_total_annual:,.2f} USD"
        )

    # --- TAB 3: Downstream ---
    with tabs[3]:
        st.subheader("Downstream Maliyetler (Laminasyon & Bağlantı Elemanları)")

        downstream_df = st.session_state.get(
            "downstream_df", st.session_state.get("downstream_default_calc")
        )
        if downstream_df is None or downstream_df.empty:
            downstream_df = st.session_state.get(
                "downstream_default_calc",
                init_cost_table_downstream(
                    part_area_m2,
                    paste_per_part_g,
                    foil_length_per_part_m,
                    cable_length_per_part_m,
                    pvb_m2_per_m2_pdlc,
                    band_m_per_m2_pdlc,
                ),
            )

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
                        "enerji",
                        "işçilik",
                        "bakım",
                        "genel_gider",
                        "personel",
                    ],
                ),
                "Kalem": st.column_config.TextColumn("Kalem"),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi", options=VALUE_CHAIN_ACTIVITIES
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_m2",
                        "per_lot",
                        "per_yil",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Birim": st.column_config.TextColumn("Birim"),
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
                    options=["Sabit", "Değişken"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
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
            downstream_df_input, annual_good_m2, annual_lot_count, lot_duration_hours
        )

        st.write("##### Hesaplanmış Downstream Maliyet Tablosu")
        st.dataframe(
            downstream_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Downstream (atanmış) toplam yıllık maliyet: {downstream_total_annual:,.2f} USD"
        )

    # --- TAB 4: Destek Birimleri ---
    with tabs[4]:
        st.subheader("Destek Birimleri (Overhead & Genel Giderler)")

        support_df = st.session_state.get("support_df", init_cost_table_support())
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
                        "enerji",
                        "işçilik",
                        "bakım",
                        "genel_gider",
                        "personel",
                    ],
                ),
                "Kalem": st.column_config.TextColumn("Kalem"),
                "Değer Zinciri Aktivitesi": st.column_config.SelectboxColumn(
                    "Değer Zinciri Aktivitesi", options=VALUE_CHAIN_ACTIVITIES
                ),
                "Dağıtım hedefi": st.column_config.SelectboxColumn(
                    "Dağıtım hedefi",
                    options=[
                        "per_m2",
                        "per_lot",
                        "per_yil",
                        "per_ay",
                        "per_saat",
                        "per_kişi_saat",
                    ],
                ),
                "Birim": st.column_config.TextColumn("Birim"),
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
                    options=["Sabit", "Değişken"],
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
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
            support_df_input, annual_good_m2, annual_lot_count, lot_duration_hours
        )

        st.write("##### Hesaplanmış Destek Birimleri Maliyet Tablosu")
        st.dataframe(
            support_df_calc, hide_index=True, use_container_width=True
        )
        st.info(
            f"Destek birimleri (atanmış) toplam yıllık maliyet: {support_total_annual:,.2f} USD"
        )

    # --- TAB 5: CAPEX / Yatırım ---
    with tabs[5]:
        st.subheader("CAPEX / Yatırım Kalemleri")

        st.caption(
            "Ekipman, bina, kalıp/tooling vb. yatırımların yıllık CAPEX yükünü hesaplar "
            "ve break-even analizinde sabit maliyete ekler."
        )

        # CAPEX tablosu: kullanıcı girdisi
        capex_df_input = st.data_editor(
            st.session_state["capex_df"],
            key="capex_editor",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "CAPEX kalemi": st.column_config.TextColumn("CAPEX kalemi"),
                "Kategori": st.column_config.SelectboxColumn(
                    "Kategori",
                    options=["Ekipman", "Bina", "Kalıp/Tooling", "Diğer"],
                ),
                "Toplam yatırım (USD)": st.column_config.NumberColumn(
                    "Toplam yatırım (USD)",
                    format="%.2f",
                    min_value=0.0,
                ),
                "Ekonomik ömür (yıl)": st.column_config.NumberColumn(
                    "Ekonomik ömür (yıl)",
                    format="%.1f",
                    min_value=1.0,
                ),
                "Atama oranı (%)": st.column_config.NumberColumn(
                    "Atama oranı (%)",
                    format="%.1f",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )
        # Son girilen tabloyu session_state'e geri yaz
        st.session_state["capex_df"] = capex_df_input

        # CAPEX yıllık yükü hesaplama
        capex_df_calc = capex_df_input.copy()

        # Guard: eksik kolonlar varsa oluştur
        for col in ["Toplam yatırım (USD)", "Ekonomik ömür (yıl)", "Atama oranı (%)"]:
            if col not in capex_df_calc.columns:
                capex_df_calc[col] = 0.0

        capex_df_calc["Toplam yatırım (USD)"] = capex_df_calc[
            "Toplam yatırım (USD)"
        ].fillna(0.0)
        capex_df_calc["Ekonomik ömür (yıl)"] = capex_df_calc[
            "Ekonomik ömür (yıl)"
        ].replace(0, np.nan)
        capex_df_calc["Atama oranı (%)"] = capex_df_calc[
            "Atama oranı (%)"
        ].fillna(100.0)

        # Yıllık CAPEX = yatırım / ömür * atama oranı
        capex_df_calc["Yıllık CAPEX (USD)"] = (
            capex_df_calc["Toplam yatırım (USD)"]
            / capex_df_calc["Ekonomik ömür (yıl)"]
            * capex_df_calc["Atama oranı (%)"]
            / 100.0
        )

        # Toplam yıllık CAPEX yükü
        capex_total_annual = float(
            capex_df_calc["Yıllık CAPEX (USD)"].fillna(0.0).sum()
        )

        if annual_good_m2 > 0:
            capex_cost_per_m2 = capex_total_annual / annual_good_m2
        else:
            capex_cost_per_m2 = 0.0

        st.dataframe(
            capex_df_calc,
            hide_index=True,
            use_container_width=True,
        )

        col_cap1, col_cap2 = st.columns(2)
        col_cap1.metric(
            "Toplam yıllık CAPEX yükü (USD)", f"{capex_total_annual:,.2f}"
        )
        if annual_good_m2 > 0:
            col_cap2.metric(
                "CAPEX yükü / m² (USD)", f"{capex_cost_per_m2:,.2f}"
            )
        else:
            col_cap2.metric("CAPEX yükü / m² (USD)", "Tanımsız")

    # --- TAB 6: Özet & Break-even ---
    with tabs[6]:
        st.subheader("Özet Maliyetler ve Break-even Analizi")

        # Toplam yıllık maliyetler (OPEX + CAPEX yıllık yükü)
        total_annual_cost = calculate_total_cost(
            upstream_total_annual,
            midstream_total_annual,
            downstream_total_annual,
            support_total_annual,
            capex_total_annual,
        )

        # m² başına maliyetler
        if annual_good_m2 > 0:
            upstream_cost_per_m2 = upstream_total_annual / annual_good_m2
            midstream_cost_per_m2 = midstream_total_annual / annual_good_m2
            downstream_cost_per_m2 = downstream_total_annual / annual_good_m2
            support_cost_per_m2 = support_total_annual / annual_good_m2
            capex_cost_per_m2 = capex_total_annual / annual_good_m2
            total_cost_per_m2 = total_annual_cost / annual_good_m2
        else:
            upstream_cost_per_m2 = midstream_cost_per_m2 = 0.0
            downstream_cost_per_m2 = support_cost_per_m2 = 0.0
            capex_cost_per_m2 = 0.0
            total_cost_per_m2 = 0.0

        # Metrikler
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        col5.metric(
            "CAPEX yükü / m² (USD)", f"{capex_cost_per_m2:,.2f}"
        )
        col6.metric("Toplam maliyet / m² (USD)", f"{total_cost_per_m2:,.2f}")

        # Sabit / değişken toplam
        # Not: total_fixed_annual artık OPEX sabit maliyetlerine ek olarak yıllık CAPEX yükünü de içerir.
        total_fixed_annual = (
            upstream_fixed_annual
            + midstream_fixed_annual
            + downstream_fixed_annual
            + support_fixed_annual
            + capex_total_annual
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
                "Seviye": "CAPEX",
                "Yıllık maliyet (USD)": capex_total_annual,
                "Maliyet / m² (USD)": capex_cost_per_m2,
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

        # Satış fiyatı bu sekmede giriliyor
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

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Toplam sabit maliyet (USD/yıl)", f"{total_fixed_annual:,.0f}")
        col_b2.metric("Değişken maliyet / m² (USD)", f"{variable_cost_per_m2:,.2f}")
        col_b3.metric(
            "Katkı marjı / m² (USD)", f"{contribution_margin_per_m2:,.2f}"
        )

        col_be1, col_be2 = st.columns(2)
        if not np.isnan(break_even_m2):
            col_be1.metric("Break-even hacmi (m²/yıl)", f"{break_even_m2:,.0f}")
        else:
            col_be1.metric("Break-even hacmi (m²/yıl)", "Tanımsız")

        if not np.isnan(break_even_lot):
            col_be2.metric("Break-even lot sayısı (yıl)", f"{break_even_lot:,.1f}")
        else:
            col_be2.metric("Break-even lot sayısı (yıl)", "Tanımsız")

        st.write("---")
        st.write("### Kâr / Zarar Eğrisi (Senaryo)")

        max_default = max(
            annual_good_m2 * 1.5,
            break_even_m2 * 1.5 if not np.isnan(break_even_m2) else annual_good_m2 * 1.5,
        )
        scenario_max_m2 = st.number_input(
            "Senaryo grafiği için maksimum üretim (m²/yıl)",
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
                "Break-even noktası hesaplanamıyor: katkı marjı ≤ 0 veya sabit maliyet 0 olabilir."
            )


# -------------------------------------------------------------------
# VCA Sayfası
# -------------------------------------------------------------------


def render_vca_page():
    st.subheader("PDLC Değer Zinciri Analizi (Porter VCA)")

    vca_df = st.session_state.get("vca_df", get_default_value_chain_df())
    vca_df_input = st.data_editor(
        vca_df,
        key="vca_editor",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
    )
    st.session_state["vca_df"] = vca_df_input

    df_vca_scored = compute_vca_scores(vca_df_input)

    st.write("### Skorlanmış Değer Zinciri Tablosu")
    st.dataframe(df_vca_scored, hide_index=True, use_container_width=True)

    st.write("---")
    st.write("### Aktivite Bazlı Öncelik Haritası")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df_vca_scored["norm_Stratejik Önem (1-5)"],
        df_vca_scored["norm_Maliyet Etkisi (1-5)"],
        s=200 * df_vca_scored["priority_score"].fillna(0.0),
        alpha=0.7,
    )
    ax.set_xlabel("Stratejik Önem (normalized)")
    ax.set_ylabel("Maliyet Etkisi (normalized)")
    ax.set_title("Değer Zinciri Aktivite Öncelik Haritası")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.write("---")
    st.write("### Özet: Öncelikli Aktivite Alanları")
    top_activities = (
        df_vca_scored.sort_values("priority_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.dataframe(top_activities, hide_index=True, use_container_width=True)


# -------------------------------------------------------------------
# Ana uygulama
# -------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="PDLC Değer Zinciri + Maliyet & Break-even",
        layout="wide",
    )

    st.title("PDLC Değer Zinciri + Maliyet Entegre Arayüzü")

    # Genel parametreleri session_state'ten yükle veya defaultları al
    params = st.session_state.get("params", init_param_defaults())

    ensure_cost_session_state(downstream_df_init=None)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Genel Parametreler")

        annual_good_m2 = st.number_input(
            "Yıllık iyi ürün miktarı (m²/yıl)",
            min_value=0.0,
            value=float(params.get("annual_good_m2", 100_000.0)),
            step=1_000.0,
        )
        params["annual_good_m2"] = annual_good_m2

        annual_lot_count = st.number_input(
            "Yıllık lot sayısı",
            min_value=0.0,
            value=float(params.get("annual_lot_count", 300.0)),
            step=10.0,
        )
        params["annual_lot_count"] = annual_lot_count

        lot_duration_hours = st.number_input(
            "Bir lot süresi (saat)",
            min_value=0.0,
            value=float(params.get("lot_duration_hours", 8.0)),
            step=1.0,
        )
        params["lot_duration_hours"] = lot_duration_hours

        roll_length_m = st.number_input(
            "Rulo uzunluğu (m)",
            min_value=0.0,
            value=float(params.get("roll_length_m", 300.0)),
            step=10.0,
        )
        params["roll_length_m"] = roll_length_m

        roll_width_m = st.number_input(
            "Rulo genişliği (m)",
            min_value=0.0,
            value=float(params.get("roll_width_m", 1.6)),
            step=0.1,
        )
        params["roll_width_m"] = roll_width_m

        scrap_each_end_m = st.number_input(
            "Her uçta fire (m)",
            min_value=0.0,
            value=float(params.get("scrap_each_end_m", 4.0)),
            step=0.5,
        )
        params["scrap_each_end_m"] = scrap_each_end_m
        st.session_state["scrap_each_end_m"] = scrap_each_end_m

        part_length_m = st.number_input(
            "Parça boyu (m)",
            min_value=0.0,
            value=float(params.get("part_length_m", 1.6)),
            step=0.1,
        )
        params["part_length_m"] = part_length_m

        part_width_m = st.number_input(
            "Parça eni (m)",
            min_value=0.0,
            value=float(params.get("part_width_m", 0.6)),
            step=0.1,
        )
        params["part_width_m"] = part_width_m

        paste_per_part_g = st.number_input(
            "Parça başına iletken pasta (g)",
            min_value=0.0,
            value=float(params.get("paste_per_part_g", 0.5)),
            step=0.1,
        )
        params["paste_per_part_g"] = paste_per_part_g

        foil_length_per_part_m = st.number_input(
            "Parça başı folyo uzunluğu (m)",
            min_value=0.0,
            value=float(params.get("foil_length_per_part_m", 0.5)),
            step=0.05,
        )
        params["foil_length_per_part_m"] = foil_length_per_part_m

        cable_length_per_part_m = st.number_input(
            "Parça başı kablo uzunluğu (m)",
            min_value=0.0,
            value=float(params.get("cable_length_per_part_m", 0.5)),
            step=0.1,
        )
        params["cable_length_per_part_m"] = cable_length_per_part_m

        pvb_m2_per_m2_pdlc = st.number_input(
            "1 m² PDLC başına PVB alanı (m²)",
            min_value=0.0,
            value=float(params.get("pvb_m2_per_m2_pdlc", 1.0)),
            step=0.1,
        )
        params["pvb_m2_per_m2_pdlc"] = pvb_m2_per_m2_pdlc

        band_m_per_m2_pdlc = st.number_input(
            "1 m² PDLC başına busbar band uzunluğu (m)",
            min_value=0.0,
            value=float(params.get("band_m_per_m2_pdlc", 4.0)),
            step=0.1,
        )
        params["band_m_per_m2_pdlc"] = band_m_per_m2_pdlc

        st.session_state["params"] = params

        st.write("---")
        st.write("Aktif sayfa:")

        page = st.radio(
            "Sayfa Seçimi",
            ["Maliyet & Break-even", "Değer Zinciri (Porter VCA)"],
            index=0,
        )

        # ------------------------------------------------------------
        # Konfigürasyon save/load (JSON) – önceki versiyondan geri eklendi
        # Burada:
        #  - Parametreler (geometri, üretim, fiyat)
        #  - Tüm maliyet tabloları (upstream/midstream/downstream/support/CAPEX)
        #  - VCA tablosu
        # tek bir JSON dosyası olarak indirilebilir
        #  - Daha önce kaydedilmiş bir JSON tekrar yüklenip aynı senaryo geri çağrılabilir.
        # ------------------------------------------------------------
        st.subheader("Konfigürasyon kaydet / yükle")
        with st.expander("Konfigürasyon yönetimi", expanded=False):
            st.markdown(
                "- Geçerli parametreler ve tabloları JSON olarak indirebilir,\n"
                "- Daha önce kaydedilmiş bir JSON'u yükleyerek aynı senaryoya geri dönebilirsin."
            )

            # JSON'dan yükleme
            uploaded_cfg = st.file_uploader(
                "Konfigürasyon dosyası (JSON) yükle",
                type=["json"],
                key="config_file_uploader",
            )
            if uploaded_cfg is not None:
                if st.button("Yükle ve uygula", key="load_config_button"):
                    try:
                        cfg = json.load(uploaded_cfg)
                        params_loaded = cfg.get("params", {})

                        # Eski versiyonla uyumlu parametre listesi
                        param_keys = [
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
                            "production_input_mode",  # Yeni kodda kullanılmıyor ama uyumluluk için okunuyor
                            "annual_good_m2",
                            "annual_lot_count",
                            "selling_price_per_m2",
                        ]

                        # Parametreleri session_state'e yaz
                        for p in param_keys:
                            if p in params_loaded:
                                st.session_state[p] = params_loaded[p]

                        # params dict'ini de güncelle
                        ss_params = st.session_state.get("params", {})
                        for p in param_keys:
                            if p in params_loaded:
                                ss_params[p] = params_loaded[p]
                        st.session_state["params"] = ss_params

                        # DataFrame mapping
                        df_mapping = {
                            "vca_df": "vca_df",
                            "upstream_df": "upstream_df",
                            "midstream_df": "midstream_df",
                            "downstream_df": "downstream_df",
                            "support_df": "support_df",
                            "capex_df": "capex_df",  # Yeni: CAPEX tablosu da konfig'e dahil
                        }
                        for key_cfg, key_state in df_mapping.items():
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

            # Geçerli durumu JSON olarak indirme
            param_keys_out = [
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
                "production_input_mode",  # Yeni kodda yok ama eski JSON'larla uyumlu olsun diye alan bırakıyoruz
                "annual_good_m2",
                "annual_lot_count",
                "selling_price_per_m2",
            ]
            params_out = {p: st.session_state.get(p) for p in param_keys_out}

            cfg_out = {
                "params": params_out,
            }

            # DataFrame'leri ekle (orient="list" – eski versiyonla uyumlu)
            def df_to_cfg(key: str, default_df: pd.DataFrame | None = None):
                df = st.session_state.get(key, default_df)
                if df is not None:
                    cfg_out[key] = df.to_dict(orient="list")

            df_to_cfg("vca_df", get_default_value_chain_df())
            df_to_cfg("upstream_df", init_cost_table_upstream())
            df_to_cfg("midstream_df", init_cost_table_midstream())
            # downstream defaultu geometriye bağlı, ama o anki tabloyu kaydetmek daha doğru
            df_to_cfg("downstream_df")
            df_to_cfg("support_df", init_cost_table_support())
            df_to_cfg("capex_df", init_capex_table())

            cfg_json = json.dumps(cfg_out, ensure_ascii=False, indent=2)
            st.download_button(
                "Geçerli konfigürasyonu indir (JSON)",
                data=cfg_json.encode("utf-8"),
                file_name="pdlc_value_chain_config.json",
                mime="application/json",
                key="download_config_button",
            )

    # Ana içerik
    if page == "Maliyet & Break-even":
        render_cost_page(
            annual_good_m2=annual_good_m2,
            annual_lot_count=annual_lot_count,
            lot_duration_hours=lot_duration_hours,
            roll_length_m=roll_length_m,
            roll_width_m=roll_width_m,
            part_length_m=part_length_m,
            part_width_m=part_width_m,
            paste_per_part_g=paste_per_part_g,
            foil_length_per_part_m=foil_length_per_part_m,
            cable_length_per_part_m=cable_length_per_part_m,
            pvb_m2_per_m2_pdlc=pvb_m2_per_m2_pdlc,
            band_m_per_m2_pdlc=band_m_per_m2_pdlc,
        )
    else:
        render_vca_page()


if __name__ == "__main__":
    main()
