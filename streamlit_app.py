import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import streamlit as st

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    layout="wide"
)


REQUIRED_COLUMNS = {
    "defect_id",
    "product_id",
    "defect_type",
    "defect_date",
    "defect_location",
    "severity",
    "inspection_method",
    "repair_cost",
}


@st.cache_data
def load_and_process_data(file_path: str):
    data = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Kolom berikut tidak ditemukan pada dataset: "
            + ", ".join(sorted(missing_columns))
        )

    data = data.copy()
    data["defect_date"] = pd.to_datetime(
        data["defect_date"],
        errors="coerce"
    )

    severity_mapping = {
        "Minor": 1,
        "Moderate": 2,
        "Critical": 3
    }
    data["severity_score"] = data["severity"].map(severity_mapping)

    if data["severity_score"].isna().any():
        invalid_values = data.loc[
            data["severity_score"].isna(), "severity"
        ].dropna().unique()

        raise ValueError(
            "Terdapat nilai severity yang tidak dikenali: "
            + ", ".join(map(str, invalid_values))
        )

    features = data[["repair_cost", "severity_score"]]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    wcss = []
    for k in range(1, 11):
        elbow_model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            n_init=10
        )
        elbow_model.fit(features_scaled)
        wcss.append(elbow_model.inertia_)

    kmeans_model = KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    )
    data["cluster_kmeans"] = kmeans_model.fit_predict(features_scaled)

    hierarchy_model = AgglomerativeClustering(
        n_clusters=3,
        metric="euclidean",
        linkage="ward"
    )
    data["cluster_hierarchy"] = hierarchy_model.fit_predict(features_scaled)

    silhouette_kmeans = silhouette_score(
        features_scaled,
        data["cluster_kmeans"]
    )
    silhouette_hierarchy = silhouette_score(
        features_scaled,
        data["cluster_hierarchy"]
    )
    agreement_score = adjusted_rand_score(
        data["cluster_kmeans"],
        data["cluster_hierarchy"]
    )

    linkage_matrix = sch.linkage(features_scaled, method="ward")

    return (
        data,
        features_scaled,
        wcss,
        silhouette_kmeans,
        silhouette_hierarchy,
        agreement_score,
        linkage_matrix,
    )


def most_frequent(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else "-"


def create_cluster_profile(data: pd.DataFrame, cluster_column: str):
    profile = (
        data.groupby(cluster_column)
        .agg(
            jumlah_data=("defect_id", "count"),
            rata_rata_biaya=("repair_cost", "mean"),
            median_biaya=("repair_cost", "median"),
            rata_rata_severity=("severity_score", "mean"),
            severity_dominan=("severity", most_frequent),
            jenis_cacat_dominan=("defect_type", most_frequent),
            lokasi_dominan=("defect_location", most_frequent),
            metode_inspeksi_dominan=("inspection_method", most_frequent),
        )
        .reset_index()
    )

    profile["persentase_data"] = (
        profile["jumlah_data"] / len(data) * 100
    )

    profile = profile.rename(
        columns={
            cluster_column: "cluster",
            "jumlah_data": "Jumlah Data",
            "persentase_data": "Persentase (%)",
            "rata_rata_biaya": "Rata-rata Biaya",
            "median_biaya": "Median Biaya",
            "rata_rata_severity": "Rata-rata Severity",
            "severity_dominan": "Severity Dominan",
            "jenis_cacat_dominan": "Jenis Cacat Dominan",
            "lokasi_dominan": "Lokasi Dominan",
            "metode_inspeksi_dominan": "Metode Inspeksi Dominan",
        }
    )

    return profile


def cost_category(value: float, all_costs: pd.Series):
    lower = all_costs.quantile(0.33)
    upper = all_costs.quantile(0.67)

    if value >= upper:
        return "tinggi"
    if value <= lower:
        return "rendah"
    return "menengah"


def severity_category(value: float):
    if value >= 2.5:
        return "kritis"
    if value >= 1.75:
        return "menengah"
    return "relatif rendah"


def cluster_interpretation(profile_row: pd.Series, all_costs: pd.Series):
    cluster = int(profile_row["cluster"])
    cost_level = cost_category(
        profile_row["Rata-rata Biaya"],
        all_costs
    )
    severity_level = severity_category(
        profile_row["Rata-rata Severity"]
    )

    return (
        f"Cluster {cluster} berisi {int(profile_row['Jumlah Data'])} kasus "
        f"({profile_row['Persentase (%)']:.1f}% dari seluruh data). "
        f"Rata-rata biaya perbaikannya sebesar "
        f"${profile_row['Rata-rata Biaya']:,.2f}, sehingga termasuk kategori "
        f"biaya {cost_level}. Tingkat keparahannya {severity_level}, dengan "
        f"severity dominan {profile_row['Severity Dominan']}. Jenis cacat yang "
        f"paling sering muncul adalah {profile_row['Jenis Cacat Dominan']}, "
        f"sedangkan lokasi cacat dominannya adalah "
        f"{profile_row['Lokasi Dominan']}."
    )


def business_recommendation(profile_row: pd.Series, all_costs: pd.Series):
    cost_level = cost_category(
        profile_row["Rata-rata Biaya"],
        all_costs
    )
    severity_level = severity_category(
        profile_row["Rata-rata Severity"]
    )

    if cost_level == "tinggi" and severity_level == "kritis":
        return (
            "Prioritas utama. Perusahaan perlu melakukan investigasi akar "
            "masalah, memperketat inspeksi, serta mengevaluasi mesin dan "
            "proses produksi yang berkaitan dengan kelompok ini."
        )

    if cost_level == "tinggi":
        return (
            "Prioritas finansial. Fokuskan tindakan pencegahan pada penyebab "
            "biaya perbaikan yang tinggi agar kerugian dapat ditekan."
        )

    if severity_level == "kritis":
        return (
            "Prioritas keselamatan dan kualitas. Walaupun biaya rata-ratanya "
            "tidak selalu paling tinggi, cacat kritis perlu dicegah agar tidak "
            "sampai ke konsumen."
        )

    if cost_level == "rendah" and severity_level == "relatif rendah":
        return (
            "Dapat ditangani melalui inspeksi rutin dan perbaikan proses "
            "bertahap karena dampak biaya serta tingkat keparahannya lebih rendah."
        )

    return (
        "Perlu pemantauan berkala dan evaluasi proses pada jenis serta lokasi "
        "cacat yang dominan agar kelompok ini tidak berkembang menjadi risiko "
        "yang lebih besar."
    )


def plot_scatter(data: pd.DataFrame, cluster_column: str, title: str):
    figure, axis = plt.subplots(figsize=(10, 6))

    scatter = axis.scatter(
        data["repair_cost"],
        data["severity_score"],
        c=data[cluster_column],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

    axis.set_title(title)
    axis.set_xlabel("Biaya Perbaikan Produk ($)")
    axis.set_ylabel("Tingkat Keparahan Cacat")
    axis.set_yticks([1, 2, 3])
    axis.set_yticklabels(
        ["1 (Minor)", "2 (Moderate)", "3 (Critical)"]
    )
    axis.grid(alpha=0.25)

    legend = axis.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best"
    )
    axis.add_artist(legend)
    figure.tight_layout()

    return figure


st.title("Analisis Clustering Cacat Produk pada Industri Manufaktur")

st.subheader("Identitas Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    st.write("**Nama:** Herlinda Stavia Yuliani")
    st.write("**NIM:** E12.2024.01979")

with col2:
    st.write("**Kelas:** E12402")
    st.write("**Mata Kuliah:** Project Kecerdasan Buatan")

st.divider()

st.write(
    "Aplikasi ini menyajikan hasil segmentasi data cacat manufaktur menggunakan "
    "K-Means dan Hierarchical Clustering. Pengelompokan dilakukan berdasarkan "
    "biaya perbaikan dan tingkat keparahan cacat."
)

try:
    (
        df,
        x_scaled,
        wcss,
        silhouette_kmeans,
        silhouette_hierarchy,
        agreement_score,
        linkage_matrix,
    ) = load_and_process_data("defects_data.csv")
except FileNotFoundError:
    st.error(
        "File defects_data.csv tidak ditemukan. Pastikan file tersebut berada "
        "di folder utama repository GitHub."
    )
    st.stop()
except Exception as error:
    st.error(f"Dataset tidak dapat diproses: {error}")
    st.stop()


profile_kmeans = create_cluster_profile(
    df,
    "cluster_kmeans"
)
profile_hierarchy = create_cluster_profile(
    df,
    "cluster_hierarchy"
)


tab_ringkasan, tab_data, tab_kmeans, tab_hierarchy, tab_insight = st.tabs(
    [
        "Ringkasan",
        "Pemahaman Data",
        "K-Means",
        "Hierarchical Clustering",
        "Perbandingan dan Insight",
    ]
)


with tab_ringkasan:
    st.header("Ringkasan Analisis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data", f"{len(df):,}")
    col2.metric("Jumlah Variabel Awal", "8")
    col3.metric(
        "Rata-rata Biaya Perbaikan",
        f"${df['repair_cost'].mean():,.2f}"
    )
    col4.metric(
        "Kasus Critical",
        f"{(df['severity'] == 'Critical').sum():,}"
    )

    st.subheader("Tujuan Analisis")
    st.write(
        "Tujuan analisis adalah mengelompokkan kasus cacat berdasarkan "
        "kemiripan biaya perbaikan dan tingkat keparahan. Hasilnya dapat "
        "digunakan untuk menentukan kelompok cacat yang perlu diprioritaskan "
        "dalam kegiatan pengendalian kualitas."
    )

    st.subheader("Metode yang Digunakan")
    st.markdown(
        """
        1. Severity diubah menjadi nilai ordinal: Minor = 1, Moderate = 2,
           dan Critical = 3.
        2. Fitur `repair_cost` dan `severity_score` distandardisasi dengan
           StandardScaler.
        3. Jumlah cluster ditetapkan sebanyak 3 berdasarkan Elbow Method dan
           interpretasi dendrogram.
        4. Model yang dibandingkan adalah K-Means dan Agglomerative
           Hierarchical Clustering dengan metode Ward.
        """
    )

    st.subheader("Kesimpulan Awal")
    better_model = (
        "K-Means"
        if silhouette_kmeans >= silhouette_hierarchy
        else "Hierarchical Clustering"
    )

    st.write(
        f"Berdasarkan Silhouette Score, model dengan pemisahan cluster yang "
        f"lebih baik pada dataset ini adalah **{better_model}**. Penilaian "
        f"akhir tetap perlu mempertimbangkan kemudahan interpretasi dan tujuan "
        f"bisnis, bukan hanya satu metrik."
    )


with tab_data:
    st.header("Pemahaman Data")

    st.subheader("Contoh Dataset")
    st.dataframe(
        df[
            [
                "defect_id",
                "product_id",
                "defect_type",
                "defect_date",
                "defect_location",
                "severity",
                "inspection_method",
                "repair_cost",
            ]
        ].head(20),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Severity")
        severity_counts = (
            df["severity"]
            .value_counts()
            .reindex(["Minor", "Moderate", "Critical"])
            .fillna(0)
        )
        st.bar_chart(severity_counts)

    with col2:
        st.subheader("Distribusi Jenis Cacat")
        defect_counts = df["defect_type"].value_counts()
        st.bar_chart(defect_counts)

    st.subheader("Statistik Biaya Perbaikan")
    cost_stats = df["repair_cost"].describe().to_frame(
        name="repair_cost"
    )
    st.dataframe(cost_stats, use_container_width=True)

    figure_cost, axis_cost = plt.subplots(figsize=(10, 5))
    axis_cost.hist(df["repair_cost"], bins=25, edgecolor="black")
    axis_cost.set_title("Distribusi Biaya Perbaikan")
    axis_cost.set_xlabel("Biaya Perbaikan ($)")
    axis_cost.set_ylabel("Frekuensi")
    axis_cost.grid(axis="y", alpha=0.25)
    figure_cost.tight_layout()
    st.pyplot(figure_cost)
    plt.close(figure_cost)

    st.subheader("Kualitas Data")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Nilai Kosong",
        int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    )
    col2.metric(
        "Data Duplikat",
        int(
            df[
                [
                    "defect_id",
                    "product_id",
                    "defect_type",
                    "defect_date",
                    "defect_location",
                    "severity",
                    "inspection_method",
                    "repair_cost",
                ]
            ].duplicated().sum()
        )
    )
    col3.metric(
        "Rentang Tanggal",
        f"{df['defect_date'].min().date()} - "
        f"{df['defect_date'].max().date()}"
    )


with tab_kmeans:
    st.header("Hasil K-Means Clustering")

    st.write(
        "K-Means membagi data ke dalam kelompok berdasarkan kedekatan setiap "
        "titik terhadap centroid. Jumlah cluster ditentukan sebanyak 3."
    )

    st.subheader("Elbow Method")
    figure_elbow, axis_elbow = plt.subplots(figsize=(9, 5))
    axis_elbow.plot(
        range(1, 11),
        wcss,
        marker="o",
        linestyle="--"
    )
    axis_elbow.axvline(
        3,
        linestyle=":",
        label="K yang digunakan = 3"
    )
    axis_elbow.set_title("Elbow Method")
    axis_elbow.set_xlabel("Jumlah Cluster (K)")
    axis_elbow.set_ylabel("WCSS / Inersia")
    axis_elbow.set_xticks(range(1, 11))
    axis_elbow.grid(alpha=0.3)
    axis_elbow.legend()
    figure_elbow.tight_layout()
    st.pyplot(figure_elbow)
    plt.close(figure_elbow)

    st.caption(
        "Titik siku terlihat di sekitar K = 3. Setelah titik tersebut, "
        "penurunan WCSS mulai melambat."
    )

    st.subheader("Visualisasi Cluster")
    figure_kmeans = plot_scatter(
        df,
        "cluster_kmeans",
        "Hasil Segmentasi Cacat Menggunakan K-Means"
    )
    st.pyplot(figure_kmeans)
    plt.close(figure_kmeans)

    st.subheader("Profil Cluster K-Means")
    displayed_kmeans = profile_kmeans.copy()
    displayed_kmeans["Rata-rata Biaya"] = displayed_kmeans[
        "Rata-rata Biaya"
    ].map(lambda value: f"${value:,.2f}")
    displayed_kmeans["Median Biaya"] = displayed_kmeans[
        "Median Biaya"
    ].map(lambda value: f"${value:,.2f}")
    displayed_kmeans["Persentase (%)"] = displayed_kmeans[
        "Persentase (%)"
    ].round(1)
    displayed_kmeans["Rata-rata Severity"] = displayed_kmeans[
        "Rata-rata Severity"
    ].round(2)

    st.dataframe(
        displayed_kmeans,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Interpretasi K-Means")
    for _, row in profile_kmeans.iterrows():
        st.markdown(
            f"**Cluster {int(row['cluster'])}**"
        )
        st.write(
            cluster_interpretation(row, df["repair_cost"])
        )
        st.write(
            "Rekomendasi: "
            + business_recommendation(row, df["repair_cost"])
        )


with tab_hierarchy:
    st.header("Hasil Hierarchical Clustering")

    st.write(
        "Agglomerative Hierarchical Clustering menggabungkan titik atau "
        "kelompok secara bertahap berdasarkan jarak. Metode linkage yang "
        "digunakan adalah Ward."
    )

    st.subheader("Dendrogram")
    figure_dendrogram, axis_dendrogram = plt.subplots(
        figsize=(12, 6)
    )
    sch.dendrogram(
        linkage_matrix,
        ax=axis_dendrogram,
        no_labels=True
    )
    axis_dendrogram.set_title(
        "Dendrogram Cacat Manufaktur dengan Metode Ward"
    )
    axis_dendrogram.set_xlabel("Sampel Data")
    axis_dendrogram.set_ylabel("Jarak Penggabungan")
    figure_dendrogram.tight_layout()
    st.pyplot(figure_dendrogram)
    plt.close(figure_dendrogram)

    st.caption(
        "Pemisahan cabang utama pada dendrogram mendukung penggunaan "
        "3 cluster untuk model Agglomerative."
    )

    st.subheader("Visualisasi Cluster")
    figure_hierarchy = plot_scatter(
        df,
        "cluster_hierarchy",
        "Hasil Segmentasi Cacat Menggunakan Hierarchical Clustering"
    )
    st.pyplot(figure_hierarchy)
    plt.close(figure_hierarchy)

    st.subheader("Profil Cluster Hierarchical")
    displayed_hierarchy = profile_hierarchy.copy()
    displayed_hierarchy["Rata-rata Biaya"] = displayed_hierarchy[
        "Rata-rata Biaya"
    ].map(lambda value: f"${value:,.2f}")
    displayed_hierarchy["Median Biaya"] = displayed_hierarchy[
        "Median Biaya"
    ].map(lambda value: f"${value:,.2f}")
    displayed_hierarchy["Persentase (%)"] = displayed_hierarchy[
        "Persentase (%)"
    ].round(1)
    displayed_hierarchy["Rata-rata Severity"] = displayed_hierarchy[
        "Rata-rata Severity"
    ].round(2)

    st.dataframe(
        displayed_hierarchy,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Interpretasi Hierarchical Clustering")
    for _, row in profile_hierarchy.iterrows():
        st.markdown(
            f"**Cluster {int(row['cluster'])}**"
        )
        st.write(
            cluster_interpretation(row, df["repair_cost"])
        )
        st.write(
            "Rekomendasi: "
            + business_recommendation(row, df["repair_cost"])
        )


with tab_insight:
    st.header("Perbandingan Model dan Insight Bisnis")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Silhouette K-Means",
        f"{silhouette_kmeans:.3f}"
    )
    col2.metric(
        "Silhouette Hierarchical",
        f"{silhouette_hierarchy:.3f}"
    )
    col3.metric(
        "Adjusted Rand Index",
        f"{agreement_score:.3f}"
    )

    st.subheader("Interpretasi Metrik")
    st.write(
        "Silhouette Score menunjukkan seberapa baik suatu data berada di "
        "dalam clusternya dibandingkan dengan cluster lain. Nilai yang lebih "
        "tinggi menunjukkan pemisahan cluster yang lebih jelas. Adjusted Rand "
        "Index mengukur tingkat kesesuaian pengelompokan antara kedua model."
    )

    if silhouette_kmeans > silhouette_hierarchy:
        st.success(
            "K-Means menghasilkan Silhouette Score yang lebih tinggi, sehingga "
            "struktur clusternya lebih terpisah pada dua fitur yang digunakan."
        )
    elif silhouette_hierarchy > silhouette_kmeans:
        st.success(
            "Hierarchical Clustering menghasilkan Silhouette Score yang lebih "
            "tinggi, sehingga struktur clusternya lebih terpisah pada dua fitur "
            "yang digunakan."
        )
    else:
        st.info(
            "Kedua model menghasilkan Silhouette Score yang sama."
        )

    st.subheader("Insight Bisnis Utama")

    highest_cost_kmeans = profile_kmeans.loc[
        profile_kmeans["Rata-rata Biaya"].idxmax()
    ]
    highest_severity_kmeans = profile_kmeans.loc[
        profile_kmeans["Rata-rata Severity"].idxmax()
    ]
    largest_kmeans = profile_kmeans.loc[
        profile_kmeans["Jumlah Data"].idxmax()
    ]

    st.markdown(
        f"""
        1. **Risiko biaya tertinggi** terdapat pada Cluster
           {int(highest_cost_kmeans['cluster'])} K-Means, dengan rata-rata
           biaya perbaikan sekitar
           **${highest_cost_kmeans['Rata-rata Biaya']:,.2f}**.
        2. **Keparahan tertinggi** terdapat pada Cluster
           {int(highest_severity_kmeans['cluster'])} K-Means, dengan rata-rata
           severity sebesar
           **{highest_severity_kmeans['Rata-rata Severity']:.2f}**.
        3. **Kelompok dengan kasus terbanyak** adalah Cluster
           {int(largest_kmeans['cluster'])}, yaitu
           **{int(largest_kmeans['Jumlah Data'])} kasus** atau sekitar
           **{largest_kmeans['Persentase (%)']:.1f}%** dari seluruh data.
        """
    )

    st.subheader("Rekomendasi Manajerial")
    st.markdown(
        """
        - Kelompok dengan biaya perbaikan tertinggi perlu diprioritaskan untuk
          analisis akar penyebab dan program pencegahan cacat.
        - Kelompok dengan severity kritis perlu mendapatkan kontrol kualitas
          yang lebih ketat meskipun biaya rata-ratanya bukan yang tertinggi.
        - Jenis dan lokasi cacat dominan pada setiap cluster dapat dijadikan
          dasar untuk menentukan area proses produksi yang perlu dievaluasi.
        - K-Means lebih mudah digunakan untuk segmentasi operasional berulang,
          sedangkan dendrogram Hierarchical membantu menjelaskan hubungan
          kedekatan antar-data secara visual.
        """
    )

    st.subheader("Unduh Hasil Clustering")
    output_columns = [
        "defect_id",
        "product_id",
        "defect_type",
        "defect_date",
        "defect_location",
        "severity",
        "inspection_method",
        "repair_cost",
        "severity_score",
        "cluster_kmeans",
        "cluster_hierarchy",
    ]

    download_data = df[output_columns].copy()
    download_data["defect_date"] = download_data[
        "defect_date"
    ].dt.strftime("%Y-%m-%d")

    st.download_button(
        label="Unduh data hasil clustering",
        data=download_data.to_csv(index=False).encode("utf-8"),
        file_name="hasil_clustering_cacat_manufaktur.csv",
        mime="text/csv",
    )
