import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from sklearn.cluster import KMeans
import numpy as np

# ✅ argparse 추가: 원하는 기업 선택 가능
parser = argparse.ArgumentParser()
parser.add_argument(
    "--stock",
    type=str,
    choices=["samsung", "apple", "nvidia", "skhynix"],
    default="skhynix",  # ✅ 기본값 추가 (선택 사항)
    help="분석할 회사 이름을 선택하세요: samsung, apple, nvidia, skhynix"
)

try:
    args = parser.parse_args()
    company = args.stock  # 선택한 기업 저장
except SystemExit:
    print("❌ 오류: --stock 인자가 필요합니다. 실행 예시: python aggregate.py --stock samsung")
    exit(1)

def setup(output_dir="_5_aggregating/chart"):
    """
    📌 차트를 저장할 디렉토리 생성
    - 경로를 현재 작업 디렉토리를 기준으로 설정
    - 디렉토리가 없으면 자동 생성
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    root_dir = os.getcwd()
    chart_dir = os.path.join(root_dir, output_dir)
    os.makedirs(chart_dir, exist_ok=True)
    print(f"📂 Chart directory created at: {chart_dir}")
    return chart_dir

def upload_to_hdfs(local_path, hdfs_path):
    """
    📌 HDFS에 파일 업로드
    - subprocess를 이용하여 hdfs dfs -put 실행
    """
    try:
        cmd = f"hdfs dfs -put -f {local_path} {hdfs_path}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Uploaded {local_path} to HDFS: {hdfs_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ HDFS Upload Failed: {e}")

def load_and_preprocess(file_path, data_dir="_0_data/_3_predict"):
    """
    📌 데이터 로드 및 전처리
    - 지정된 경로에서 CSV 파일을 불러옴
    - 날짜 및 시간 관련 컬럼 추가 (year, month, hour)
    - 공포-탐욕 지수(%) 계산
    """
    root_dir = os.getcwd()
    base_path = os.path.join(root_dir, data_dir)
    full_path = os.path.join(base_path, file_path)

    print(f"📥 Loading data from: {full_path}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Data file not found: {full_path}")

    df = pd.read_csv(full_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m')
    df["공포탐욕지수"] = df["prob_greed"] * 100
    return df

def calculate_fear_greed(df, company, chart_dir, hdfs_dir):
    df_hourly = df.groupby(["year", "hour"]).agg({"공포탐욕지수": "mean"}).reset_index()
    hourly_path = os.path.join(chart_dir, f"{company}_hourly_feargreed_score_bert.csv")
    df_hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    upload_to_hdfs(hourly_path, os.path.join(hdfs_dir, f"{company}_hourly_feargreed_score_bert.csv"))

    df_monthly = df.groupby("month").agg({"공포탐욕지수": "mean"}).reset_index()
    monthly_path = os.path.join(chart_dir, f"{company}_monthly_feargreed_score_bert.csv")
    df_monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    upload_to_hdfs(monthly_path, os.path.join(hdfs_dir, f"{company}_monthly_feargreed_score_bert.csv"))

    return df_monthly

def calculate_change_rate(df, company, chart_dir, hdfs_dir):
    df["feargreed_diff"] = df.groupby("year")["공포탐욕지수"].diff()
    df_change_rate = df.groupby(["year", "hour"]).agg({"feargreed_diff": "mean"}).reset_index()
    change_rate_path = os.path.join(chart_dir, f"{company}_feargreed_change_rate.csv")
    df_change_rate.to_csv(change_rate_path, index=False, encoding="utf-8-sig")
    upload_to_hdfs(change_rate_path, os.path.join(hdfs_dir, f"{company}_feargreed_change_rate.csv"))

    return df_change_rate

# ✅ 이동 평균 분석 (주석 유지)
# def calculate_moving_average(df_monthly, company, chart_dir):
#     df_monthly["단기_이동평균"] = df_monthly["공포탐욕지수"].rolling(window=7).mean()
#     df_monthly["장기_이동평균"] = df_monthly["공포탐욕지수"].rolling(window=30).mean()
#     moving_avg_path = os.path.join(chart_dir, f"{company}_moving_average.csv")
#     df_monthly.to_csv(moving_avg_path, index=False, encoding="utf-8-sig")
#     return df_monthly

# ✅ 클러스터 분석 (주석 유지)
# def cluster_analysis(df_monthly, company, chart_dir, n_clusters=3):
#     df_numeric = df_monthly.drop(columns=["month"]).dropna()
#     if len(df_numeric) < n_clusters:
#         print(f"⚠️ 클러스터링 불가능 (데이터 부족: {len(df_numeric)}개).")
#         df_monthly["클러스터"] = np.nan
#         return df_monthly

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_numeric)
    df_clusters = pd.DataFrame({"클러스터": clusters}, index=df_numeric.index)
    df_monthly = pd.concat([df_monthly, df_clusters], axis=1)

    cluster_path = os.path.join(chart_dir, f"{company}_cluster_analysis.csv")
    df_monthly.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    return df_monthly

def save_plots(df_change_rate, df_monthly, company, chart_dir):
    years = df_change_rate["year"].unique()
    for year in years:
        df_yearly = df_change_rate[df_change_rate["year"] == year]
        plt.figure(figsize=(12, 6))
        plt.plot(df_yearly["hour"], df_yearly["feargreed_diff"], marker='o', linestyle='-', color='red')
        plt.axhline(0, color='gray', linestyle='--')
        plt.grid(True)
        change_rate_plot = os.path.join(chart_dir, f"{company}_{year}_fear_and_greed_change_rate.png")
        plt.savefig(change_rate_plot)
        plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly["month"], df_monthly["공포탐욕지수"], marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    monthly_plot = os.path.join(chart_dir, f"{company}_monthly_fear_and_greed.png")
    plt.savefig(monthly_plot)
    plt.close()

def main():
    chart_dir = setup()
    file_path = f"{company}_predict_bert.csv"
    hdfs_dir = f"/user/hdfs/{company}/"

    df = load_and_preprocess(file_path)
    df_monthly = calculate_fear_greed(df, company, chart_dir, hdfs_dir)
    df_change_rate = calculate_change_rate(df, company, chart_dir, hdfs_dir)
    save_plots(df_change_rate, df_monthly, company, chart_dir)

if __name__ == "__main__":
    main()
