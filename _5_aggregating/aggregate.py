import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path

# ✅ argparse 추가: 원하는 기업 선택 가능
parser = argparse.ArgumentParser()
parser.add_argument(
    "--stock",
    type=str,
    choices=["samsung", "apple", "nvidia", "skhynix"],
    default="skhynix",  # ✅ 기본값 추가
    help="분석할 회사 이름을 선택하세요: samsung, apple, nvidia, skhynix"
)
args = parser.parse_args()
company = args.stock  # 선택한 기업 저장

def setup(output_dir="data/chart"):
    """
    📌 차트를 저장할 디렉토리 생성
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    root_dir = Path.cwd()
    chart_dir = root_dir / output_dir
    chart_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Chart directory created at: {chart_dir}")
    return chart_dir

def upload_to_hdfs(local_path, hdfs_path):
    """
    📌 HDFS에 파일 업로드
    """
    try:
        local_path = Path(local_path).as_posix()  # ✅ 경로를 HDFS에 적합한 형식으로 변환
        hdfs_path = Path(hdfs_path).as_posix()
        
        # 기존 파일 삭제 후 업로드
        subprocess.run(f"hdfs dfs -rm -f {hdfs_path}", shell=True, check=False)
        subprocess.run(f"hdfs dfs -put {local_path} {hdfs_path}", shell=True, check=True)
        print(f"✅ Uploaded {local_path} to HDFS: {hdfs_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ HDFS Upload Failed: {e}")

def load_and_preprocess(file_path, data_dir="_0_data/_3_predict"):
    """
    📌 데이터 로드 및 전처리
    """
    root_dir = os.getcwd()
    base_path = os.path.join(root_dir, data_dir)
    full_path = os.path.join(base_path, file_path)

    if not os.path.exists(full_path):
        print(f"❌ Data file not found: {full_path}")
        exit(1)

    print(f"📥 Loading data from: {full_path}")
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

def save_plots(df_change_rate, df_monthly, company, chart_dir):
    years = df_change_rate["year"].unique()
    for year in years:
        df_yearly = df_change_rate[df_change_rate["year"] == year]
        plt.figure(figsize=(12, 6))
        plt.plot(df_yearly["hour"], df_yearly["feargreed_diff"], marker='o', linestyle='-', color='red')
        plt.axhline(0, color='gray', linestyle='--')
        plt.grid(True)
        plt.subplots_adjust(bottom=0.15)
        change_rate_plot = os.path.join(chart_dir, f"{company}_{year}_fear_and_greed_change_rate.png")
        plt.savefig(change_rate_plot)
        plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly["month"], df_monthly["공포탐욕지수"], marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
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
