from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, hour, year, lag, month
from pyspark.sql.window import Window
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import mysql.connector

# ✅ 환경 설정 함수
def setup(output_dir="chart"):
    """
    - 한글 폰트 설정
    - 결과 저장 폴더 생성 (스크립트 파일 위치 기준)
    - Spark 세션 생성 및 설정
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 스크립트 파일의 디렉토리를 기준으로 chart 폴더 경로 설정
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 스크립트 파일 위치
    except NameError:  # Jupyter Notebook 등에서 __file__이 정의되지 않은 경우
        script_dir = os.getcwd()
    
    chart_dir = os.path.join(script_dir, output_dir)
    os.makedirs(chart_dir, exist_ok=True)
    
    # 디렉토리 생성 확인
    if not os.path.exists(chart_dir):
        raise FileNotFoundError(f"Failed to create chart directory: {chart_dir}")
    
    print(f"Chart directory created at: {chart_dir}")
    
    spark = (SparkSession.builder
             .master("local")
             .appName("SentimentAggregation")
             .config("spark.ui.showConsoleProgress", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("INFO")
    return spark, chart_dir

# ✅ 데이터 로드 및 전처리 함수
def load_and_preprocess(spark, file_path, data_dir="../_0_data/_3_predict"):
    """
    - CSV 파일을 읽어와 DataFrame으로 변환
    - 날짜, 시간, 연도 컬럼 추가
    - 공포탐욕지수 계산
    """
    # 스크립트 파일의 디렉토리를 기준으로 데이터 경로 설정
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    base_path = os.path.join(script_dir, data_dir)
    full_path = os.path.join(base_path, file_path)
    print(f"Loading data from: {full_path}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    df = spark.read.option("header", True).option("encoding", "UTF-8").csv(f"file:///{full_path}", inferSchema=True)
    df = df.withColumn("date", to_date(col("time")))
    df = df.withColumn("hour", hour(col("time")))
    df = df.withColumn("year", year(col("date")))
    df = df.withColumn("공포탐욕지수", col("prob_greed") * 100)
    return df

# ✅ 공포탐욕지수 평균 계산 및 저장 함수
def calculate_fear_greed(df, company, chart_dir):
    """
    - 시간대별 평균 공포탐욕지수 계산 및 저장
    - 월간 평균 공포탐욕지수 계산 및 저장
    """
    df_hourly = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    hourly_path = os.path.join(chart_dir, f"{company}_hourly_feargreed_score_bert.csv")
    df_hourly.toPandas().to_csv(hourly_path, index=False, encoding="utf-8-sig")
    print(f"Saved hourly fear-greed score to: {hourly_path}")

    df = df.withColumn("month", col("date").substr(1, 7))
    df_monthly = df.groupBy("month").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_monthly_pandas = df_monthly.toPandas()
    df_monthly_pandas = df_monthly_pandas.dropna(subset=["month"])
    df_monthly_pandas["month"] = df_monthly_pandas["month"].astype(str)
    df_monthly_pandas = df_monthly_pandas.sort_values(by="month").reset_index(drop=True)
    monthly_path = os.path.join(chart_dir, f"{company}_monthly_feargreed_score_bert.csv")
    df_monthly_pandas.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    print(f"Saved monthly fear-greed score to: {monthly_path}")
    return df_monthly_pandas

# ✅ 공포탐욕지수 변화율 계산 함수
def calculate_change_rate(df, company, chart_dir):
    """
    - 시간대별 공포탐욕지수 변화율 계산
    - 결과 CSV 저장
    """
    window_spec = Window.partitionBy("year").orderBy("hour")
    df = df.withColumn("feargreed_diff", col("공포탐욕지수") - lag(col("공포탐욕지수"), 1).over(window_spec))
    df_change_rate = df.groupBy("year", "hour").agg(avg("feargreed_diff").alias("변화율"))
    df_change_rate_pandas = df_change_rate.toPandas()
    df_change_rate_pandas = df_change_rate_pandas.sort_values(by=["year", "hour"]).reset_index(drop=True)
    change_rate_path = os.path.join(chart_dir, f"{company}_feargreed_change_rate.csv")
    df_change_rate_pandas.to_csv(change_rate_path, index=False, encoding="utf-8-sig")
    print(f"Saved change rate to: {change_rate_path}")
    return df_change_rate_pandas

# ✅ 이동 평균 분석 함수 (주석 처리)
# def calculate_moving_average(df_monthly_pandas, company, chart_dir):
#     """
#     - 단기(7일) 및 장기(30일) 이동 평균 계산
#     - 골든크로스 및 데드크로스 감지
#     """
#     df_monthly_pandas["단기_이동평균"] = df_monthly_pandas["평균_공포탐욕지수"].rolling(window=7).mean()
#     df_monthly_pandas["장기_이동평균"] = df_monthly_pandas["평균_공포탐욕지수"].rolling(window=30).mean()
#     moving_avg_path = os.path.join(chart_dir, f"{company}_moving_average.csv")
#     df_monthly_pandas.to_csv(moving_avg_path, index=False, encoding="utf-8-sig")
#     print(f"Saved moving average to: {moving_avg_path}")
#     return df_monthly_pandas

# ✅ 클러스터 분석 함수 (주석 처리 유지)
# def cluster_analysis(df_monthly_pandas, company, chart_dir, n_clusters=3):
#     """
#     - K-Means를 활용한 감성 데이터 클러스터링
#     """
#     df_numeric = df_monthly_pandas.drop(columns=["month"]).dropna()
#
#     # 데이터 개수 확인
#     if len(df_numeric) < n_clusters:
#         print(f"⚠️ {company} 데이터에서 클러스터링을 수행할 수 없습니다 (데이터 개수 부족: {len(df_numeric)}개).")
#         df_monthly_pandas["클러스터"] = np.nan
#         return df_monthly_pandas
#
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(df_numeric)
#
#     # 클러스터 결과를 원본 DataFrame에 병합
#     df_clusters = pd.DataFrame({"클러스터": clusters}, index=df_numeric.index)
#     df_monthly_pandas = pd.concat([df_monthly_pandas, df_clusters], axis=1)
#
#     cluster_path = os.path.join(chart_dir, f"{company}_cluster_analysis.csv")
#     df_monthly_pandas.to_csv(cluster_path, index=False, encoding="utf-8-sig")
#     print(f"Saved cluster analysis to: {cluster_path}")
#     return df_monthly_pandas

# ✅ 시각화 함수
def save_plots(df_change_rate_pandas, df_monthly_pandas, company, chart_dir):
    """
    - 공포탐욕지수 변화율 및 월간 평균 그래프 저장
    """
    print(f"Chart directory: {chart_dir}")
    years = df_change_rate_pandas["year"].unique()

    for year in years:
        df_yearly = df_change_rate_pandas[df_change_rate_pandas["year"] == year]
        plt.figure(figsize=(12, 6))
        plt.plot(df_yearly["hour"], df_yearly["변화율"], marker='o', linestyle='-', color='red', label=f'{year}년 공포탐욕지수 변화율')
        plt.axhline(0, color='gray', linestyle='--', label='기준선')
        plt.title(f"{company} {year}년 시간대별 공포탐욕지수 변화율")
        plt.xlabel("시간")
        plt.ylabel("변화율")
        plt.legend()
        plt.grid(True)
        change_rate_plot = os.path.join(chart_dir, f"{company}_{year}_fear_and_greed_change_rate.png")
        plt.savefig(change_rate_plot)
        plt.close()
        print(f"Saved change rate plot to: {change_rate_plot}")

    print(f"--- {company} 월간 데이터 확인 ---")
    print(df_monthly_pandas)
    print("------------------------------------")

    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly_pandas["month"], df_monthly_pandas["평균_공포탐욕지수"], marker='o', linestyle='-', color='blue', label='월간 평균 공포탐욕지수')
    plt.title(f"{company} 월간 평균 공포탐욕지수")
    plt.xlabel("월")
    plt.ylabel("평균 공포탐욕지수")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    monthly_plot = os.path.join(chart_dir, f"{company}_monthly_fear_and_greed.png")
    plt.savefig(monthly_plot)
    plt.close()
    print(f"Saved monthly plot to: {monthly_plot}")

# MySQL에 데이터 삽입하는 함수
def insert_to_mysql(cursor, company, df_hourly_pandas, df_monthly_pandas, df_change_rate_pandas):
    def replace_nan(val):
        return None if pd.isna(val) else val

    df_hourly_pandas_cleaned = df_hourly_pandas.dropna(subset=['year', 'hour'])
    for index, row in df_hourly_pandas_cleaned.iterrows():
        sql = "INSERT INTO hourly_feargreed_bert (company, year, hour, 평균_공포탐욕지수) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s"
        val = (company, int(row['year']), int(row['hour']), replace_nan(row['평균_공포탐욕지수']), replace_nan(row['평균_공포탐욕지수']))
        cursor.execute(sql, val)

    for index, row in df_monthly_pandas.iterrows():
        sql = "INSERT INTO monthly_feargreed_bert (company, month, 평균_공포탐욕지수) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE 평균_공포탐욕지수=%s"
        val = (company, row['month'], replace_nan(row['평균_공포탐욕지수']), replace_nan(row['평균_공포탐욕지수']))
        cursor.execute(sql, val)

    df_change_rate_pandas_cleaned = df_change_rate_pandas.dropna(subset=['year', 'hour'])
    for index, row in df_change_rate_pandas_cleaned.iterrows():
        sql = "INSERT INTO feargreed_change_rate (company, year, hour, 변화율) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 변화율=%s"
        val = (company, int(row['year']), int(row['hour']), replace_nan(row['변화율']), replace_nan(row['변화율']))
        cursor.execute(sql, val)

# ✅ 실행 코드
if __name__ == "__main__":
    spark, chart_dir = setup(output_dir="chart")

    # MySQL 연결 설정
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Qlsl0316',
        'database': 'stock_analysis'
    }

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        print("MySQL 데이터베이스 연결 성공")
        companies = ["samsung", "apple", "nvidia", "skhynix"]
        for company in companies:
            print(f"--- 현재 처리 중인 회사: {company} ---")
            try:
                file_path = f"{company}_predict_bert.csv"
                df = load_and_preprocess(spark, file_path, data_dir="../_0_data/_3_predict")
                df_monthly_pandas = calculate_fear_greed(df, company, chart_dir)
                df_change_rate_pandas = calculate_change_rate(df, company, chart_dir)
                # df_monthly_pandas = calculate_moving_average(df_monthly_pandas, company, chart_dir)
                # df_monthly_pandas = cluster_analysis(df_monthly_pandas, company, chart_dir)
                print(f"About to call save_plots for {company}")
                save_plots(df_change_rate_pandas, df_monthly_pandas, company, chart_dir)

                df_hourly_pandas = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수")).toPandas()

                insert_to_mysql(cursor, company, df_hourly_pandas, df_monthly_pandas, df_change_rate_pandas)
                conn.commit()
                print(f"{company} 데이터 MySQL 저장 완료")
            except Exception as e:
                print(f"{company} 처리 중 오류 발생: {e}")
                conn.rollback()
                continue
    except mysql.connector.Error as err:
        print(f"MySQL 오류 발생: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL 연결 종료")

    spark.stop()