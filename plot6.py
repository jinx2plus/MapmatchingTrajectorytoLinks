# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 11:13:03 2025

@author: Yong Jin Park
"""

from pathlib import Path
import geopandas as gpd
import folium
from shapely.geometry import LineString, Point, Polygon
import branca.colormap as cm

import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import os
import numpy as np
import json 
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import box
from shapely.strtree import STRtree
import gc
from pathlib import Path
# import os
from glob import glob
import time
from line_profiler import profile
# from sqlalchemy import create_engine
import psycopg2
import logging, psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

plt.rc('font', family='NanumGothic')

#%%
# 기준 경로 및 검색 패턴 설정
BASE_PATH = Path.cwd()
# MODIFIED: '251011' 폴더를 모든 하위 디렉토리에서 찾도록 패턴 복원
# SEARCH_PATTERN = 
YEAR = 2025

# --- 2. 링크(Links) 데이터 로드 및 전처리 ---
# 이 부분은 이전과 동일하게 유지됩니다.
try:
    logging.info("원본 링크 Shapefile을 로드합니다.")
    # 사용자의 환경에 맞게 실제 파일 경로를 지정해야 합니다.
    links_shp_path = "/data1/DTG/2023_1/JB/JBLINK.shp"
    links = gpd.read_file(links_shp_path, engine='pyogrio', columns=['LINK_ID', "ROAD_RANK", 'geometry'])
    
    logging.info("ROAD_RANK가 '103'인 링크만 필터링합니다.")
    links103_ids = links[links['ROAD_RANK'] == "103"]['LINK_ID'].tolist()
    links_filtered = links[links['LINK_ID'].isin(links103_ids)].copy()
    logging.info(f"총 {len(links_filtered)}개의 '103' 등급 링크가 로드되었습니다.")

except Exception as e:
    logging.error(f"링크 Shapefile 로드 또는 처리 중 오류 발생: {e}")
    links_filtered = None

#%%
# 1. 파일 경로 설정
#해당 파이썬 파일이 있는 하위 폴더를 모두 찾아서 "dtgsum_link_2025_FINAL_RESULT.geoparquet" 파일을 찾아서 모두 합치는 코드로 수정해야함
# input_path = Path(r"????")
# input_path = Path(os.getcwd())
input_path = Path.cwd()
# intermediate_files = glob(os.path.join(input_path, f"dtgsum_link_2025_*.geoparquet"))
# intermediate_files = glob(os.path.join(input_path, f"251011/dtgsum_link_2025_*.geoparquet"))
search_pattern = "dps_RES0012522/output_JB/251004/251011/dtgsum_link_2025_FINAL_RESULT.geoparquet"

intermediate_files = list(input_path.glob(search_pattern))
print(f"'{search_pattern}' 패턴으로 총 {len(intermediate_files)}개의 파일을 찾았습니다.")

print(intermediate_files)
#dtgsum_link_2025_051_002
# intermediate_files = glob(os.path.join(input_path, f"dtgsum_link_2025_FINAL_RESULT.geoparquet"))

if not intermediate_files:
    raise FileNotFoundError(f"지정된 경로에 GeoParquet 파일이 없습니다: {input_path}")

# --- 효율적으로 개선된 데이터 집계 부분 ---
agg_results = []
# geometry 정보를 담아둘 GeoDataFrame (첫 번째 파일로 초기화)
geometry_gdf = gpd.read_parquet(intermediate_files[0])[['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID']).set_index('LINK_ID')

for f in tqdm(intermediate_files, desc="Aggregating chunk files"):
    chunk_df = pd.read_parquet(f, columns=['LINK_ID', 'vehicle_count', 'VLM'])
    current_agg = chunk_df.groupby('LINK_ID')[['vehicle_count', 'VLM']].sum()
    agg_results.append(current_agg)

logging.info("모든 파일 집계를 시작합니다...")
# 모든 집계 결과를 한 번에 합산
final_agg_df = pd.concat(agg_results).groupby('LINK_ID').sum().reset_index()

logging.info("집계 완료. Geometry 정보와 병합을 시작합니다.")
# 집계 결과와 Geometry 정보를 LINK_ID 기준으로 병합
# 이제 final_gdf는 GeoDataFrame이 됩니다.
final_gdf = gpd.GeoDataFrame(final_agg_df.merge(geometry_gdf, on='LINK_ID', how='inner'))

links = gpd.read_file(r"/data1/DTG/2023_1/JB/JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
# links = gpd.read_file(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
links103 = links[links['ROAD_RANK']=="103"]['LINK_ID'].tolist()
links103 = links[links['LINK_ID'].isin(links103)]

# 최종 집계 결과에 링크의 geometry 정보 결합
links_geom = links103[['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID'])
final_merged_gdf = pd.merge(links_geom, final_gdf, on='LINK_ID', how='left')

# 집계되지 않은 링크는 0으로 채우기
final_merged_gdf['vehicle_count'] = final_merged_gdf['vehicle_count'].fillna(0).astype(int)
final_merged_gdf['VLM'] = final_merged_gdf['VLM'].fillna(0).astype(int)
print(final_merged_gdf.columns)

del final_merged_gdf['geometry_y']
final_merged_gdf = final_merged_gdf.rename(columns={'geometry_x': 'geometry'})
print(final_merged_gdf.columns)
final_merged_gdf.set_geometry("geometry")

final_merged_gdf = final_merged_gdf.set_crs(epsg=32652, allow_override=True)
# --- 코드 실행 부분 ---
# 'ratio' 컬럼 계산
final_merged_gdf['ratio'] = np.where(
    final_merged_gdf['VLM'] == 0, 
    0, 
    (final_merged_gdf['vehicle_count'] / final_merged_gdf['VLM']) * 100
)

final_merged_gdf['ratio'] = final_merged_gdf['ratio'].round(1)
print(final_merged_gdf.info())

print(final_merged_gdf.head())
print(final_merged_gdf.tail())

#%%
output_path_excel = os.path.join(input_path, "final_merged_gdf2시간이상.xlsx")
final_merged_gdf.to_excel(output_path_excel, index=False)

output_path_gdf = os.path.join(input_path, "final_merged_gdf2시간이상.geoparquet")
final_merged_gdf.to_parquet(output_path_gdf, engine='pyarrow', compression='snappy', index=False)

#%%
# --- 지도 생성 함수 (수정 없음, 그대로 사용) ---
def create_map(gdf, column_name, output_filename):
    """
    주어진 GeoDataFrame과 컬럼을 사용하여 Folium 지도를 생성하고 HTML 파일로 저장합니다.
    (함수 내용은 원본과 동일)
    """
    try:
        print(f"'{column_name}'에 대한 지도 생성을 시작합니다...")
        gdf_copy = gdf.copy()
        gdf_copy[column_name] = gdf_copy[column_name].fillna(0)

        if gdf_copy.crs is None:
            gdf_copy = gdf_copy.set_crs(epsg=32652, allow_override=True)

        gdf_copy = gdf_copy.to_crs(epsg=4326)
        center = ((gdf_copy.total_bounds[1] + gdf_copy.total_bounds[3]) / 2.0,
                  (gdf_copy.total_bounds[0] + gdf_copy.total_bounds[2]) / 2.0)
        
        m = folium.Map(location=center, zoom_start=11, control_scale=True)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=True
        ).add_to(m)
        
        valid_data = gdf_copy[gdf_copy[column_name] >= 0][column_name]
        vmin = valid_data.min() if not valid_data.empty else 0
        vmax = valid_data.max() if not valid_data.empty else 1

        colormap = cm.LinearColormap(colors=["black", "blue", "red"], vmin=vmin, vmax=vmax, caption=column_name)
        
        def style_fn(feature):
            value = feature['properties'].get(column_name, 0)
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
            
            if value < 0:
                return {'fillOpacity': 0, 'weight': 0}
            else:
                return {'color': colormap(value), 'weight': 3, 'opacity': 0.8}
        
        folium.GeoJson(
            gdf_copy,
            name=column_name,
            style_function=style_fn,
            tooltip=folium.features.GeoJsonTooltip(
                fields=["LINK_ID", column_name], 
                aliases=["LINK_ID:", f"{column_name}:"]
            ),
            overlay=True,
            control=True
        ).add_to(m)
        
        colormap.add_to(m)
        folium.LayerControl().add_to(m)
        
        m.save(output_filename)
        print(f"✅ 성공! 지도를 '{output_filename}' 파일로 저장했습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: '{column_name}' 지도 생성 중 문제가 발생했습니다. ({repr(e)})")


# 각 컬럼에 대한 지도 생성 함수 호출 (이제 GeoDataFrame인 final_gdf를 전달)
create_map(final_merged_gdf, 'vehicle_count', 'map_2시간이상주행차량대수.html')
create_map(final_merged_gdf, 'VLM', 'map_2시간이상전체교통량.html')
create_map(final_merged_gdf, 'ratio', 'map_2시간이상ratio.html')

print("\n🎉 모든 지도 생성이 완료되었습니다!")

#%%
# 1. 파일 경로 설정
#해당 파이썬 파일이 있는 하위 폴더를 모두 찾아서 "dtgsum_link_2025_FINAL_RESULT.geoparquet" 파일을 찾아서 모두 합치는 코드로 수정해야함
# input_path = Path(r"????")
# input_path = Path(os.getcwd())
input_path = Path.cwd()
# intermediate_files = glob(os.path.join(input_path, f"dtgsum_link_2025_*.geoparquet"))
# intermediate_files = glob(os.path.join(input_path, f"251011/dtgsum_link_2025_*.geoparquet"))
# search_pattern = "**/251012/dtgsum_link_2025_*.geoparquet"
search_pattern = "dps_RES0012522/output_JB/251004/251012/dtgsum_link_2025_FINAL_RESULT.geoparquet"

intermediate_files = list(input_path.glob(search_pattern))
print(f"'{search_pattern}' 패턴으로 총 {len(intermediate_files)}개의 파일을 찾았습니다.")

print(intermediate_files)
#dtgsum_link_2025_051_002
# intermediate_files = glob(os.path.join(input_path, f"dtgsum_link_2025_FINAL_RESULT.geoparquet"))

if not intermediate_files:
    raise FileNotFoundError(f"지정된 경로에 GeoParquet 파일이 없습니다: {input_path}")

# --- 효율적으로 개선된 데이터 집계 부분 ---
agg_results = []
# geometry 정보를 담아둘 GeoDataFrame (첫 번째 파일로 초기화)
geometry_gdf = gpd.read_parquet(intermediate_files[0])[['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID']).set_index('LINK_ID')

for f in tqdm(intermediate_files, desc="Aggregating chunk files"):
    chunk_df = pd.read_parquet(f, columns=['LINK_ID', 'vehicle_count', 'VLM'])
    current_agg = chunk_df.groupby('LINK_ID')[['vehicle_count', 'VLM']].sum()
    agg_results.append(current_agg)

logging.info("모든 파일 집계를 시작합니다...")
# 모든 집계 결과를 한 번에 합산
final_agg_df = pd.concat(agg_results).groupby('LINK_ID').sum().reset_index()

logging.info("집계 완료. Geometry 정보와 병합을 시작합니다.")
# 집계 결과와 Geometry 정보를 LINK_ID 기준으로 병합
# 이제 final_gdf는 GeoDataFrame이 됩니다.
final_gdf = gpd.GeoDataFrame(final_agg_df.merge(geometry_gdf, on='LINK_ID', how='inner'))

links = gpd.read_file(r"/data1/DTG/2023_1/JB/JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
# links = gpd.read_file(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
links103 = links[links['ROAD_RANK']=="103"]['LINK_ID'].tolist()
links103 = links[links['LINK_ID'].isin(links103)]

# 최종 집계 결과에 링크의 geometry 정보 결합
links_geom = links103[['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID'])
final_merged_gdf = pd.merge(links_geom, final_gdf, on='LINK_ID', how='left')

# 집계되지 않은 링크는 0으로 채우기
final_merged_gdf['vehicle_count'] = final_merged_gdf['vehicle_count'].fillna(0).astype(int)
final_merged_gdf['VLM'] = final_merged_gdf['VLM'].fillna(0).astype(int)
print(final_merged_gdf.columns)

del final_merged_gdf['geometry_y']
final_merged_gdf = final_merged_gdf.rename(columns={'geometry_x': 'geometry'})
print(final_merged_gdf.columns)
final_merged_gdf.set_geometry("geometry")

final_merged_gdf = final_merged_gdf.set_crs(epsg=32652, allow_override=True)
# --- 코드 실행 부분 ---
# 'ratio' 컬럼 계산
final_merged_gdf['ratio'] = np.where(
    final_merged_gdf['VLM'] == 0, 
    0, 
    (final_merged_gdf['vehicle_count'] / final_merged_gdf['VLM']) * 100
)

final_merged_gdf['ratio'] = final_merged_gdf['ratio'].round(1)
print(final_merged_gdf.info())

print(final_merged_gdf.head())
print(final_merged_gdf.tail())

#%%
output_path_excel = os.path.join(input_path, "final_merged_gdf2시간30분이상.xlsx")
final_merged_gdf.to_excel(output_path_excel, index=False)

output_path_gdf = os.path.join(input_path, "final_merged_gdf2시간30분이상.geoparquet")
final_merged_gdf.to_parquet(output_path_gdf, engine='pyarrow', compression='snappy', index=False)

#%%

# 각 컬럼에 대한 지도 생성 함수 호출 (이제 GeoDataFrame인 final_gdf를 전달)
create_map(final_merged_gdf, 'vehicle_count', 'map_2시간30분이상주행차량대수.html')
create_map(final_merged_gdf, 'VLM', 'map_2시간30분이상_전체교통량.html')
create_map(final_merged_gdf, 'ratio', 'map_2시간30분이상_ratio.html')

print("\n🎉 모든 지도 생성이 완료되었습니다!")
