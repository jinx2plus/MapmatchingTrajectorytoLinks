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

import geopandas as gpd
from sqlalchemy import create_engine
import psycopg2

def fetch_geodata(
    table_name,
    geometry_column="geom",
    crs=None,
    dbname="tams",
    host="***.***.***.***",
    port="5432",
    user="postgres",
    password=""
):
    """
    Fetch geospatial data from a PostgreSQL/PostGIS database into a GeoDataFrame.
    
    Returns:
    - GeoDataFrame or None if an error occurs
    """
    # Establish database connection
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print(f"Connected to database '{dbname}'.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
    
    # Create the connection string for SQLAlchemy
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    
    # Query to fetch data
    query = f'SELECT * FROM public."{table_name}"'
    
    # Fetch data into a GeoDataFrame
    try:
        gdf = gpd.read_postgis(query, engine, geom_col=geometry_column)
        print(f"Data fetched successfully from table '{table_name}'. Rows: {len(gdf)}")
        
        # Optionally set the CRS
        if crs:
            gdf = gdf.set_crs(crs, allow_override=True)
    except Exception as e:
        print(f"Error fetching data from table '{table_name}': {e}")
        gdf = None
    finally:
        # Close the database connection
        conn.close()
        engine.dispose()
        print(f"Database connection closed for table '{table_name}'.")
    
    return gdf

def cdmatch_IKSAN(IC):
    def uppercase_cols_except_geom(gdf):
        """지오메트리 열을 제외한 모든 열 이름을 대문자로 변환"""
        # 현재 활성화된 지오메트리 열 이름 가져오기
        geometry_col = gdf.geometry.name  # 이 방법이 가장 안전함
        
        gdf.columns = [col if col == geometry_col else col.upper() for col in gdf.columns]
        return gdf
    
    EMD_IKSAN = fetch_geodata("EMD_IKSAN")
    original_ic_upper_cols = [col.upper() for col in IC.columns if col != IC.geometry.name]
    ##############################################################################################################################
    # IMSI = gpd.sjoin(IC.to_crs(epsg=32652), EMD_IKSAN[['ADM_NM','CITY','geom']], how="left",predicate='intersects')
    emd_proj = EMD_IKSAN.copy()
    emdgeom = EMD_IKSAN.geometry.name
    columns_to_delete = ['CITY', 'DONGNAME', 'DONG_NAME']

    for col in columns_to_delete:
        if col in IC.columns:
            print(f"Found and deleted column: {col}")
            IC = IC.drop(columns=[col])
        
    ic_proj = IC.copy()
    # icgeom = IC.geometry.name

    ic_proj['temp_id_ic'] = range(len(ic_proj))
    
    intersections = gpd.overlay(
        ic_proj,
        emd_proj[['ADM_NM', 'CITY', emdgeom]], # Select only the columns you need from EMD_IKSAN
        how='intersection',
        keep_geom_type=True # Allows for changes in geometry type if needed
    )
    
    intersections['overlap_area'] = intersections.geometry.area
    
    idx = intersections.groupby('temp_id_ic')['overlap_area'].idxmax()
    
    best_matches = intersections.loc[idx]
    final_result = ic_proj.merge(
        best_matches[['temp_id_ic', 'ADM_NM', 'CITY']],
        on='temp_id_ic',
        how='left'
    )
    final_result = final_result.drop(columns='temp_id_ic')
    
    if 'CHUNG' in final_result.columns:
        final_result = final_result.drop(columns='CHUNG')
        
    ##############################################################################################################################
    # final_result['CHUNG'] = 'IKSAN'
    # IMSI['CITY'] = IMSI['CITY_right']
    # IMSI['DONG_NAME'] = IMSI['ADM_NM']
    # del IMSI['CITY_right'], IMSI['ADM_NM'], IMSI['CITY_left']
    # IMSI2 = uppercase_cols_except_geom(final_result)
    # print("IMSI2.colsss",IMSI2.columns)

    # print(IMSI2.columns)
    # 최종적으로 필요한 열 목록을 생성
    # 원본 IC의 열 순서를 최대한 유지하고, 새로 추가된 열을 뒤에 붙임
    final_cols_order = original_ic_upper_cols + ['ADM_NM', 'CITY', 'CHUNG']
    
    # IMSI2에 실제로 존재하는 열들만 final_cols_order 순서에 맞게 필터링
    existing_cols = [col for col in final_cols_order if col in final_result.columns]
    
    # geometry 열 이름 가져오기
    geom_col_name = final_result.geometry.name
    
    # geometry 열이 existing_cols에 없으면 추가
    if geom_col_name not in existing_cols:
        existing_cols.append(geom_col_name)
    
    # 필터링된 열 목록으로 데이터프레임 재구성
    final_result= final_result[existing_cols]
    
    # 열 이름 변경
    if 'ADM_NM' in final_result.columns:
        final_result= final_result.rename(columns={'ADM_NM': 'DONG_NAME'})
        
    ## ==================== 수정된 부분 종료 ==================== ##
    if 'INDEX_RIGHT' in final_result.columns:
        final_result= final_result.drop(columns='INDEX_RIGHT')
    # 최종 열 목록 출력
    final_result= final_result.loc[:, ~final_result.columns.duplicated(keep='last')]
    
    if "DONGNAME" in final_result.columns:
        final_result = final_result.rename(columns={"DONGNAME": 'DONG_NAME'})
    
    print("Final Columns:", final_result.columns.to_list())
    return final_result

pp = links.columns
print(pp)

links103_ = cdmatch_IKSAN(links103)
print(links.info())
print(links.head())

# links103_ = links103_[links103_['CITY']==None]
links103_ = links103_[links103_['CITY'].isna()]
# links = links[links['CITY'].isna()]
print(links103_.info())

links103_ = links103_[pp]

# 최종 집계 결과에 링크의 geometry 정보 결합
links_geom = links103_ [['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID'])
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

final_merged_gdf = final_merged_gdf.loc[final_merged_gdf['ratio']<90]
final_merged_gdf = final_merged_gdf.sort_values("vehicle_count",ascending=False)
final_merged_gdf = final_merged_gdf.iloc[:10]
final_merged_gdf.reset_index(inplace=True)

print(final_merged_gdf.head())


#%%
output_path_excel = os.path.join(input_path, "final_merged_gdf2시간이상.xlsx")
final_merged_gdf.to_excel(output_path_excel, index=False)

output_path_gdf = os.path.join(input_path, "final_merged_gdf2시간이상.geoparquet")
final_merged_gdf.to_parquet(output_path_gdf, engine='pyarrow', compression='snappy', index=False)

#%%
# # --- 지도 생성 함수 (수정 없음, 그대로 사용) ---
# def create_map(gdf, column_name, output_filename):
#     """
#     주어진 GeoDataFrame과 컬럼을 사용하여 Folium 지도를 생성하고 HTML 파일로 저장합니다.
#     (함수 내용은 원본과 동일)
#     """
#     try:
#         print(f"'{column_name}'에 대한 지도 생성을 시작합니다...")
#         gdf_copy = gdf.copy()
#         gdf_copy[column_name] = gdf_copy[column_name].fillna(0)

#         if gdf_copy.crs is None:
#             gdf_copy = gdf_copy.set_crs(epsg=32652, allow_override=True)

#         gdf_copy = gdf_copy.to_crs(epsg=4326)
#         center = ((gdf_copy.total_bounds[1] + gdf_copy.total_bounds[3]) / 2.0,
#                   (gdf_copy.total_bounds[0] + gdf_copy.total_bounds[2]) / 2.0)
        
#         m = folium.Map(location=center, zoom_start=11, control_scale=True)

#         folium.TileLayer(
#             tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#             attr="Esri",
#             name="Esri Satellite",
#             overlay=False,
#             control=True
#         ).add_to(m)
        
#         valid_data = gdf_copy[gdf_copy[column_name] >= 0][column_name]
#         vmin = valid_data.min() if not valid_data.empty else 0
#         vmax = valid_data.max() if not valid_data.empty else 1

#         colormap = cm.LinearColormap(colors=["black", "blue", "red"], vmin=vmin, vmax=vmax, caption=column_name)
        
#         def style_fn(feature):
#             value = feature['properties'].get(column_name, 0)
#             try:
#                 value = float(value)
#             except (ValueError, TypeError):
#                 value = 0.0
            
#             if value < 0:
#                 return {'fillOpacity': 0, 'weight': 0}
#             else:
#                 return {'color': colormap(value), 'weight': 3, 'opacity': 0.8}
        
#         folium.GeoJson(
#             gdf_copy,
#             name=column_name,
#             style_function=style_fn,
#             tooltip=folium.features.GeoJsonTooltip(
#                 fields=["LINK_ID", column_name], 
#                 aliases=["LINK_ID:", f"{column_name}:"]
#             ),
#             overlay=True,
#             control=True
#         ).add_to(m)
        
#         colormap.add_to(m)
#         folium.LayerControl().add_to(m)
        
#         m.save(output_filename)
#         print(f"✅ 성공! 지도를 '{output_filename}' 파일로 저장했습니다.")

#     except Exception as e:
#         print(f"❌ 오류 발생: '{column_name}' 지도 생성 중 문제가 발생했습니다. ({repr(e)})")

#%% 251013 새로운 범주!
import folium
import branca.colormap as cm
import mapclassify  # Natural Breaks 계산을 위해 임포트

# --- 지도 생성 함수 (커스텀 HTML 범례만 포함) ---
def create_map(gdf, column_name, output_filename, k=6):
    """
    주어진 GeoDataFrame과 컬럼을 사용하여 Folium 지도를 생성하고 HTML 파일로 저장합니다.
    (자연 구분법, 커스텀 범례 표 기능 포함)
    """
    try:
        print(f"'{column_name}'에 대한 지도 생성을 시작합니다 (Natural Breaks, k={k})...")
        gdf_copy = gdf.copy()
        
        valid_data = gdf_copy[gdf_copy[column_name] > 0][column_name]

        if valid_data.empty or len(valid_data.unique()) < k:
            print(f"⚠️ 경고: '{column_name}'에 대한 유효 데이터가 부족하여 분류를 수행할 수 없습니다.")
            return

        classifier = mapclassify.NaturalBreaks(valid_data, k=k)
        bins = classifier.bins
        
        # 범례 데이터프레임 및 HTML 테이블 생성
        gradient_generator = cm.LinearColormap(
            colors=['black', 'blue', 'red']
        )
        
        # 2. 위에서 만든 그래디언트에서 k개의 단계별 색상을 추출합니다.
        #    이렇게 하면 k값이 바뀌어도 자동으로 그 개수에 맞는 색상 리스트가 생성됩니다.
        colors = [gradient_generator.rgb_hex_str(i / (k - 1)) for i in range(k)]
        
        legend_data = []
        lower_bound = valid_data.min()
        
        for i, upper_bound in enumerate(bins):
            color = colors[i % len(colors)]
            range_str = f"{int(lower_bound):,} ~ {int(upper_bound):,}"
            
            legend_data.append({
                '범주': f'Class {i+1}',
                '범위': range_str,
                '색상': f'<div style="background-color:{color}; width:25px; height:18px; border:1px solid grey;"></div>'
            })
            lower_bound = upper_bound + 1
            
        legend_df = pd.DataFrame(legend_data)

        # 지도 시각화 좌표계 변환 및 중심 계산
        if gdf_copy.crs is None: gdf_copy = gdf_copy.set_crs(epsg=32652, allow_override=True)
        gdf_copy = gdf_copy.to_crs(epsg=4326)
        center = ((gdf_copy.total_bounds[1] + gdf_copy.total_bounds[3]) / 2.0, (gdf_copy.total_bounds[0] + gdf_copy.total_bounds[2]) / 2.0)
        
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Esri Satellite",overlay=True,control=True).add_to(m)
        
        # folium.TileLayer(
        #     tiles='https://map.pstatic.net/nrb/styles/basic/1723793741/{z}/{x}/{y}.png?mt=bg.ol.ts.l.l',
        #     attr='NAVER',
        #     name='Naver Map(일반)',
        #     overlay=False, # False로 설정하여 기본 지도로 사용
        #     control=True
        # ).add_to(m)
        
        # 범주형 색상맵 생성 (스타일 함수에서 사용)
        colormap = cm.StepColormap(colors=colors[:k], index=[valid_data.min()] + list(bins), vmin=valid_data.min(), vmax=valid_data.max())
        
        # 스타일 함수
        def style_fn(feature):
            value = feature['properties'].get(column_name, 0)
            try: value = float(value)
            except (ValueError, TypeError): value = 0.0
            if value <= 0: return {'fillOpacity': 0, 'weight': 0}
            else: return {'color': colormap(value), 'weight': 3, 'opacity': 0.8}
        
        folium.GeoJson(gdf_copy, name=column_name, style_function=style_fn, tooltip=folium.features.GeoJsonTooltip(fields=["LINK_ID", column_name], aliases=["LINK_ID:", f"{column_name}:"])).add_to(m)
        
        # 커스텀 HTML 범례를 지도에 추가
        legend_html_table = legend_df.to_html(border=0, classes='table table-sm', index=False, escape=False)
        
        legend_html = f'''
             <div style="position: fixed; 
                         bottom: 20px; right: 20px; width: auto; height: auto; 
                         border:2px solid grey; z-index:9999; font-size:14px;
                         background-color:rgba(255, 255, 255, 0.85);
                         padding: 5px;">
             &nbsp; <b>{column_name}</b><br>
             {legend_html_table}
             </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        
        m.save(output_filename)
        print(f"✅ 성공! 지도를 '{output_filename}' 파일로 저장했습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: '{column_name}' 지도 생성 중 문제가 발생했습니다. ({repr(e)})")

# --- 스크립트 실행 예시 (k=7, 즉 7개 범주로 분류) ---
# k 값을 조절하여 원하는 범주 수로 지도를 생성할 수 있습니다.

# 'final_merged_gdf' GeoDataFrame이 이 코드 앞에 정의되어 있어야 합니다.
# 예: final_merged_gdf = gpd.read_parquet("path/to/your/final_merged_gdf.geoparquet")

print("지도 생성을 시작합니다...")

#%%
# 각 컬럼에 대한 지도 생성 함수 호출 (이제 GeoDataFrame인 final_gdf를 전달)
create_map(final_merged_gdf, 'vehicle_count', 'map_2시간이상주행차량대수_nb7.html',k=7)
create_map(final_merged_gdf, 'VLM', 'map_2시간이상전체교통량_nb7.html',k=7)
create_map(final_merged_gdf, 'ratio', 'map_2시간이상ratio_nb7.html',k=7)

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
# create_map(final_merged_gdf, 'vehicle_count', 'map_2시간30분이상주행차량대수.html')
# create_map(final_merged_gdf, 'VLM', 'map_2시간30분이상_전체교통량.html')
# create_map(final_merged_gdf, 'ratio', 'map_2시간30분이상_ratio.html')
# 7개 범주(k=7)로 분류하여 지도 생성
create_map(final_merged_gdf, 'vehicle_count', 'map_2시간30분이상주행차량대수_nb7.html', k=7)
create_map(final_merged_gdf, 'VLM', 'map_2시간30분이상_전체교통량_nb7.html', k=7)
create_map(final_merged_gdf, 'ratio', 'map_2시간30분이상_ratio_nb7.html', k=7)
print("\n🎉 모든 지도 생성이 완료되었습니다!")

