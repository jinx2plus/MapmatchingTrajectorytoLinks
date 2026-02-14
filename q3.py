# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 04:10:59 2025

@author: Yong Jin Park
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:23:46 2025

@author: Yong Jin Park
"""

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
import logging
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
# from io import StringIO
print("YES")

import processingDTGJB
from processingDTGJB import (
    fetch_geodata,
    # upload_geodata,
    aggall,
    inlocation,
    # inIKSAN,
    process_chunk,
    process_chunk2,
    countNOs_optimized,
    process_vds_and_links,
    find_nearest_link_optimized,
    inCHUNG,
    # inCHUNG2,
    uppercase_cols_except_geom,
    create_date_hour_df,
    removefromisland,
    filter_files_from_chunk,
    extract_chunk_number,
    # allnationROI,
    extract_chunk_numbers,
    eraseunder2hours,
    tripsbtw15,
    #filter_non_driving,
    #remove_gps_scribble,
    remove_gps_scribble_improved,
    eraseunder2hoursandhalf
)

# check_memory_usage 함수 (from)
def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Current memory usage: {memory_mb:.2f} MB")
    return f"Current memory usage: {memory_mb:.2f} MB" # 로깅 메시지용으로 문자열도 반환

#%%
# links = gpd.read_file(r"/data1/DTG/NODELINK_NOTSPLIT/MOCT_LINK_utm.shp", engine='pyogrio')
links = gpd.read_file(r"/data1/DTG/2023_1/JB/JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
# links = gpd.read_file(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\JBLINK.shp", engine='pyogrio',columns=['LINK_ID',"ROAD_RANK",'geometry'])
links103 = links[links['ROAD_RANK']=="103"]['LINK_ID'].tolist()

#마지막에 일반국도로만 필터하기!
# links = inJB(links)[inJB(links)['ROAD_RANK']=="103"]
# links = inJB(links)
link_tree = STRtree(links.geometry.values)

# links.to_file("links.gpkg")

#%%
fromfile= "ggg"
current_dir = os.getcwd()
from datetime import datetime# 현재 날짜를 "yymmdd" 형식으로 변환
todaydate = datetime.today().strftime("%y%m%d")
outputfoldername = [
    name for name in os.listdir(os.getcwd())
    if 'output' in name and os.path.isdir(os.path.join(os.getcwd(), name))
]

folderpath = f'{current_dir}/output_JB'

all_items_in_folder = os.listdir(folderpath)
numeric_folders = []

for item_name in all_items_in_folder:
    item_path = os.path.join(folderpath, item_name)
    
    # Check if the item is a directory and its name ends with an underscore followed by a number
    # if os.path.isdir(item_path) and "_" in item_name and item_name.split("_")[-1].isdigit():
    if os.path.isdir(item_path) and item_name.split("_")[-1].isdigit():
        
        # Extract the numeric part, convert it to an integer, and append it
        numeric_part = item_name
        numeric_folders.append(int(numeric_part))

# Now numeric_folders will contain integers like [250605, ...]
print(numeric_folders,"숫자이름의 폴더이름은??!!")

# 4. 숫자 폴더가 하나 이상 있으면, 그중 가장 큰 숫자를 찾습니다.
if numeric_folders != []:
    latest_number = max(numeric_folders)
    print(f"찾아낸 숫자 폴더들: {numeric_folders}")
    # 5. 찾은 최신 번호를 todaydate2 변수에 할당합니다.
    todaydate2 = str(latest_number)
else:
    logging.warning(f"숫자이름 폴더가 없음 !!! ERROR")

# # Step 2: Define relevant columns for max value calculation
# relevant_columns = ['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 'Q_LTURN', 'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE']
# sum_columns = ['VLM']

# logging.info()
logging.info(check_memory_usage())
check_memory_usage()

#%%
current_dir = os.getcwd()
dirpath = f'{current_dir}/{outputfoldername[0]}/{todaydate2}'
filenames = [g for g in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, g))]

yearfetch = gpd.read_file(f'{current_dir}/{outputfoldername[0]}/{todaydate2}/{filenames[0]}', rows=1,engine='pyogrio')

yearsall = yearfetch['date'].astype(str).str[:4].astype(int)

print(current_dir, yearsall,"CURRENTDIR!!")

#%%

# chunk_size=10000000
@profile
# (상단 import 및 설정은 그대로 유지)
# (상단 import 및 설정은 그대로 유지)
# ... links, link_tree 등 변수 설정 ...

@profile
def append_to_output(gdf, file_path, year, output_folder, VLMgdf):
    """
    요청하신대로 청크별 중간 결과를 파일로 저장하는 함수.
    """
    if gdf is None or gdf.empty:
        logging.warning("GDF is empty, skipping file save.")
        return

    # 1. 링크별 차량 수 집계
    linkgroupby = gdf.groupby('LINK_ID')['NO_TRIPID'].nunique().reset_index()
    linkgroupby = linkgroupby.rename(columns={'NO_TRIPID': 'vehicle_count'})
    
    # 2. 링크 shape 정보와 병합
    links2 = links[links['ROAD_RANK'] == "103"]
    linkgroupby2 = pd.merge(links2[['LINK_ID', "geometry"]], linkgroupby, on='LINK_ID', how='left')

    # 3. VLM(전체 교통량) 정보와 병합
    final_chunk_gdf = pd.merge(linkgroupby2, VLMgdf, on='LINK_ID', how='left')
    logging.info(f"{final_chunk_gdf.head()}")
    # NaN 값을 0으로 채우기
    final_chunk_gdf['vehicle_count'] = final_chunk_gdf['vehicle_count'].fillna(0).astype(int)
    final_chunk_gdf['VLM'] = final_chunk_gdf['VLM'].fillna(0).astype(int)
    final_chunk_gdf = final_chunk_gdf[(final_chunk_gdf['vehicle_count'] != 0) | (final_chunk_gdf['VLM'] != 0)]
    final_chunk_gdf = final_chunk_gdf[(final_chunk_gdf['vehicle_count'] > 0) | (final_chunk_gdf['VLM'] > 0)]

    # 4. 요청하신 형식으로 파일 이름 생성 및 저장
    file_name = f"dtgsum_link_{year}"
    chunk_num = extract_chunk_number(file_path)  # 기존 helper 함수 사용
    chunk_num2 = extract_chunk_numbers(file_path) # 기존 helper 함수 사용
    
    output_path = os.path.join(output_folder, f"{file_name}_{chunk_num2}_{chunk_num}.geoparquet")
    
    if not final_chunk_gdf.empty:
        gdf_out = gpd.GeoDataFrame(final_chunk_gdf, geometry='geometry', crs=links.crs)
        gdf_out.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        logging.info(f"Saved intermediate chunk file: {output_path}")
    else:
        logging.warning(f"Final chunk GDF is empty for {file_path}, chunk {chunk_num}.")

for y in yearsall:
    folderpath = f'{current_dir}/{outputfoldername[0]}/{todaydate2}'
    # folderpath = f'{current_dir}/onlyone{y}/'
    print(folderpath)
    file_list = glob(os.path.join(folderpath, '*.gpkg'))
    file_list.sort()
    
    file_list2 = []
    for f in file_list:
        filename = os.path.basename(f)
        parts = filename.split("_")
        # if len(parts) >= 2 and parts[1] == '0':
        if len(parts) >= 2:
            file_list2.append(f)
    file_list = file_list2
    # logging.info(file_list)
    
    #############################################
    originalfilelist = file_list
    #############################################
    
    try: 
        if fromfile is not None and fromfile.endswith(".gpkg"): 
            file_list2 = filter_files_from_chunk(file_list, fromfile)
        else:
            file_list2 = file_list
    except NameError:
        file_list2 = file_list
    print(file_list2)
    logging.info(check_memory_usage())
    ##############################################
    output_folder = f"{folderpath}/{todaydate}"
    
    # 필요시 output_folder에 대한 추가 작업 수행
    chunk_size = 10000000  # Adjust based on available memory
    # chunk_size = 5000000  # Adjust based on available memory
    
    start_time = time.time()
    
@profile
def merge_and_process_files(file_list, output_folder, year):
    """
    최적화된 로직으로 파일을 처리하고, 청크별로 중간 결과 파일을 저장하는 함수.
    """
    os.makedirs(output_folder, exist_ok=True)
    req_cols = ['NO', 'date', 'time', 'geometry']

    for file_path in tqdm(file_list, desc=f"Processing files for year {year}"):
        logging.info(f"Processing file: {file_path}")
        try:
            # gdf = gpd.read_file(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\output_JB\251004\020_chunk_002.gpkg", engine='pyogrio', columns=req_cols)
            gdf = gpd.read_file(file_path, encoding='utf-8', engine='pyogrio',columns=req_cols)
            #FIX!!
            # gdf = gdf[gdf['date'].between(20250301, 20250307)]
            gdf['date'] = gdf['date'].astype(str)
            gdf['time'] = gdf['time'].astype(str).str.zfill(8).str[:6]
            gdf['timestamp'] = pd.to_datetime(gdf['date'] + gdf['time'], format='%Y%m%d%H%M%S')

            for i in range(0, len(gdf), chunk_size):
                chunk = gdf.iloc[i:i+chunk_size].copy()
                chunk.drop_duplicates(inplace=True)
                
                # 1. 삭제할 조건을 하나의 변수에 저장합니다.
                condition_to_drop = (chunk['NO'] == 4347) & \
                                    (chunk['date'] == "20250331") & \
                                    (chunk['time'].astype(int).between(95840, 105426))
                
                # (추가) 삭제될 행의 개수를 계산합니다.
                num_to_drop = condition_to_drop.sum()
                
                if num_to_drop and num_to_drop>0:
                    original_count = len(chunk)
                    
                    # (추가) logging.info()를 사용하여 로그를 남깁니다.
                    logging.info(f"원본 행 개수: {original_count}. 삭제 조건에 맞는 행: {num_to_drop}개.")
                    
                    # 2. 조건에 해당하지 않는 행만 남깁니다.
                    chunk = chunk[~condition_to_drop].reset_index(drop=True)
                    
                    # (추가) 작업 완료 후의 상태를 로그로 남깁니다.
                    logging.info(f"삭제 작업 완료. 현재 행 개수: {len(chunk)}.")
                
                if chunk.empty:
                    continue
                
                # chunk22 =filter_non_driving(chunk)
                # chunk22 = remove_gps_scribble(chunk)
                chunk33 = remove_gps_scribble_improved(chunk)
                del chunk
                # 1. 트립 분리
                chunk2, chunkall = tripsbtw15(chunk33)
                del chunk33
                # 2. 링크 매칭 (★★★★★ 최적화 포인트: 단 1회 수행 ★★★★★)
                matched_all_trips = find_nearest_link_optimized(chunkall, links, link_tree)
                
                # 일반국도용 전체 교통량(VLM) 계산
                VLMgdf = countNOs_optimized(matched_all_trips, links103)
                if VLMgdf is None or VLMgdf.empty:
                    # VLM이 없는 경우, 빈 DataFrame을 만들어 이후 merge 에러 방지
                    VLMgdf = pd.DataFrame({'LINK_ID': [], 'VLM': []})

                # 3. 2시간 미만 트립 ID 필터링
                # valid_trip_ids = eraseunder2hours(chunk2)['NO_TRIPID'].unique()
                valid_trip_ids = eraseunder2hoursandhalf(chunk2)['NO_TRIPID'].unique()
                final_chunk = matched_all_trips[matched_all_trips['NO_TRIPID'].isin(valid_trip_ids)]
                final_chunk.reset_index(drop=True)
                # 4. 중간 결과 파일 저장
                append_to_output(final_chunk, file_path, year, output_folder, VLMgdf)

                # 메모리 정리
                del chunk2, chunkall, matched_all_trips, final_chunk, VLMgdf
                gc.collect()

            del gdf
            gc.collect()
            logging.info(check_memory_usage())

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

# --- 메인 실행 로직 ---
start_time = time.time()

for year in [y]:
    yearly_files = [f for f in file_list2]
    logging.info(f"Processing {len(yearly_files)} files for year {year}")
    
    # 중간 결과가 저장될 폴더
    # output_folder = f"{folderpath}/{todaydate}"

    if yearly_files:
        # 1. 파일 처리 및 중간 청크 파일들 생성
        merge_and_process_files(yearly_files, output_folder, year)

        # 2. 모든 중간 결과 파일들을 취합하여 최종 결과물 생성
        logging.info("Starting final merge of all intermediate chunk files...")
        intermediate_files = glob(os.path.join(output_folder, f"dtgsum_link_{year}_*.geoparquet"))
        
        if not intermediate_files:
            logging.warning("No intermediate files were found. Cannot create a final result.")
        else:
            # 점진적 집계를 위한 초기 빈 DataFrame 생성
            final_agg_df = pd.DataFrame()
        
            # 중간 파일들을 하나씩 순회하며 집계 결과를 누적
            for f in tqdm(intermediate_files, desc="Aggregating chunk files"):
                # 중간 파일 하나를 읽어옴
                chunk_gdf = gpd.read_parquet(f)
                
                # 현재 파일의 데이터 집계
                # geometry는 집계에 필요 없으므로 제외하여 성능 향상
                current_agg = chunk_gdf.groupby('LINK_ID')[['vehicle_count', 'VLM']].sum().reset_index()
        
                if final_agg_df.empty:
                    # 첫 번째 파일인 경우, 바로 할당
                    final_agg_df = current_agg
                else:
                    # 기존 집계 결과와 현재 파일의 집계 결과를 합침
                    final_agg_df = pd.concat([final_agg_df, current_agg])
                    # 합친 결과에서 다시 LINK_ID별로 합산하여 누적
                    final_agg_df = final_agg_df.groupby('LINK_ID')[['vehicle_count', 'VLM']].sum().reset_index()
            
            logging.info("Final aggregation complete. Merging with geometry.")
        
            # 최종 집계 결과에 링크의 geometry 정보 결합
            links_geom = links[['LINK_ID', 'geometry']].drop_duplicates(subset=['LINK_ID'])
            final_merged_gdf = pd.merge(links_geom, final_agg_df, on='LINK_ID', how='left')
            
            # 집계되지 않은 링크는 0으로 채우기
            final_merged_gdf['vehicle_count'] = final_merged_gdf['vehicle_count'].fillna(0).astype(int)
            final_merged_gdf['VLM'] = final_merged_gdf['VLM'].fillna(0).astype(int)
            final_merged_gdf= final_merged_gdf[(final_merged_gdf['vehicle_count'] != 0) | (final_merged_gdf['VLM'] != 0)]
            # final_merged_gdf= final_merged_gdf[(final_merged_gdf['vehicle_count'] > 0) | (final_merged_gdf['VLM'] > 0)]
            final_merged_gdf = final_merged_gdf[final_merged_gdf['LINK_ID'].isin(links103)]
            
            # 최종 결과 파일 저장
            final_output_path = os.path.join(output_folder, f"dtgsum_link_{year}_FINAL_RESULT.geoparquet")
            final_merged_gdf.to_parquet(final_output_path, engine='pyarrow', compression='snappy', index=False)
            logging.info(f"Successfully created final result file at: {final_output_path}")

# (실행 시간 계산 및 출력은 기존과 동일)
end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Code execution time: {execution_time/60:.2f} minutes {execution_time%60:.2f} seconds")
logging.info(check_memory_usage())

#%%
