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

import processingDTG
from processingDTG import (
    fetch_geodata,
    upload_geodata,
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
    extract_chunk_numbers
)

# check_memory_usage 함수 (from)
def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Current memory usage: {memory_mb:.2f} MB")
    return f"Current memory usage: {memory_mb:.2f} MB" # 로깅 메시지용으로 문자열도 반환

#%%
# links = gpd.read_file(r"/data1/DTG/NODELINK_NOTSPLIT/MOCT_LINK_utm.shp", engine='pyogrio')
links = gpd.read_file(r"/data1/DTG/NODELINK_NOTSPLIT/MOCT_LINK_utm.shp", engine='pyogrio')

def inJB(links):
    ##링크ID의 앞3자리수를 이용한 지역 분별 방법!! IKSAN!!
    #links = removefromisland(links, CHUNG_ROI)
    if "LINK_ID" in links.columns: 
        link_prefix = links['LINK_ID'].str[:3].astype(int)
    elif "NODE_ID" in links.columns: 
        link_prefix = links['NODE_ID'].str[:3].astype(int)
    else: 
        print("THERE ARE NO LINKID OR NODEID")
        
    
    #전라도 익산청만 필터하긔!!
    # Create a boolean mask for the two valid ranges
    # 광주: 175 to 182
    # condition1 = link_prefix.between(175, 182)
    
    # 전라북도: 305 to 323
    # 전라남도: 324 to 349
    # So.. 전라도: 305 ~ 349
    condition2 = link_prefix.between(305, 323)
    
    # Combine the conditions with OR and filter the GeoDataFrame
    links = links[condition2].copy()
    # links = links[condition1|condition2].copy()
    return links

#마지막에 일반국도로만 필터하기!
# links = inJB(links)[inJB(links)['ROAD_RANK']=="103"]
links = inJB(links)
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
dirpath = f'{current_dir}/{outputfoldername[0]}/{todaydate}'
filenames = [g for g in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, g))]

yearfetch = gpd.read_file(f'{current_dir}/{outputfoldername[0]}/{todaydate}/{filenames[0]}', rows=1,engine='pyogrio')

yearsall = yearfetch['date'].astype(str).str[:4].astype(int)

print(current_dir, yearsall,"CURRENTDIR!!")

#%%

@profile
def append_to_output(gdf, output_files, year, file_path):
    """
    Append aggregated data to output files incrementally.
    Parameters:
    - gdf: GeoDataFrame to aggregate and save
    - output_files: Dictionary of output file paths
    - year: Year for filtering
    """
    
    if gdf is None or gdf.empty:
        return
    
    # # Create the DataFrame for 2022, 2023, 2024
    # date_hour_df = create_date_hour_df(f'{y}', f'{y}')
    # date_hour_df['date'] = date_hour_df['date'].astype(int)

    # gdf2 = date_hour_df.merge(
    #     gdf,
    #     on=['date', 'hour'],
    #     how='left'
    # )
    
    # #pd.datetime 생성 
    # datetime_str = gdf2['date'].astype(str) + gdf2['hour'].astype(str).str.zfill(2)

    # # pd.to_datetime을 사용하여 datetime 타입으로 변환 후 새 칼럼에 할당
    # gdf2['datetime'] = pd.to_datetime(datetime_str, format='%Y%m%d%H')
    
    # 1. 'date'와 'time'을 합쳐서 datetime 객체 생성
    # 'time' 컬럼을 문자열로 변환하고 8자리로 맞추기 위해 앞에 0을 채움
    # gdf['time_str'] = gdf['time'].astype(str).str.zfill(8)
    # 'date'와 'time_str'을 합쳐서 timestamp 생성
    # gdf['timestamp'] = pd.to_datetime(gdf['date'].astype(str) + gdf['time'].str.zfill(6), format='%Y%m%d%H%M%S')
    
    gdf = gdf.sort_values(by=["NO","timestamp", "LINK_ID"])
    
    # for col in sum_columns:
    #         gdf2[col] = gdf2[col].fillna(0)
            
    # # gdf2['hour3'] = (gdf2['hour'] // 3) * 3
    
    # group_configs = [
    #     ('daily_hour', ['date', 'hour', 'V_TYPE']),
    #     # ('daily_3hours', ['date', 'hour3', 'V_TYPE']),
    #     # ('daily', ['year', 'month', 'day', 'V_TYPE']),
    #     # ('monthly', ['year', 'month', 'V_TYPE']),
    #     ('yearly', ['year', 'V_TYPE'])
    #     # ('daily_hour_novtype', ['date', 'hour']),
    #     # ('daily_novtype', ['year', 'month', 'day']),
    #     # ('monthly_novtype', ['year', 'month']),
    #     # ('yearly_novtype', ['year'])
    # ]
    
    # 2. 'NO'(차량 ID)로 그룹화
    grouped = gdf.groupby('NO', observed=True)
    
    # 3. 각 그룹(차량)의 첫 시각과 마지막 시각을 계산
    # agg 함수를 사용하여 min, max 값을 동시에 구함
    durations = grouped['timestamp'].agg(['min', 'max'])
    
    # 주행 시간 (마지막 시각 - 첫 시각) 계산
    durations['duration'] = durations['max'] - durations['min']
    durations['min'] = durations['min'].astype(str)
    durations['max'] = durations['max'].astype(str)
    print(durations.iloc[0])
    # durations.to_xlsx("durations2.xlsx",encoding="UTF-8")
    
    # with pd.ExcelWriter(
    #     "durations_formatted.xlsx",             # 저장할 파일 이름
    #     engine="xlsxwriter",                    # 사용할 엔진
    #     datetime_format="yyyy-mm-dd hh:mm:ss"   # 모든 날짜/시간 컬럼에 적용할 형식
    # ) as writer:
    #     durations.to_excel(writer, sheet_name="Durations", index=True)

    print("파일 'durations_formatted.xlsx' 저장이 완료되었습니다.")
    
    # 4. 주행 시간이 2시간 이상인 차량 필터링
    two_hours = pd.Timedelta(hours=2)
    long_trips = durations[durations['duration'] >= two_hours]
    print(long_trips,"233")
    gdf = gdf[gdf['NO'].isin(long_trips.index.tolist())]
    
    logging.info(f"{gdf['NO'].unique()} /// ddddd!!!!")
    
    print(gdf.info(),"DDDDD!")
    
    # for time_period, group_columns in group_configs:
    # 각 LINK_ID별로 고유한 NO(차량)의 개수를 계산
    linkgroupby = gdf.groupby('LINK_ID')['NO'].nunique().reset_index()
    
    # 집계된 컬럼 이름 변경 (NO -> vehicle_count)
    linkgroupby = linkgroupby.rename(columns={'NO': 'vehicle_count'})
    
    # 이후 로직은 동일
    linkgroupby2 = pd.merge(linkgroupby, links[['LINK_ID',"ROAD_RANK", 'geometry']], on='LINK_ID', how='left')
    print(linkgroupby2.shape,"1")
    linkgroupby2= linkgroupby2[linkgroupby2['ROAD_RANK']=="103"]
    print(linkgroupby2.shape,"2")
    del linkgroupby2['ROAD_RANK']
    # linkgroupby2= linkgroupby2[linkgroupby2['ROAD_RANK']==103]
    # print(linkgroupby2.shape,"3")
    
    # print(linkgroupby.columns)
    # linkgroupby = linkgroupby[linkgroupby['year'].astype(str) == str(year)]
    
    file_name = f"dtgsum_link_{year}"
    # csv_path = os.path.join(output_folder, f"{file_name}.csv")
    # gpkg_path = os.path.join(output_folder, f"{file_name}.gpkg")
    chunk_num = extract_chunk_number(file_path)
    #number_match = re.search(r'_(\d+)_chunk_', file_path)
    # number_match = re.search(r'_(\d+)_chunk_', file_path)
    chunk_num2 = extract_chunk_numbers(file_path)
    # chunk_num2 = extract_chunk_number(file_path)
    gpkg_path = os.path.join(output_folder, f"{file_name}_{chunk_num2}_{chunk_num}.gpkg")
    # Append to CSV
    # mode = 'a' if os.path.exists(csv_path) else 'w'
    # linkgroupby.to_csv(csv_path, mode=mode, index=False, header=not os.path.exists(csv_path))
    # logging.info(f"Appended to CSV: {csv_path}")
    
    # Append to GPKG file (merge with existing if exists)
    if linkgroupby2.shape[0]>1: 
        gdf_out = gpd.GeoDataFrame(linkgroupby2, geometry='geometry', crs=32652)
    else: 
        logging.info(f"THERE IS NONE CALC")
    
    if os.path.exists(gpkg_path):
        existing_gdf = gpd.read_file(gpkg_path, engine='pyogrio')
        gdf_out = pd.concat([existing_gdf, gdf_out]).drop_duplicates().reset_index(drop=True)
    
    # gdf_out2 = aggall(gdf_out)
    #gdf_out2 = inCHUNG(gdf_out2, roi_results, roi_geometries)
    # gdf_out2  = inCHUNG2(gdf_out2, roi_results, roi_geometries)
    # del gdf_out
    ##매번 폴더에 파일을 저장하고 불러오는데 오래걸려서 개선필요!!
    logging.info(f"gdf_out.to_file")
    gdf_out.to_file(gpkg_path, driver='GPKG', encoding='UTF-8', use_arrow=True,index=False)
    logging.info(f"Updated GPKG: {gpkg_path}")
    logging.info(check_memory_usage())

#%%

for y in yearsall:
    folderpath = f'{current_dir}/{outputfoldername[0]}/{todaydate}'
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
    logging.info(file_list)
    
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
    output_folder = f"{folderpath}/{todaydate2}"
    
    # 필요시 output_folder에 대한 추가 작업 수행
    chunk_size = 10000000  # Adjust based on available memory
    # chunk_size = 5000000  # Adjust based on available memory
    
    start_time = time.time()
    
    @profile
    def merge_and_process_files(file_list, output_folder, year):
        """
        Process all .gpkg files for a given year, saving incrementally.
        Parameters:
        - file_list: List of .gpkg file paths
        - output_folder: Directory to save output
        - year: Year for output filename
        Returns:
        - listsofsaved: List of all saved table names
        """
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize listsofsaved to accumulate across all files and chunks
        # all_listsofsaved = []
        logging.info(f"{file_list}")
        logging.info(f"{len(file_list)} 개!!")
        
        for file_path in tqdm(file_list, desc="Processing files"):
            logging.info(f"Processing file: {file_path}")
            try:
                # gdf = gpd.read_file(file_path, encoding='utf-8', engine='pyogrio',rows=1)
                gdf = gpd.read_file(file_path, encoding='utf-8', engine='pyogrio')
                logging.info(check_memory_usage())
                # chunk 단위로 나누어 처리
                print(file_path, "는", len(gdf), "의 ROWS...")
                
                for i in range(0, len(gdf), chunk_size):
                    chunk = gdf.iloc[i:i+chunk_size]
                    
                    if chunk.empty:
                        logging.warning(f"Empty chunk in file: {file_path}")
                        continue
                        
                    processed_chunk = process_chunk2(chunk)
                    # processed_chunk = inlocation(chunk,roi_union, roi_box)
                    logging.info(check_memory_usage())
                    
                    # check_memory_usage()
                    vds_not_matched = find_nearest_link_optimized(processed_chunk, links, link_tree)
                    logging.info(check_memory_usage())
                    # check_memory_usage()
                    # vds_not_matched = uppercase_cols_except_geom(vds_not_matched)
                    # Pass existing listsofsaved and get updated one
                    # all_listsofsaved = append_to_output(
                    #     vds_not_matched, {}, year, file_path, all_listsofsaved
                    # )
                    
                    # logging.info(f"Current listsofsaved: {all_listsofsaved}")
                    append_to_output(vds_not_matched, {}, year,file_path)
                    del processed_chunk, vds_not_matched
                    gc.collect()
                    
                del gdf
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
        
        # return all_listsofsaved
    
    logging.info(check_memory_usage())
#%%

print("BREAK_______________________________________")

#%%
# Main execution
for year in [y]:
    yearly_files = [f for f in file_list2 if str(year) in os.path.basename(f)]
    yearly_files = [f for f in file_list2]
    
    logging.info(f"Processing {len(yearly_files)} files for year ")
    
    if yearly_files:
        merge_and_process_files(yearly_files, output_folder, year)
    else:
        logging.warning(f"No files found for year ")

# Execution time
end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Code execution time: {execution_time/60:.2f} minutes {execution_time%60:.2f} seconds")
# logging.info(f"Final listsofsaved: {final_listsofsaved}")

# 최종 결과 반환/사용
# print(f"모든 처리가 완료되었습니다. 저장된 테이블 목록: {final_listsofsaved}")
logging.info(check_memory_usage())
check_memory_usage()

#%%

# input_path = Path(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\output_JB\251001\251001\dtgsum_link_2025_0_000.gpkg")
# # output_html = Path("vehicle_count_map.html")

# # Inspect the GeoPackage to check CRS and coordinate ranges, and if needed, create a corrected HTML map.
# from pathlib import Path
# import geopandas as gpd
# import folium
# from shapely.geometry import LineString, Point, Polygon

# # input_path = Path("/mnt/data/dtgsum_link_2025_0_000.gpkg")
# if not input_path.exists():
#     raise FileNotFoundError(f"Input file not found: {input_path}")

# gdf = gpd.read_file(input_path)

# info = {}
# info['crs'] = gdf.crs
# info['crs_epsg'] = None
# try:
#     info['crs_epsg'] = gdf.crs.to_epsg() if gdf.crs else None
# except Exception:
#     info['crs_epsg'] = str(gdf.crs)

# info['count'] = len(gdf)
# info['geom_type'] = gdf.geometry.geom_type.unique().tolist()
# info['total_bounds'] = gdf.total_bounds.tolist()  # minx, miny, maxx, maxy

# # sample first geometry coordinates
# sample_coords = None
# if not gdf.empty:
#     geom0 = gdf.geometry.iloc[0]
#     if isinstance(geom0, (LineString, Polygon)):
#         coords = list(geom0.coords) if isinstance(geom0, LineString) else list(geom0.exterior.coords)
#         sample_coords = coords[:5]
#     elif isinstance(geom0, Point):
#         sample_coords = [(geom0.x, geom0.y)]
#     else:
#         sample_coords = str(geom0)

# info['sample_coords_first_geom'] = sample_coords

# # Decide whether CRS seems like UTM (large numbers) or already lat/lon
# minx, miny, maxx, maxy = info['total_bounds']
# def likely_utm(bounds):
#     minx, miny, maxx, maxy = bounds
#     # UTM in meters roughly between (100000..800000, 3500000..4600000) for Korea
#     if (minx > 10000 and maxx > 10000) and (miny > 100000 and maxy > 100000):
#         return True
#     return False

# info['likely_utm_like_coords'] = likely_utm(info['total_bounds'])

# # Print inspection results
# print("CRS:", info['crs'])
# print("CRS EPSG:", info['crs_epsg'])
# print("Number of features:", info['count'])
# print("Geometry types:", info['geom_type'])
# print("Total bounds (minx, miny, maxx, maxy):", info['total_bounds'])
# print("Sample coords (first geom):", info['sample_coords_first_geom'])
# print("Likely UTM-like coordinates?:", info['likely_utm_like_coords'])

# # If CRS is None or not EPSG:4326, set/transform assuming EPSG:32652 if it looks like UTM.
# corrected_html = Path("vehicle_count_map_crs_checked.html")
# try:
#     gdf2 = gdf.copy()
#     if gdf2.crs is None:
#         print("\nCRS is missing. Setting CRS to EPSG:32652 (as you mentioned) and converting to EPSG:4326 for web map.")
#         gdf2 = gdf2.set_crs(epsg=32652, allow_override=True)
#     elif gdf2.crs.to_epsg() == 4326:
#         print("\nCRS already EPSG:4326. No transformation needed for web map.")
#     else:
#         print(f"\nCRS is {gdf2.crs}. Will convert from this CRS to EPSG:4326 for web mapping.")
#     # Transform to 4326 for mapping
#     gdf2 = gdf2.to_crs(epsg=4326)
#     # Create folium map
#     center = ((gdf2.total_bounds[1] + gdf2.total_bounds[3]) / 2.0, (gdf2.total_bounds[0] + gdf2.total_bounds[2]) / 2.0)
#     m = folium.Map(location=center, zoom_start=11, control_scale=True)
    
#     # simple style: color by vehicle_count using continuous colormap
#     import branca.colormap as cm
#     vmin, vmax = gdf2['vehicle_count'].min(), gdf2['vehicle_count'].max()
#     colormap = cm.LinearColormap(colors=["blue","yellow","red"], vmin=vmin, vmax=vmax, caption="vehicle_count")
#     def style_fn(feat):
#         vc = feat['properties'].get('vehicle_count', 0)
#         try:
#             vc = float(vc)
#         except:
#             vc = 0.0
#         return {'color': colormap(vc), 'weight':3, 'opacity':0.8}
#     folium.GeoJson(
#         gdf2,
#         style_function=style_fn,
#         tooltip=folium.features.GeoJsonTooltip(fields=["LINK_ID","vehicle_count"], aliases=["LINK_ID:","vehicle_count:"])
#     ).add_to(m)
#     colormap.add_to(m)
#     m.save(str(corrected_html))
#     print("\nSaved corrected map to:", corrected_html)
# except Exception as e:
#     print("\nError while creating corrected map:", repr(e))

# # Expose info dict for user message
# info

# #%%
# from pathlib import Path
# import geopandas as gpd
# import folium
# from shapely.geometry import LineString, Point, Polygon
# import branca.colormap as cm

# # 파일 경로를 본인 환경에 맞게 설정해주세요.
# input_path = Path(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\output_JB\251001\251001\dtgsum_link_2025_0_000.gpkg")
# corrected_html = Path("vehicle_count_map_google_satellite.html")

# if not input_path.exists():
#     raise FileNotFoundError(f"Input file not found: {input_path}")

# gdf = gpd.read_file(input_path)

# # If CRS is None or not EPSG:4326, set/transform assuming EPSG:32652 if it looks like UTM.
# try:
#     gdf2 = gdf.copy()
#     if gdf2.crs is None:
#         print("\nCRS is missing. Setting CRS to EPSG:32652 and converting to EPSG:4326 for web map.")
#         gdf2 = gdf2.set_crs(epsg=32652, allow_override=True)
#     elif gdf2.crs.to_epsg() == 4326:
#         print("\nCRS already EPSG:4326. No transformation needed for web map.")
#     else:
#         print(f"\nCRS is {gdf2.crs}. Will convert from this CRS to EPSG:4326 for web mapping.")

#     # Transform to 4326 for mapping
#     gdf2 = gdf2.to_crs(epsg=4326)
    
#     # --- 지도 생성 (수정된 부분) ---
#     # 1. 지도 중심 좌표 계산
#     center = ((gdf2.total_bounds[1] + gdf2.total_bounds[3]) / 2.0, (gdf2.total_bounds[0] + gdf2.total_bounds[2]) / 2.0)
    
#     # 2. 기본 지도 생성 (기본값은 OpenStreetMap)
#     m = folium.Map(location=center, zoom_start=11, control_scale=True)

#     # 3. Google Satellite 타일 레이어 추가
#     folium.TileLayer(
#         tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
#         attr='Google',
#         name='Google Satellite',
#         overlay=False,
#         control=True
#     ).add_to(m)
    
#     # 4. (추천) Google Hybrid (위성 + 도로명) 타일 레이어 추가
#     folium.TileLayer(
#         tiles='https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}',
#         attr='Google',
#         name='Google Hybrid',
#         overlay=False,
#         control=True
#     ).add_to(m)

#     # 5. 레이어 컨트롤 추가 (지도 오른쪽 위에 레이어 선택 메뉴 생성)
#     folium.LayerControl().add_to(m)
#     # --- 수정 끝 ---

#     # simple style: color by vehicle_count using continuous colormap
#     vmin, vmax = gdf2['vehicle_count'].min(), gdf2['vehicle_count'].max()
#     colormap = cm.LinearColormap(colors=["blue","yellow","red"], vmin=vmin, vmax=vmax, caption="vehicle_count")
    
#     def style_fn(feat):
#         vc = feat['properties'].get('vehicle_count', 0)
#         try:
#             vc = float(vc)
#         except:
#             vc = 0.0
#         return {'color': colormap(vc), 'weight':3, 'opacity':0.8}
        
#     folium.GeoJson(
#         gdf2,
#         name='Vehicle Count Links', # 1. 레이어 이름 부여
#         style_function=style_fn,
#         tooltip=folium.features.GeoJsonTooltip(fields=["LINK_ID","vehicle_count"], aliases=["LINK_ID:","vehicle_count:"]),
#         overlay=True,              # 2. 이 레이어가 배경지도를 덮는 '오버레이'임을 명시
#         control=True               # 3. 레이어 컨트롤에 포함시켜 껐다 켰다 할 수 있게 함
#     ).add_to(m)
        
#     colormap.add_to(m)
    
#     m.save(str(corrected_html))
#     print(f"\nSaved map with Google Satellite layer to: {corrected_html}")

# except Exception as e:
#     print("\nError while creating corrected map:", repr(e))
    
# #%%

# from pathlib import Path
# import geopandas as gpd
# import folium
# from shapely.geometry import LineString, Point, Polygon
# import branca.colormap as cm

# # 파일 경로를 본인 환경에 맞게 설정해주세요.
# input_path = Path(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\output_JB\251001\251001\dtgsum_link_2025_0_000.gpkg")
# corrected_html = Path("vehicle_count_map_final22.html")

# if not input_path.exists():
#     raise FileNotFoundError(f"Input file not found: {input_path}")

# gdf = gpd.read_file(input_path)

# try:
#     gdf2 = gdf.copy()
#     if gdf2.crs is None:
#         gdf2 = gdf2.set_crs(epsg=32652, allow_override=True)

#     # WGS84 (EPSG:4326)으로 좌표계 변환
#     gdf2 = gdf2.to_crs(epsg=4326)

#     # 지도 중심 좌표 계산
#     center = ((gdf2.total_bounds[1] + gdf2.total_bounds[3]) / 2.0, (gdf2.total_bounds[0] + gdf2.total_bounds[2]) / 2.0)
    
#     # 기본 지도 생성
#     m = folium.Map(location=center, zoom_start=11, control_scale=True)

#     # Google Satellite 타일 레이어 추가
#     # folium.TileLayer(
#     #     tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
#     #     attr='Google',
#     #     name='Google Satellite',
#     #     overlay=False,
#     #     control=True
#     # ).add_to(m)
    
#     # # Google Hybrid 타일 레이어 추가
#     # folium.TileLayer(
#     #     tiles='https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}',
#     #     attr='Google',
#     #     name='Google Hybrid',
#     #     overlay=False,
#     #     control=True
#     # ).add_to(m)
    
#     folium.TileLayer(
#     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#     attr="Esri",
#     name="Esri Satellite",
#     overlay=False,
#     control=True
#     ).add_to(m)
    
#     # 색상 맵 설정
#     vmin, vmax = gdf2['vehicle_count'].min(), gdf2['vehicle_count'].max()
#     colormap = cm.LinearColormap(colors=["blue","yellow","red"], vmin=vmin, vmax=vmax, caption="vehicle_count")
    
#     def style_fn(feat):
#         vc = feat['properties'].get('vehicle_count', 0)
#         try:
#             vc = float(vc)
#         except (ValueError, TypeError):
#             vc = 0.0
#         return {'color': colormap(vc), 'weight':3, 'opacity':0.8}
    
#     # 💡 GeoJson 레이어 추가 (수정된 부분)
#     folium.GeoJson(
#         gdf2,
#         name='Vehicle Count Links', # 이름 부여
#         style_function=style_fn,
#         tooltip=folium.features.GeoJsonTooltip(fields=["LINK_ID","vehicle_count"], aliases=["LINK_ID:","vehicle_count:"]),
#         overlay=True,              # 오버레이 레이어로 명시
#         control=True               # 레이어 컨트롤에 포함
#     ).add_to(m)
    
#     # 범례와 레이어 컨트롤 추가
#     colormap.add_to(m)
#     folium.LayerControl().add_to(m)
    
#     m.save(str(corrected_html))
#     print(f"\nSaved final map to: {corrected_html}")

# except Exception as e:
#     print("\nAn error occurred:", repr(e))
    
