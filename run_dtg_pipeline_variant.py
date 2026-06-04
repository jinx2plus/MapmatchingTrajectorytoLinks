# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 09:25:50 2025

@author: Yong Jin Park
"""

#%%
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import geopandas as gpd
from project_paths import resolve_repo_file
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

#%%
#240821) save: dataframes by year in one dictionary 
#20240809??받�? ?�이???�전 ?�이?��? ?�른 ?��? 차종??구분??
# yearsofall = [2021,2022,2023,2024]

# yearsofall = [2024]
df = {}
# carid_to_no_mapping = {}
# global_carid_mapping ={}

#%% 3/24
start_time = datetime.now()
print(f"?�작 ?�간: {start_time}")

def is_valid_coordinate(lon, lat):
    """
    좌표값의 ?�효?�을 검?�하???�수
    """
    
    try:
        return (
            isinstance(lon, (int, float)) and
            isinstance(lat, (int, float)) and
            np.isfinite(lon) and
            np.isfinite(lat) and
            124 <= lon <= 132 and
            33 <= lat <= 43
        )
    except:
        return False

def inIKSAN(points):

    JLLNEW2_path = resolve_repo_file("JBROI2.shp")
    JLLNEW2 =gpd.read_file(JLLNEW2_path, encoding='utf-8')
    JLLNEW2 = JLLNEW2.to_crs(32652)
    roi_union = JLLNEW2.geometry.union_all()
    # roi_box = box(*roi_union.bounds)
    # del JLLNEW2
    # vds2 = TA.sjoin(JLLNEW2, how="inner", predicate="within").drop(columns=['index_right']).drop_duplicates()
    # points_within = points[points.geometry.intersects(roi_union)]
    points_within = points.sjoin(JLLNEW2, how="inner", predicate="within").drop(columns=['index_right'])
    points_within = points_within.drop(columns=['id','BASE_DATE','SIDO_CD','SIDO_NM',"CHUNG"])
    print(len(points_within),"�?)
    # vds = TA.sjoin(nodebuffer_gdf, how="inner", predicate="within").drop(columns=['index_right']).drop_duplicates()

    # tree2 = STRtree([roi_union])
    # points_within = points[points.geometry.apply(lambda x: tree2.query(x.buffer).size > 0)]
    
    # nodebuffer180m = nodes.to_crs(32652).geometry.buffer(180,resolution=5,mitre_limit=2)
    # tree2 = STRtree(nodes.geometry.values)
    # vds = vdso[vdso.geometry.apply(lambda x: len(tree2.query(x.buffer(180,resolution=5,mitre_limit=2))) > 0)]
    
    # # R-tree�??�용??공간 ?�덱???�성
    # spatial_index = points.sindex

    # # ROI?� 교차?�는 ???�보�??�터�?
    # possible_matches_index = list(spatial_index.query(roi_union.bounds, predicate='intersects'))
    # possible_matches = points.iloc[possible_matches_index]
    
    # # ?�확??ROI ?��????�치???�들�??�터�?
    # points_within = possible_matches[possible_matches.within(roi_union)]
    
    return points_within

#?�플
# def get_no_for_carid(carid):
#     """
#     carid???�??고유??NO 값을 반환?�고, ?�요???�로??값을 추�?
#     """
    
#     global carid_to_no_mapping, next_no
#     if carid not in carid_to_no_mapping:
#         carid_to_no_mapping[carid] = next_no
#         next_no += 1
#     return carid_to_no_mapping[carid]

def process_large_file(file_index, chunk_size=10000000,global_carid_mapping=None):
    
    """
    ?�?�량 ?�일??�?�� ?�위�?처리?�는 ?�수
    """
    
    filepath = f'/data1/DTG/2023/WEEKLY/20250814/output_data-{file_index}.txt'
    current_max_no = max(global_carid_mapping.values()) if global_carid_mapping else 0
    # file_index = 0
    
    try:
        # chunks = pd.read_csv(filepath, sep='|', header=None, chunksize=chunk_size, encoding='utf-8-sig')
        chunks = pd.read_csv(filepath, sep='|', header=None, chunksize=chunk_size, encoding='utf-8-sig')
        
        for chunk_num, chunk in enumerate(chunks):
            try:
                
                # 컬럼�?지??
                chunk.columns = ['date', 'time', 'carid', 'V_TYPE', 'lon', 'lat', 
                               'OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 
                               'Q_STOP', 'Q_LTURN', 'Q_RTURN', 'Q_UTURN', 
                               'Q_OVERTAKE', 'Q_LCHANGE']
                
                columns_to_drop = ['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 'Q_LTURN', 'Q_RTURN',
                                   'Q_UTURN','Q_OVERTAKE', 'Q_LCHANGE']
                chunk.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                #1주값�?뽑기!!
                chunk['date'] = '2025' + chunk['date'].astype(str).str[4:]
                chunk = chunk.loc[(chunk['V_TYPE']==31)| (chunk['V_TYPE']==32)][['date', 'time', 'carid', 'lon', 'lat']]
                # chunk = chunk[chunk['date'].between(20250301, 20250307)]
                
                # 좌표 ?�이??검�?�??�터�?
                # NaN, 무한?� �??�거
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.dropna(subset=['lon', 'lat'])
                
                # ?�효??좌표값만 ?�터�?
                valid_coords = chunk.apply(lambda row: is_valid_coordinate(row['lon'], row['lat']), axis=1)
                chunk = chunk[valid_coords]
                
                if len(chunk) == 0:
                    print(f"No valid coordinates in chunk {chunk_num}")
                    continue
                
                # ?�이???�렬
                chunk = chunk.sort_values(['carid', 'date', 'time'])
                # carid�??��???NO�?변??
                unique_carids = chunk['carid'].unique()
                for carid in unique_carids:
                    if carid not in global_carid_mapping:
                        current_max_no += 1
                        global_carid_mapping[carid] = current_max_no
                
                # 매핑??NO �??�용
                chunk['NO'] = chunk['carid'].map(global_carid_mapping)
                chunk = chunk.drop('carid', axis=1)
                
                # carid�?NO�?변??
                # 250325 3/25 ?�정 #?�플
                # chunk['NO'] = chunk['carid'].apply(get_no_for_carid)
                # chunk = chunk.drop('carid', axis=1)
                
                # ?�래
                # chunk['NO'] = pd.factorize(chunk['carid'])[0] + 1
                # chunk = chunk.drop('carid', axis=1)
                
                # hour 컬럼 추�?
                chunk['hour'] = chunk['time'].astype(str).str[:2]
                
                # 컬럼 ?�정??
                columns_order = ['date', 'time', 'hour', 'NO', 'lon', 'lat']
                chunk = chunk[columns_order]
                
                # ?�이???�??최적??
                for col in chunk.select_dtypes(include=['float64']).columns:
                    chunk[col] = chunk[col].astype('float32')
                for col in chunk.select_dtypes(include=['int64']).columns:
                    chunk[col] = chunk[col].astype('int32')
                
                # 좌표�?범위 ?�인???�한 로깅
                print(f"Longitude range: {chunk['lon'].min()} to {chunk['lon'].max()}")
                print(f"Latitude range: {chunk['lat'].min()} to {chunk['lat'].max()}")
                
                
                try:
                    # GeoDataFrame 변??
                    gdf = gpd.GeoDataFrame(
                        chunk,
                        geometry=gpd.points_from_xy(
                            chunk['lon'], 
                            chunk['lat'],
                            crs="EPSG:4326"
                        )
                    )
                    
                    # 좌표�?변??
                    gdf = gdf.to_crs(32652)
                    
                    # #좌표�??�인
                    # print("?�드?�이?��? ROI?�이?�의 crs가 같�?가?", gdf.crs==JLLNEW2.crs)
                    
                    gdf2 = inIKSAN(gdf)
                    #gdf2= gdf[gdf.intersects(JLLNEW2.union_all(), align=True)]
                    # gdf2 = gdf.copy()
                    del gdf
                    
                    # 출력 ?�렉?�리 ?�성
                    from datetime import datetime# ?�재 ?�짜�?"yymmdd" ?�식?�로 변??
                    todaydate = datetime.today().strftime("%y%m%d")
                    
                    folderpath = f"output_JB/"
                    os.makedirs(folderpath, exist_ok=True)
                    output_dir = f"{folderpath}{todaydate}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # �?��별로 ?�일 ?�??
                    output_filename = os.path.join(output_dir, f"{file_index}_chunk_{chunk_num:03d}.gpkg")
                    # gdf.to_file(output_filename, driver='ESRI Shapefile', encoding='UTF-8')
                    gdf2.to_file(output_filename, driver='GPKG', encoding="utf-8",engine="pyogrio")
                    
                    # 종료 ?�간 출력
                    end_time = datetime.now()
                    print(f"Processed {output_filename}: {len(gdf2)} rows")
                    print(f"종료 ?�간: {end_time}")
                    
                except Exception as e:
                    print(f"Error creating GeoDataFrame for chunk {chunk_num}: {e}")
                    # 문제가 ?�는 ?�이??로깅
                    problem_data = chunk[~chunk.apply(lambda row: is_valid_coordinate(row['lon'], row['lat']), axis=1)]
                    print(f"Problematic coordinates:\n{problem_data[['lon', 'lat']]}")
                    continue

                # 메모�??�리
                del chunk, gdf2
                gc.collect()
                
                # return global_carid_mapping
            
            except Exception as e:
                print(f"Error processing chunk {chunk_num} of file {filepath}: {e}")
                continue
            
            # return global_carid_mapping

    except Exception as e:
        print(f"Error opening file {filepath}: {e}")
    return global_carid_mapping
    
#%%
def dtg_optimized():
    """
    최적?�된 메인 처리 ?�수
    """
    global_carid_mapping = {}

    for file_index in tqdm(range(10), desc=f"Processing__"):
        global_carid_mapping = process_large_file(file_index, global_carid_mapping=global_carid_mapping)
        
        # ?�재까�???매핑 ?�태 ?�??(?�택?�항)
        mapping_filename = f'carid_mapping__{file_index}.json'
        
        with open(mapping_filename, 'w') as f:
            json.dump(global_carid_mapping, f)
    
    return global_carid_mapping
    
# 매핑 ?�보�??�일�??�?�하???�수
def save_mapping(mapping_dict, filename):
    """매핑 ?�보�??�일�??�??""
    import json
    with open(filename, 'w') as f:
        json.dump(mapping_dict, f)

# 매핑 ?�보�??�일?�서 로드?�는 ?�수
def load_mapping(filename):
    """?�일?�서 매핑 ?�보�?로드"""
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 매핑 검�??�수
def verify_mapping(mapping_dict):
    """매핑???��??�을 검�?""
    reverse_mapping = {}
    for carid, no in mapping_dict.items():
        if no in reverse_mapping and reverse_mapping[no] != carid:
            print(f"Inconsistency found: NO {no} is mapped to multiple carids")
        reverse_mapping[no] = carid
    return len(mapping_dict) == len(set(mapping_dict.values()))


# ?�행
if __name__ == "__main__":
    # yearsofall = [2024]
    
    global_carid_mapping = dtg_optimized()
        
    save_mapping(global_carid_mapping, f'carid_mapping_.json')

    mapping_filename = f'carid_mapping__final.json'
        
    final_mapping = load_mapping(mapping_filename)
        
    if verify_mapping(final_mapping):
        print("Mapping verification successful")
    else:
        print("Mapping verification failed")

#%%


#%%


#%%



