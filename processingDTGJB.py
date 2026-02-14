import pandas as pd
# from datetime import datetime
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
import psutil

# from sqlalchemy import create_engine
import psycopg2

plt.rc('font', family='NanumGothic')
# from io import StringIO

#%%

# import movingpandas as mpd
# import geopandas as gpd
from sqlalchemy import create_engine
# from geopandas_postgis import PostGIS
# from geodatasets import get_path
# import geopandas as gpd
import pyogrio
from project_paths import resolve_repo_file
import fiona
# import os
from sqlalchemy import create_engine
import gpxpy

from sqlalchemy import inspect
import pandas as pd
import numpy as np
import chardet
import shapefile
import psycopg2
import folium
import re
# from rtree import index
from shapely.geometry import Polygon, LineString
# import GeoJson
# import pandas as pd
from shapely.geometry import Point
# from shapely.ops import transform
# import pyproj
# from functools import partial
from shapely.strtree import STRtree
# import math
# from math import ceil 
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from pathlib import Path
from line_profiler import profile
print("module imported success! processingDTG")
#JLLNEW2 =gpd.read_file(r"JLLNEW2.shp", encoding='utf-8', engine='pyogrio')
roi_union_path = resolve_repo_file("roi_union.gpkg")
roi_union = gpd.read_file(roi_union_path, engine='pyogrio').geometry.iloc[0]
roi_box_path = resolve_repo_file("roi_box.gpkg")
roi_box = gpd.read_file(roi_box_path, engine='pyogrio').geometry.iloc[0]
def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Current memory usage: {memory_mb:.2f} MB")

def fetch_geodata(
    table_name,
    geometry_column="geom",
    crs=None,
    dbname="******",
    host="******",
    port="******",
    user="******",
    password="******"
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
        # print(f"Connected to database '{dbname}'.")
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

import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from shapely.geometry import Point
from shapely.ops import transform
import shapely

def _to_2d(geom):
    """Z 값을 ?�거(2D 강제)."""
    if geom is None or geom.is_empty:
        return geom
    
    # Shapely 2.0 ?�상
    if hasattr(shapely, "force_2d"):
        return shapely.force_2d(geom)
    
    # Shapely 1.x: z 좌표 무시
    return transform(lambda x, y, z=None: (x, y), geom)


import csv
def copy_from_stringio(df, table):
    """
    메모�??�의 CSV�??�용?�여 PostgreSQL??COPY 명령???�행?�는 ?�퍼 ?�수.
    """
    # 1. DataFrame??메모�????�일(StringIO)처럼 ?�작?�는 버퍼???�니??
    #    - header=False: 컬럼명�? ?��? ?�습?�다.
    #    - index=False: Pandas ?�덱?�는 ?��? ?�습?�다.
    #    - sep='\t': ??���??�드�?구분?�니??(?�표가 ?�이?�에 ?�함??경우 ?��?.
    buffer = StringIO()
    df.to_csv(buffer, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    buffer.seek(0) # 버퍼??커서�?�??�으�??�동?�킵?�다.
    
    # 2. ?�이?�베?�스??직접 ?�결?�여 COPY 명령???�행?�니??
    connection = table.bind.raw_connection()
    try:
        cursor = connection.cursor()
        # STDIN(?��? ?�력)?�로부???�이?��? 복사?�라???��?
        cursor.copy_expert(f"COPY {table.schema}.{table.name} FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')", buffer)
        connection.commit()
        logging.info("COPY command executed successfully.")
    except Exception as e:
        connection.rollback()
        logging.error(f"COPY command failed: {e}")
        raise e
    finally:
        connection.close()


@profile
def aggall(gdf):
    # Step 2: Define relevant columns
    # relevant_columns = ['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 
                        # 'Q_LTURN', 'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE']
    gdf = gdf[~(gdf[['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 'Q_LTURN', 'Q_RTURN',        
                     'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE', 'VLM']] == 0).all(axis=1)].reset_index(drop=True)
    for col in gdf.select_dtypes(include=['float64']).columns:
        if col != gdf.geometry.name:
            gdf[col] = gdf[col].astype(np.int32)
            
    if gdf.empty:
        print(f"  All rows removed (all OV_SPD to VLM are zero) in gdf). Skipping layer.")
    
    # Define output path
    logging.info(f"{len(gdf)} aggall ")
    # Drop the temporary max_value column
    return gdf

@profile
#링크buffer(10m) ?��???point�??�터?�기 ?�함!
#?��? dtg?�이?�는 ?�면?�로???�치??것도 ?�어????��?�야 ??
def inlocation(points, roi_union, roi_box):
    """
    Filter points within the union of polygons in 공간?�범??using STRtree.
    Parameters:
    - points: GeoDataFrame with Point geometries
    Returns:
    - GeoDataFrame of points within the polygon
    """

    ###
    # logging.info(f"points within points_in_box_START")
    points_in_box = points[points.geometry.within(roi_box)]
    # logging.info(f"points within points_in_box_END // {len(points_in_box)}")
    
    # logging.info(f"points_within_START")
    points_within = gpd.sjoin(
        points_in_box,
        gpd.GeoDataFrame(geometry=[roi_union], crs=32652,index=[0]),
        how="inner",
        predicate="within"
    ).drop(columns=['index_right'])
    # points_within = points_in_box[points_in_box.geometry.within(roi_union)]
    
    # logging.info(f"points_within_END len(points_within)")
    # print("def inlocation_?�터", len(points_within))
    logging.info(f"{len(points_within)} points within ROIBOX, def inlocation!!")
    
    return points_within

@profile
def inIKSAN(points,JLLNEW2):
    """
    Filter points within the union of polygons in JLLNEW3.shp using STRtree.
    Parameters:
    - points: GeoDataFrame with Point geometries
    Returns:
    - GeoDataFrame of points within the polygon
    """
    # points = points.to_crs(32652)
    points_within = points.sjoin(JLLNEW2, how="inner", predicate="within").drop(columns=['index_right'])
    
    # print(f"{len(points_within)} points within Iksan")
    logging.info(f"{len(points_within)} points within inIKSAN")
    
    return points_within

@profile
def process_chunk(chunks):
    """
    Process a chunk of a .gpkg file, applying inIKSAN and transformations.
    Parameters:
    - chunk: GeoDataFrame chunk
    Returns:
    - Processed GeoDataFrame
    """
    logging.info(f"process_chunk START, {len(chunks)}")
    
    #250714 
    chunk = inlocation(chunks,roi_union, roi_box)
    # def inlocation(points, roi_union, roi_box)
    
    #?�산�??�터
    #chunk = inIKSAN(chunk,JLLNEW2)
    
    if chunk.empty:
        return None
    
    if 'hrs' in chunk.columns:
        chunk.rename(columns={'hrs': 'hour'}, inplace=True)
    
    time_str = chunk['time'].astype(str).str.zfill(8)
    chunk['hour'] = time_str.str[0:2].astype(int)
    chunk['min'] = time_str.str[2:4].astype(int)
    chunk['sec'] = time_str.str[4:6].astype(int)
    chunk['time'] = time_str.str[:-2]
    chunk['datetime'] = pd.to_datetime(
        chunk['date'].astype(str) + chunk['time'].str.zfill(6),
        format='%Y%m%d%H%M%S'
    )
    
    chunk = chunk.sort_values(by=["NO","datetime", "date", "time", "hour", "min", "sec"])
    
    chunk['same_NO'] = (chunk['NO'] == chunk['NO'].shift(1)).astype(int)
    
    chunk['timediff2'] = chunk['datetime'] - chunk['datetime'].shift(1)
    
    chunk['less_than_1sec'] = np.where(
        (chunk['same_NO'] == 1) &
        (chunk['timediff2'].dt.total_seconds() > 0) &
        (chunk['timediff2'].dt.total_seconds() <= 2),
        1,
        0
    )
    
    columns_to_check = ['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC',
                        'Q_STOP', 'Q_LTURN', 'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE']
    
    chunk['new_column'] = np.where(
        (chunk['less_than_1sec'] == 1) &
        (chunk[columns_to_check] == chunk[columns_to_check].shift(1)).any(axis=1),
        1,
        0
    )
    
    for column in columns_to_check:
        new_col_name = f'same_{column}'
        chunk[new_col_name] = np.where(
            (chunk['new_column'] == 1) &
            (chunk[column].shift(1) == chunk[column]) &
            (chunk[column] == 1) &
            (chunk[column].shift(1) == 1),
            1,
            0
        )
        chunk[column] = chunk[new_col_name]
    
    final_columns = ['date', 'time', 'hour', 'NO', 'V_TYPE', 'lon', 'lat', 'OV_SPD',
                        'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 'Q_LTURN',
                        'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE', 'geometry']
    chunk = chunk[final_columns]
    
    chunk['year'] = chunk['date'].astype(str).str[:4]
    chunk['month'] = chunk['date'].astype(str).str[4:6]
    chunk['day'] = chunk['date'].astype(str).str[6:8]
    
    print(chunk.loc[:,"OV_SPD":"Q_LCHANGE"].sum())
    if chunk.loc[:,"OV_SPD":"Q_LCHANGE"].sum().sum() == 0:
        print("STH WRONGGGGGGGGGGGGGGGGGGGGGGGGG")
    
    # whatyearisit = chunk['year'].unique().astype('int64')
    logging.info(f"process_chunk {len(chunk)}")
    return chunk

@profile
def process_chunk2(chunks):
    """
    Process a chunk of a .gpkg file, applying inIKSAN and transformations.
    Parameters:
    - chunk: GeoDataFrame chunk
    Returns:
    - Processed GeoDataFrame
    """
    logging.info(f"process_chunk START, {len(chunks)}")
    
    #250714 
    chunk = inlocation(chunks,roi_union, roi_box)
    # def inlocation(points, roi_union, roi_box)
    
    #?�산�??�터
    #chunk = inIKSAN(chunk,JLLNEW2)
    
    if chunk.empty:
        return None
    
    # if 'hrs' in chunk.columns:
    #     chunk.rename(columns={'hrs': 'hour'}, inplace=True)
    
    # time_str = chunk['time'].astype(str).str.zfill(8)
    # chunk['hour'] = time_str.str[0:2].astype(int)
    # chunk['min'] = time_str.str[2:4].astype(int)
    # chunk['sec'] = time_str.str[4:6].astype(int)
    # chunk['time'] = time_str.str[:-2]
    # chunk['datetime'] = pd.to_datetime(
    #     chunk['date'].astype(str) + chunk['time'].str.zfill(6),
    #     format='%Y%m%d%H%M%S'
    # )
    
    # chunk = chunk.sort_values(by=["NO",])
    
    # chunk['same_NO'] = (chunk['NO'] == chunk['NO'].shift(1)).astype(int)
    
    # chunk['timediff2'] = chunk['datetime'] - chunk['datetime'].shift(1)
    
    # chunk['less_than_1sec'] = np.where(
    #     (chunk['same_NO'] == 1) &
    #     (chunk['timediff2'].dt.total_seconds() > 0) &
    #     (chunk['timediff2'].dt.total_seconds() <= 2),
    #     1,
    #     0
    # )
    
    # columns_to_check = ['OV_SPD', 'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC',
    #                     'Q_STOP', 'Q_LTURN', 'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE']
    
    # chunk['new_column'] = np.where(
    #     (chunk['less_than_1sec'] == 1) &
    #     (chunk[columns_to_check] == chunk[columns_to_check].shift(1)).any(axis=1),
    #     1,
    #     0
    # )
    
    # for column in columns_to_check:
    #     new_col_name = f'same_{column}'
    #     chunk[new_col_name] = np.where(
    #         (chunk['new_column'] == 1) &
    #         (chunk[column].shift(1) == chunk[column]) &
    #         (chunk[column] == 1) &
    #         (chunk[column].shift(1) == 1),
    #         1,
    #         0
    #     )
    #     chunk[column] = chunk[new_col_name]
    
    # final_columns = ['date', 'time', 'hour', 'NO', 'V_TYPE', 'lon', 'lat', 'OV_SPD',
    #                     'LNG_OVSPD', 'Q_ACC', 'Q_START', 'Q_DEC', 'Q_STOP', 'Q_LTURN',
    #                     'Q_RTURN', 'Q_UTURN', 'Q_OVERTAKE', 'Q_LCHANGE', 'geometry']
    # chunk = chunk[final_columns]
    
    # chunk['year'] = chunk['date'].astype(str).str[:4]
    # chunk['month'] = chunk['date'].astype(str).str[4:6]
    # chunk['day'] = chunk['date'].astype(str).str[6:8]
    
    # print(chunk.loc[:,"OV_SPD":"Q_LCHANGE"].sum())
    # if chunk.loc[:,"OV_SPD":"Q_LCHANGE"].sum().sum() == 0:
    #     print("STH WRONGGGGGGGGGGGGGGGGGGGGGGGGG")
    
    # whatyearisit = chunk['year'].unique().astype('int64')
    logging.info(f"process_chunk {len(chunk)}")
    return chunk

@profile
def countNOs_optimized(chunk,links103):
    logging.info("countNOs_optimized")
    chunk['NO_TRIPID'] = chunk['NO_TRIPID'].astype('category')
    # chunk['timestamp'] = pd.to_datetime(chunk['date'].astype(str) + chunk['time'].str.zfill(6), format='%Y%m%d%H%M%S')
    # chunk['timestamp'] = pd.to_datetime(
    #     chunk['date'].astype(str) + chunk['time'].astype(str).str.zfill(6),
    #     format='%Y%m%d%H%M%S%f'
    # )
    #logging.info("CHECK2")
    chunk['VLM'] = chunk.groupby(['LINK_ID'])['NO_TRIPID'].transform('nunique').astype("Int32")
    # chunk['VLM'] = chunk.groupby(['LINK_ID', 'date', 'time', 'hour',"V_TYPE"])['NO'].transform('nunique').fillna(0).astype(np.int32)
    # chunk['VLM'] = chunk.groupby(['LINK_ID', 'date', 'time', 'hour'])['NO'].transform('nunique').astype(np.int32)
    chunk = chunk[['LINK_ID',"VLM"]]
    chunk = chunk[chunk['LINK_ID'].isin(links103)]
    logging.info("countNOs_optimized_FINISHED")
    chunk = chunk.drop_duplicates()
    chunk.reset_index(drop=True,inplace=True)
    chunk['VLM'] = chunk['VLM'].astype(int)
    return chunk

##########################################################
# ORIGINAL process_vds_and_links(chunk, links, link_tree):
@profile
def process_vds_and_links(chunk, links, link_tree):
    start = time.time()
    """
    Process VDS matching and link assignment for a chunk.
    Parameters:
    - chunk: Processed GeoDataFrame chunk
    - links: GeoDataFrames for links
    - link_tree: STRtrees for spatial queries
    Returns:
    - chunk GeoDataFrame with LINK_ID and VLM
    """
    chunk = chunk.sort_values(by=["NO","V_TYPE", "date", "time", "hour"])
    if chunk is None or chunk.empty:
        return None
    
    # logging.info(f"{chunk.columns}")
    
    if not chunk.empty:
        # chunk['LINK_ID'] = chunk.geometry.apply(
            # lambda x: links.iloc[link_tree.nearest(x)]['LINK_ID'] if link_tree.nearest(x) is not None else None
        # )
        joined_chunk = gpd.sjoin_nearest(chunk, links[['LINK_ID', 'geometry']], how='left', max_distance=250)
        
        # 4. 공간 조인 ???�나???�이 ?�러 링크?� ?�일??거리???�을 경우 중복??발생?????�습?�다.
        # ?�본 'chunk'???�덱?��? 기�??�로 중복???�거?�여 �??�에 ?�나??LINK_ID�??�당?�도�?보장?�니??
        # 'sjoin_nearest'???�해 추�???'index_right' 컬럼?� ?�거?�니??
        if joined_chunk.index.has_duplicates:
            print("THERE ARE DUplicates!!")
            
            duplicated_rows_count = joined_chunk.index.duplicated().sum()
            print(f"Total number of duplicated rows: {duplicated_rows_count}")
            
            unique_indices_with_duplicates = joined_chunk.index.value_counts().gt(1).sum()
            print(f"Number of unique indices that have duplicates: {unique_indices_with_duplicates}")
            
            result = joined_chunk[~joined_chunk.index.duplicated(keep='first')]
        else:
            print("중복?�음....")
            result = joined_chunk # 중복???�으�?그�?�??�용
        
        result = result.drop(columns=['index_right'], errors='ignore')
    #####################
    chunk = countNOs_optimized(result)
    del result
    gc.collect()
    # chunk['VLM'] = chunk.groupby(['LINK_ID', 'date', 'time', 'hour'])['NO'].transform('nunique')
    #####################
    
    logging.info(f"process_vds_and_links {len(chunk)}")
    # print(f"chunkCOLUMNS,{chunk.columns}")
    print(f"{chunk.head()}")
    print(f"Runtime: {time.time() - start} seconds")
    logging.info(check_memory_usage())
    
    return chunk

#!!
@profile
def find_nearest_link_optimized(points, links_gdf, link_tree):
    logging.info(f"{check_memory_usage()} - find_nearest_link_optimized START")
    start = time.time()
    points = points.sort_values(by=["NO_TRIPID","timestamp"])
    if points is None or points.empty:
        return None
    
    """
    Finds the nearest link for each point using STRtree.
    This is often faster than sjoin_nearest for large datasets.
    """
    
    # Use sjoin_nearest with optimized parameters
    result = gpd.sjoin_nearest(
        points, 
        links_gdf[['LINK_ID', 'geometry']],
        how='left',
        max_distance=100,
        distance_col='distance'
    )
    
    #print(result['distance'].dtype)
    
    result.reset_index(drop=True,inplace=True)
    
    #Handle duplicates efficiently
    if result.index.has_duplicates:
        # Keep closest match for each point
        result = result.loc[result.groupby(result.index)['distance'].idxmin()]
        result = result[~result.index.duplicated(keep='first')]
        result.reset_index(drop=True,inplace=True)
    
    # Clean up columns
    result = result.drop(columns=['index_right'], errors='ignore')
    #교통??계산!
    # result_gdf = countNOs_optimized(result)
    result = result.drop_duplicates()
    # result_gdf.reset_index(drop=True,inplace=True)
    result.reset_index(drop=True,inplace=True)
    logging.info(f"find_nearest_link_optimized {len(result)}")
    # print(f"chunkCOLUMNS,{chunk.columns}")
    print(f"{result.head()}")
    print(f"Runtime: {time.time() - start} seconds")
    # logging.info()
    print(check_memory_usage(),"499")
    return result
    
# ALLOCATE per CHUNG's ROI
@profile 
def inCHUNG(points, roi_results, roi_geometries):
    """
    Filter points within polygons in roi_results using STRtree.
    Parameters:
    - points: GeoDataFrame with Point geometries
    - roi_results: Dict of {name: STRtree} for polygon filtering
    - roi_geometries: Dict of {name: Polygon} for within checks
    Returns:
    - Dict of {name: GeoDataFrame} with points within each polygon
    """
    results = {}
    # points = points.to_crs(32652)
    
    for name, roi_tree in roi_results.items():
        # print(name,roi_tree,"CHECK")
        try:
            roi_union = roi_geometries[name]  # Get the corresponding geometry

            #######
            roi_box = box(*roi_union.bounds)
            # points_in_box = points[points.geometry.intersects(roi_box)]
            points_in_box = points[points.geometry.within(roi_box)]

            ########
            points_within2 = gpd.sjoin(
                points_in_box,
                gpd.GeoDataFrame(geometry=[roi_union], crs=32652, index=[0]),
                how="inner",
                predicate="within"
            ).drop(columns=['index_right'])
            # points_within2 = points_in_box[points_in_box.geometry.within(roi_union)]
            
            # if not points_within.empty:
            points_within2['CHUNG'] = name.split("_")[0]
            # logging.info(f"{len(points_within2)} points within {name}")
            results[name] = points_within2
            # else:
            #     logging.info(f"No points within {name}")
                
        except Exception as e:
            logging.error(f"Error processing {name}: {str(e)}")
            continue
            
    return results

def uppercase_cols_except_geom(gdf):
    """지?�메?�리 ?�을 ?�외??모든 ???�름???�문자�?변??""
    # geopandas GeoDataFrame?��? ?�인
    if isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry is not None:
        geometry_col = gdf.geometry.name
        gdf.columns = [col if col == geometry_col else col.upper() for col in gdf.columns]
    else:
        # pandas DataFrame??경우 그냥 모든 ???�름 ?�문자�?변??        gdf.columns = [col.upper() for col in gdf.columns]
    
    return gdf

def create_date_hour_df(start_year='{y}', end_year='{y}'):
    """
    Create a DataFrame with date and hour columns for the specified years.
    
    Parameters:
    - start_year: Starting year (default: 2022)
    - end_year: Ending year (default: 2024)
    
    Returns:
    - DataFrame with 'date' (YYYY-MM-DD) and 'hour' (0-23) columns
    """
    # Generate all dates from start_year-01-01 to end_year-12-31
    dates = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D",
        tz="Asia/Seoul"
    )
    
    # Create a list of hours (0 to 23)
    hours = list(range(24))
    
    # Create a DataFrame with all combinations of dates and hours
    df = pd.DataFrame(
        [(d, h) for d in dates for h in hours],
        columns=['date', 'hour']
    )
    
    # Convert date to YYYY-MM-DD format (optional, as date_range already provides datetime)
    df['date'] = df['date'].dt.strftime('%Y%m%d')
    
    return df

def removefromisland(links, CHUNG_ROI):
    
    #?�주ROI ?�당 link?�거
    JEJUROI = CHUNG_ROI[CHUNG_ROI['CHUNG']=="JEJU"]
    # links = gpd.GeoDataFrame(links, geometry='geometry', crs=32652)

    print("JEJU ?�당 LINK ?�거", len(links[links.geometry.within(JEJUROI.geometry.iloc[0])]),'�?)
    links = links[~links.geometry.within(JEJUROI.geometry.iloc[0])]

    #?�도ROI ?�당 link?�거
    DOKDOROI = CHUNG_ROI[CHUNG_ROI['CHUNG']=="DOKDO"]
    print("?�도 ?�당 LINK ?�거", len(links[links.geometry.within(DOKDOROI.geometry.iloc[0])]),'�?)
    links = links[~links.geometry.within(DOKDOROI.geometry.iloc[0])]
    # Validate and rename geometry column
    print("링크개수", links.shape)
    return links

def filter_files_from_chunk(files, fromfile):
    if isinstance(fromfile, str): 
        # Find the index of the file that contains the fromfile string
        start_index = next((i for i, f in enumerate(files) if fromfile in f), None)
    
        if start_index is None:
            print(f"Warning: {fromfile} not found in the file list")
            return files  # Return empty list if fromfile not found
    
        print(f"Found {fromfile} at index {start_index}")
        print(f"Removing {start_index} files from the beginning")
    
    return files[start_index:]

def extract_chunk_number(file_path):
    # ?? '2024_0_chunk_000.gpkg'?�서 '000' 추출
    base = os.path.basename(file_path)
    m = re.search(r'chunk_(\d+)', base)
    if m:
        return m.group(1)
    else:
        return "NA"
    
def extract_chunk_numbers(file_path):
        # ?? '2024_0_chunk_000.gpkg'?�서 '000' 추출
    base = os.path.basename(file_path)
    m = re.search(r'(\d+)_chunk', base)
    if m:
        return m.group(1)
    else:
        return "NA"

def tripsbtw15(gdf):
    # Ensure data is sorted
    gdf = gdf.sort_values(by=['NO', 'timestamp'])
    
    # Calculate the difference in time from the previous point for each car
    gdf['time_diff'] = gdf.groupby('NO')['timestamp'].diff()
    
    # Define the gap that separates trips (e.g., 1 hour)
    gap_threshold = pd.Timedelta(minutes=15)
    # gap_threshold = pd.Timedelta(hours=2)
    # Create a boolean column that is True at the start of every new trip.
    
    # The first point for a car (where time_diff is NaT) is also the start of a trip.
    gdf['is_new_trip'] = gdf['time_diff'] > gap_threshold

    # Use cumsum() within each group. This is faster than .apply().
    # It increments the trip_id every time it encounters 'True' in the 'is_new_trip' column.
    gdf['trip_id'] = gdf.groupby('NO')['is_new_trip'].cumsum()
    # A new trip starts if the time_diff is larger than our threshold.
    # The cumsum() function creates a unique, incrementing ID for each trip segment per car.
    # gdf['trip_id'] = gdf.groupby('NO')['time_diff'].apply(lambda x: (x > gap_threshold).cumsum())
    gdf['NO_TRIPID'] = gdf['NO'].astype(str) + '_' + gdf['trip_id'].astype(str)

    # Group by the unique trip identifier
    grouped_trips = gdf.groupby(['NO', 'trip_id'])
    
    # Calculate the duration of each trip
    trip_durations = grouped_trips['timestamp'].agg(lambda x: x.max() - x.min()).rename('duration')
    
    # Identify the trips that are 2 hours or longer
    min_duration = pd.Timedelta(hours=2)
    long_trips = trip_durations[trip_durations >= min_duration]
    
    # Use the index of long_trips to filter the original DataFrame
    # We need to get the ('NO', 'trip_id') pairs into a list
    long_trip_indices = long_trips.index.tolist()
    
    # Filter the original gdf to keep only rows that are part of a long trip
    # This creates a boolean mask for filtering
    is_long_trip = gdf.set_index(['NO', 'trip_id']).index.isin(long_trip_indices)
    gdf_filtered = gdf[is_long_trip].copy()
    gdf_filtered = gdf_filtered.drop(columns=['time_diff', 'is_new_trip', 'trip_id',"NO"])
    
    # Add this line to create the unique trip ID
    # Optional: Clean up helper columns
    # gdf_filtered = gdf_filtered.drop(columns=['time_diff', 'trip_id'])
    # final_gdf = final_gdf.drop(columns=['cutoff_time'])
    
    gdf= gdf[['date', 'time', 'NO_TRIPID', 'geometry', 'timestamp']]
    gdf= gdf.drop_duplicates()
    return gdf_filtered, gdf

def eraseunder2hoursandhalf(gdf_filtered):
    # Ensure data is sorted
    gdf_filtered = gdf_filtered.sort_values(by=['NO_TRIPID', 'timestamp'])
    
    # Add this line to create the unique trip ID
    # Optional: Clean up helper columns
    # gdf_filtered = gdf_filtered.drop(columns=['time_diff', 'trip_id'])
    
    trip_start_times = gdf_filtered.groupby('NO_TRIPID')['timestamp'].min()
    # two_hours = pd.Timedelta(hours=2)
    two_hoursandhalf = pd.Timedelta(minutes=150)
    trip_cutoff_times = trip_start_times + two_hoursandhalf
    gdf_filtered['cutoff_time'] = gdf_filtered['NO_TRIPID'].map(trip_cutoff_times)
    final_gdf = gdf_filtered[gdf_filtered['timestamp'] >= gdf_filtered['cutoff_time']].copy()

    # Optional: You can now drop the helper column
    final_gdf = final_gdf.drop(columns=['cutoff_time'])
    
    return final_gdf

#%%
from haversine import haversine, Unit
from scipy.spatial.distance import pdist, squareform  # pairwise 거리 계산??(?�경 지??

def remove_gps_scribble(gdf, min_duration_minutes=15, stop_radius_meters=350):
    """
    # [EPSG:32652 최적?? ?�정 ?�간 ?�상, ?�정 반경 ?�에 머무???�차 ?�이?��? ?�거?�니??
    # haversine ?�???�클리드 거리�??�용?�고, ?�산 ?�도�?최적?�했?�니??

    # :param gdf: GeoDataFrame with ['NO', 'timestamp', 'geometry'] (CRS: EPSG:32652)
    # :param min_duration_minutes: 최소 ?�차 ?�간 (�?
    # :param stop_radius_meters: 최�? ?�차 반경 (미터)
    # :return: ?�기 ?�차 ?�이?��? ?�거??GeoDataFrame
    """
    if gdf.empty:
        return gdf
    
    ll = len(gdf)
    print(f"?�거 ???�본 ?�이???? {ll}�?)
    
    gdf = gdf.sort_values(by=['NO', 'timestamp']).copy()
    
    # --- 1. ?�도 기반 '?��?' ?�심 ?�인???�별 (최적?�된 방식) ---
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    gdf['time_delta_sec'] = gdf.groupby('NO')['timestamp'].diff().dt.total_seconds()
    
    # ?�클리드 거리 계산 (벡터???�산)
    delta_x = gdf['lon'] - gdf.groupby('NO')['lon'].shift()
    delta_y = gdf['lat'] - gdf.groupby('NO')['lat'].shift()
    gdf['distance_m'] = np.hypot(delta_x, delta_y).fillna(0)
    
    gdf['speed_kmh'] = (gdf['distance_m'] / gdf['time_delta_sec'] * 3.6).fillna(0)
    gdf['is_stopped'] = gdf['speed_kmh'] < 5
    
    # --- 2. ?�속??'?��?' ?�태�?블록?�로 그룹??---
    gdf['stop_block'] = (gdf['is_stopped'] != gdf['is_stopped'].shift()).cumsum()
    
    stopped_gdf = gdf[gdf['is_stopped']].copy()
    if stopped_gdf.empty:
        print("?�거???�기 ?�차 구간???�습?�다.")
        # ?�시 컬럼 ?�리 ???�본 반환
        return gdf.drop(columns=['lon', 'lat', 'time_delta_sec', 'distance_m', 'speed_kmh', 'is_stopped', 'stop_block'])

    # --- 3. �?'?��?' 블록??지???�간 �?공간??범위 분석 (최적?�된 방식) ---
    stopped_blocks_grouped = stopped_gdf.groupby('stop_block')
    
    # 블록�?지???�간 계산
    block_durations = stopped_blocks_grouped['timestamp'].agg(lambda x: x.max() - x.min())
    
    # 블록�?공간??범위(최�? 반경) 계산 (for 루프 ?�거)
    block_centroids = stopped_blocks_grouped.agg({'lon': 'mean', 'lat': 'mean'})
    block_centroids.rename(columns={'lon': 'lon_centroid', 'lat': 'lat_centroid'}, inplace=True)
    
    # ?�본 ?�이?�에 �?블록??중심??좌표�?매핑
    stopped_gdf = stopped_gdf.merge(block_centroids, on='stop_block', how='left')
    
    # �??�인?��? ?�당 블록??중심???�이??거리 계산 (벡터??
    dist_from_centroid_x = stopped_gdf['lon'] - stopped_gdf['lon_centroid']
    dist_from_centroid_y = stopped_gdf['lat'] - stopped_gdf['lat_centroid']
    stopped_gdf['dist_to_centroid'] = np.hypot(dist_from_centroid_x, dist_from_centroid_y)
    
    # �?블록�?최�? 반경 계산
    block_max_radiuses = stopped_gdf.groupby('stop_block')['dist_to_centroid'].max()

    # --- 4. ?�거??블록 ID ?�별 ---
    to_remove_ids = block_durations[
        (block_durations >= pd.Timedelta(minutes=min_duration_minutes)) &
        (block_max_radiuses <= stop_radius_meters)
    ].index
    
    # --- 4-1. ?�버�? ?�거?��? ?��? 블록???�인 분석 ---
    # 모든 ?�차 블록???�보(지?�시�? 최�?반경)�??�친??    block_info = pd.concat([block_durations, block_max_radiuses], axis=1)
    block_info.columns = ['duration', 'max_radius']
    
    # ?�거 조건 ?�정
    duration_condition = block_info['duration'] >= pd.Timedelta(minutes=min_duration_minutes)
    radius_condition = block_info['max_radius'] <= stop_radius_meters
    
    # ?�깝�??�거?��? ?��? 블록???�별
    # Case 1: ?�간?� 길었지�? 반경???�무 ?�었??블록
    almost_removed_large_radius = block_info[duration_condition & ~radius_condition]
    if not almost_removed_large_radius.empty:
        print("\n[?�버�? ?�간?� 충분?��?�?반경???�어 ?�거?��? ?��? 블록:")
        print(almost_removed_large_radius)
    
    # Case 2: 반경?� 좁았지�? ?�간??부족했??블록
    almost_removed_short_duration = block_info[~duration_condition & radius_condition]
    if not almost_removed_short_duration.empty:
        print("\n[?�버�? 반경?� 좁았지�??�간??부족해 ?�거?��? ?��? 블록:")
        print(almost_removed_short_duration)
        
    # --- 5. ?�본 ?�이?�에???�당 블록 ?�거 ---
    to_remove_ids = block_durations[duration_condition & radius_condition].index

    gdf_cleaned = gdf[~gdf['stop_block'].isin(to_remove_ids)].copy()
    
    # ?�거??차량 ?�보 출력 (?�버깅용)
    removed_data = gdf[gdf['stop_block'].isin(to_remove_ids)].copy()
    # removed_data.to_parquet(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\removed_data.geoparquet", engine='pyarrow', compression='snappy', index=False)
    if not removed_data.empty:
        print("?�거??차량 ID �??�이????")
        print(removed_data['NO'].value_counts())
    
    print(f"Scribble ?�이�??�거 ???�이???? {len(gdf_cleaned)}")
    if ll == len(gdf_cleaned):
        print("?�거???�이?��? ?�습?�다. (ITS THE SAME)")
    
    # ?�시 컬럼 ?�리
    cols_to_drop = ['lon', 'lat', 'time_delta_sec', 'distance_m', 'speed_kmh', 'is_stopped', 'stop_block']
    gdf_cleaned = gdf_cleaned.drop(columns=cols_to_drop)
    
    return gdf_cleaned

def remove_gps_scribble_improved(gdf, min_duration_minutes=15, stop_radius_meters=250, speed_stop_threshold=5, speed_slow_threshold=20):
    
    # [EPSG:32652 최적?? ?�정 ?�간 ?�상, ?�정 반경 ?�에 머무???�차 ?�이?��? ?�거?�니??
    # 개선: ?�도 ?�계 강화, ?�린 ?�인???�함 블록?? pairwise max distance�?반경 ?�확????

    # :param gdf: GeoDataFrame with ['NO', 'timestamp', 'geometry'] (CRS: EPSG:32652)
    # :param min_duration_minutes: 최소 ?�차 ?�간 (�?
    # :param stop_radius_meters: 최�? ?�차 반경 (미터)
    # :param speed_stop_threshold: ?��? ?�도 ?�계 (km/h, 기본 5)
    # :param speed_slow_threshold: ?�린 ?�도 ?�계 (km/h, 기본 20, 블록 ?�장??
    # :return: ?�기 ?�차 ?�이?��? ?�거??GeoDataFrame
    
    if gdf.empty:
        return gdf
    
    ll = len(gdf)
    print(f"?�거 ???�본 ?�이???? {ll}�?)
    
    gdf = gdf.sort_values(by=['NO', 'timestamp']).copy()
    
    # --- 1. ?�도 기반 '?��?/?�림' ?�인???�별 (개선: ?�린 ?�인???�함) ---
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    gdf['time_delta_sec'] = gdf.groupby('NO')['timestamp'].diff().dt.total_seconds()
    
    # ?�클리드 거리 계산 (벡터??
    delta_x = gdf['lon'] - gdf.groupby('NO')['lon'].shift()
    delta_y = gdf['lat'] - gdf.groupby('NO')['lat'].shift()
    gdf['distance_m'] = np.hypot(delta_x, delta_y).fillna(0)
    
    gdf['speed_kmh'] = (gdf['distance_m'] / gdf['time_delta_sec'] * 3.6).fillna(0)
    gdf['is_stopped'] = gdf['speed_kmh'] < speed_stop_threshold
    gdf['is_slow'] = gdf['speed_kmh'] < speed_slow_threshold  # 개선: ?�린 ?�인??추�?
    
    # --- 2. ?�속??'?��?/?�림' ?�태�?블록?�로 그룹??(개선: is_slow�??�장) ---
    gdf['stop_block'] = (gdf['is_slow'] != gdf['is_slow'].shift()).cumsum()  # ?�린 ?�태 변?�로 블록??    
    slow_gdf = gdf[gdf['is_slow']].copy()  # ?��?�??�닌 ?�린 ?�함
    if slow_gdf.empty:
        print("?�거???�기 ?�차 구간???�습?�다.")
        return gdf.drop(columns=['lon', 'lat', 'time_delta_sec', 'distance_m', 'speed_kmh', 'is_stopped', 'is_slow', 'stop_block'])

    # --- 3. �?블록??지???�간 �?공간??범위 분석 (개선: pairwise max dist) ---
    slow_blocks_grouped = slow_gdf.groupby('stop_block')
    
    # 블록�?지???�간
    block_durations = slow_blocks_grouped['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds() / 60)  # �??�위�?    
    # 블록�??�인??좌표 배열 준�?(scipy pdist??
    def calc_max_radius(group):
        if len(group) < 2:
            return 0
        coords = np.column_stack([group['lon'], group['lat']])
        pairwise_dists = squareform(pdist(coords, 'euclidean'))  # 모든 ??거리
        return np.max(pairwise_dists)  # 최�? 거리 (직경, 반경 근사)
    
    block_max_radiuses = slow_blocks_grouped.apply(calc_max_radius)  # 그룹�?max dist

    # --- 4. ?�거??블록 ID ?�별 (?�간: �??�위) ---
    to_remove_ids = block_durations[
        (block_durations >= min_duration_minutes) &
        (block_max_radiuses <= stop_radius_meters * 2)  # 직경?��?�?*2 (반경 근사)
    ].index
    
    # --- 5. ?�본 ?�이?�에???�당 블록 ?�거 ---
    gdf_cleaned = gdf[~gdf['stop_block'].isin(to_remove_ids)].copy()
    
    # ?�거???�이???�버�?(try-except 추�?)
    removed_data = gdf[gdf['stop_block'].isin(to_remove_ids)].copy()
    try:
        # removed_data.to_parquet(r"D:\NIPA_GIT\TAMSPython\TAMSPython\JB\removed_data22.geoparquet", engine='pyarrow', compression='snappy', index=False)
        print("?�거???�이???�???�료.")
    except Exception as e:
        print(f"?�???�패: {e}")
    
    if not removed_data.empty:
        print("?�거??차량 ID �??�이????")
        # print(removed_data['NO'].value_counts())
        removal_ratio = len(removed_data) / ll * 100
        print(f"?�거 비율: {removal_ratio:.2f}%")
    
    print(f"Scribble ?�이�??�거 ???�이???? {len(gdf_cleaned)}")
    if ll == len(gdf_cleaned):
        print("?�거???�이?��? ?�습?�다. (?�계�?조정 추천)")
    
    # ?�시 컬럼 ?�리
    cols_to_drop = ['lon', 'lat', 'time_delta_sec', 'distance_m', 'speed_kmh', 'is_stopped', 'is_slow', 'stop_block']
    gdf_cleaned = gdf_cleaned.drop(columns=cols_to_drop)
    
    return gdf_cleaned

