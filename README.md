A Study on the Effectiveness Analysis of Rest Areas Using Continuous Driving Time Variable
(연속운전 데이터를 활용한 졸음쉼터 효과분석)
이 저장소는 **한국교통안전공단(KOTSA)**의 의뢰를 받아 수행한 "화물차 연속운전 데이터 분석 및 졸음쉼터 입지 효과 분석"을 위한 파이프라인을 포함하고 있습니다.
This repository contains a data processing pipeline for analyzing truck continuous driving data and the effectiveness of rest area locations, commissioned by the Korea Transportation Safety Authority (TSAT), a national public agency under the Ministry of Land, Infrastructure and Transport of the Republic of Korea.

📊 Analysis Overview (분석 개요)
1. Data Collection & Processing (데이터 수집 및 가공)
Target Area: Standard road network links corresponding to National Highways 
(General National Roads) within Jeollabuk-do

Period: March 2025 – August 2025 (6 months)

Data Source: Truck Digital Tachograph (DTG) data, including GPS coordinates, vehicle IDs, and driving timestamps

Data column configuration:
 - 칼럼순서: 운행일자	운행시분초	차량번호	업종	GPSX	GPSY
 - column order: date	 time	carid	V_TYPE	lon	lat

Methodology: Extracted continuous driving durations and trips per vehicle and matched them to spatial road network links
 - 차량번호를 이용한 trip 생성 시, 데이터 탐색 : EDA for making trips using Carid.
 - 다음 그림은 특정차량(차량번호: 4347)이 20250331에 주행한 궤적을 나타냄. 오전 9시33분경부터 위경도값 측정에 오류가 있는 것으로 추정됨

![image.png](attachment:f818459c-8843-4ae9-9dba-fb98bff2725b:image.png)

 - 다음 그림은 정차한 것으로 추정되는 주행궤적을 제거한 것을 나타냄(적색 point 는 제거 대상)
 - 15분이상 250m 반경 내에서 5km/h 이하의 속도를 나타내고 있거나 20km/h 이하의 속도를 나타내는 point를 군집화하여 제거함
![image.png](attachment:c4efa29c-4515-4c15-a644-3fcbe7c93388:image.png)

 - 다음 그림은 전체 데이터 일부를 plotting 하였을 때, 새만금 구간의 일부 도로 구역(노란색 칠해진 링크)은 데이터가 없는 상황

![image.png](attachment:d4ffde21-948f-437f-8b40-e4f6808e1d73:image.png)

 - 다음 그림은 새만금 구간의 일부 도로 구역(노란색 칠해진 링크)을 통과하는 것으로 추정되는데 주행궤적이 기록되지 않은 현황

![image.png](attachment:c30f1c73-8b2f-4381-9ba1-83646b1833d1:image.png)

2. Key Statistics (주요 통계)
Traffic Volume: Max 413,661 trucks per link (Avg. 24,513)

Long-duration Driving (>2 hours): Max 29,586 trucks (59.7% of link traffic), Avg. 2,233 trucks (11.5%)

Extreme-duration Driving (>2.5 hours): Max 11,691 trucks (53.5%), Avg. 1,100 trucks (7.16%)

🗺️ Visualization Results (시각화 결과)
1. Regional Traffic Density (권역별 교통량 시각화)
High-Traffic Route: National Route 21, passing through Gunsan, Iksan, and Jeonju, showed the highest truck traffic volume

Critical Link: The road link in Oksan-myeon, Gunsan-si, was identified as the segment with the highest frequency of continuous driving exceeding the safety threshold

## 현재 폴더 구성
- `processingDTGJB.py`: 핵심 처리 유틸리티 모듈
- `processingDTGJB2.py`: `processingDTGJB`의 대체/개선 버전
- `untitled1.py`, `untitled2.py`: DTG 처리 실행용 스크립트
- `q3.py`, `q4.py`: 집계 및 후처리 실행 스크립트
- `plot5.py`, `plot6.py`: 지도 시각화 스크립트
- `JBROI.*`, `JBROI2.*`, `roi_box.gpkg`: ROI/지역 경계 데이터
- `bfg-1.15.0.jar`: Git 큰 파일 제거용 도구
- `scripts/`: 통합 실행 진입점

## 실행 경로 추천
- `python scripts/run_dtg_pipeline.py`
- `python scripts/run_dtg_pipeline_alt.py`
- `python scripts/run_q3.py`
- `python scripts/run_q4.py`
- `python scripts/plot_dtg_links.py`
- `python scripts/plot_dtg_links_alt.py`

## 기존 실행 경로
- `python untitled2.py`
- `python untitled1.py`
- `python q3.py`
- `python q4.py`
- `python plot5.py`
- `python plot6.py`

## 변경 사유 및 운영 방식
- 새 진입점으로 기존의 레거시 스크립트를 호출합니다.
- 기존 스크립트는 점진적으로 정리하면서 경로 관리와 실행 옵션을 통일해가고 있습니다.

## 설치 및 환경 설정
1. Python 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 스크립트 내부의 하드코딩 경로(예: `/data1/...`)와 DB 연결 정보를 실제 환경에 맞게 수정하세요.
3. Git에는 대용량 데이터/바이너리를 포함하지 않으므로, 로컬 환경의 `data/`, `tools/` 경로가 유효한지 확인하세요.

## 정리 예정 항목
- 데이터/도구 파일을 `data/`, `tools/`로 분리해 보관하기
- `scripts/`에서 `argparse` 기반 공통 인터페이스 정비하기
- 환경별 경로를 `.env` 또는 별도 설정 파일로 분리하기

## 통합 CLI 실행
- `python scripts/cli.py pipeline`
- `python scripts/cli.py pipeline-alt`
- `python scripts/cli.py q3`
- `python scripts/cli.py q4`
- `python scripts/cli.py plot`
- `python scripts/cli.py plot-alt`

## 산출물 구성(현재 반영)
- 큰 바이너리/데이터 분리
  - `data/` : `JBROI*`, `roi_box.gpkg`
  - `tools/` : `bfg-1.15.0.jar`
- 데이터 경로 로더는 `project_paths.py`를 통해 `data/` 기준으로 해결됩니다.


![alt text](20260215_042620.png)
## English Translation

This repository contains scripts for DTG/traffic network data processing, aggregation, and map visualization pipelines.

## Current Folder Structure
- `processingDTGJB.py`: Core processing utility module
- `processingDTGJB2.py`: Derived/alternative module of `processingDTGJB`
- `untitled1.py`, `untitled2.py`: DTG processing runner scripts
- `q3.py`, `q4.py`: Aggregation and post-processing runner scripts
- `plot5.py`, `plot6.py`: Map visualization scripts
- `JBROI.*`, `JBROI2.*`, `roi_box.gpkg`: Area/ROI data
- `bfg-1.15.0.jar`: Tool for cleaning Git history
- `scripts/`: Consolidated entry points

## Recommended Run Paths

Recommended entry points:
- `python scripts/run_dtg_pipeline.py`
- `python scripts/run_dtg_pipeline_alt.py`
- `python scripts/run_q3.py`
- `python scripts/run_q4.py`
- `python scripts/plot_dtg_links.py`
- `python scripts/plot_dtg_links_alt.py`

Legacy run paths:
- `python untitled2.py`
- `python untitled1.py`
- `python q3.py`
- `python q4.py`
- `python plot5.py`
- `python plot6.py`

The new entry points execute the existing legacy scripts directly. In later versions, you can gradually replace only the underlying target script files referenced by each entry point.

## Setup
1. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Update hardcoded input paths in scripts (for example, `/data1/...`) and database connection settings to match your environment
3. Large data files are excluded from source control. Make sure paths are valid in your environment.

## Planned Cleanup
- Move data and tool files to `data/` and `tools/`
- Refactor shared logic into common utility modules, and standardize command options using `argparse` in `scripts/` entry points
- Externalize environment-specific paths into `.env` or a dedicated configuration file

## Consolidated Run Flow
- Prefer running core code through:
  - `python scripts/cli.py pipeline`
  - `python scripts/cli.py pipeline-alt`
  - `python scripts/cli.py q3`
  - `python scripts/cli.py q4`
  - `python scripts/cli.py plot`
  - `python scripts/cli.py plot-alt`

## Artifact Organization (Currently Applied)
- Large binaries and data are separated:
  - `data/`: `JBROI*`, `roi_box.gpkg`
  - `tools/`: `bfg-1.15.0.jar`
- Existing scripts were updated to use `project_paths.py`, which now resolves data files with `data/` prioritized automatically.

🏛️ Acknowledgement
This project was developed for the Korea Transportation Safety Authority (TSAT). As a national public agency, TSAT focuses on enhancing road safety and reducing traffic accidents through data-driven research.

본 프로젝트는 국토교통부 산하 국가공공기관인 한국교통안전공단의 위험주행행동(DTG) 데이터를 바탕으로 수행되었습니다.
