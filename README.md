# JB DTG Processing

占쏙옙 占쏙옙占쏙옙年占?DTG/占쏙옙占쏙옙 占쏙옙트占쏙옙크 占쏙옙占쏙옙占쏙옙 처占쏙옙, 占쏙옙占쏙옙, 占쏙옙占쏙옙 占시곤옙화 占쏙옙占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙크占쏙옙트占쏙옙 占쏙옙占쏙옙求占?

## 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙

- `processingDTGJB.py`: 占쌕쏙옙 처占쏙옙 占쏙옙틸 占쏙옙占?- `processingDTGJB2.py`: `processingDTGJB` 占식삼옙/占쏙옙체 占쏙옙占?- `untitled1.py`, `untitled2.py`: DTG 처占쏙옙 占쏙옙占쏙옙 占쏙옙크占쏙옙트
- `q3.py`, `q4.py`: 占쏙옙占쏙옙/占쏙옙처占쏙옙 占쏙옙占쏙옙 占쏙옙크占쏙옙트
- `plot5.py`, `plot6.py`: 占쏙옙占쏙옙 占시곤옙화 占쏙옙크占쏙옙트
- `JBROI.*`, `JBROI2.*`, `roi_box.gpkg`: 占쏙옙占쏙옙/ROI 占쏙옙占쏙옙占쏙옙
- `bfg-1.15.0.jar`: Git 占쏙옙占쏙옙占썰리 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙
- `scripts/`: 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙

## 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占?
占쏙옙占싸울옙 占쏙옙占쏙옙占쏙옙(占쏙옙占쏙옙):

- `python scripts/run_dtg_pipeline.py`
- `python scripts/run_dtg_pipeline_alt.py`
- `python scripts/run_q3.py`
- `python scripts/run_q4.py`
- `python scripts/plot_dtg_links.py`
- `python scripts/plot_dtg_links_alt.py`

占쏙옙占신쏙옙 占쏙옙占쏙옙 占쏙옙占?占쏙옙占쏙옙):

- `python untitled2.py`
- `python untitled1.py`
- `python q3.py`
- `python q4.py`
- `python plot5.py`
- `python plot6.py`

占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙크占쏙옙트占쏙옙 占쌓댐옙占?占쏙옙占쏙옙占쌌니댐옙. 占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙크占쏙옙트占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙체占싹몌옙 占싯니댐옙.

## 占쏙옙占쏙옙 占쌔븝옙

1. Python 占쏙옙키占쏙옙 占쏙옙치
   ```bash
   pip install -r requirements.txt
   ```
2. 占쏙옙크占쏙옙트 占쏙옙 占싹듸옙占쌘듸옙占쏙옙 占쌉뤄옙 占쏙옙占?占쏙옙: `/data1/...`, DB 占쏙옙占쏙옙 占쏙옙占쏙옙)占쏙옙 환占썸에 占승곤옙 占쏙옙占쏙옙
3. 큰 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙 Git 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙都求占? 占쏙옙占?환占썸에占쏙옙 占쏙옙寬占?占쏙옙효占쏙옙占쏙옙 확占쏙옙

## 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙

- 占쏙옙占쏙옙占쏙옙/占쏙옙占쏙옙 占쏙옙占쏙옙占쏙옙 `data/`, `tools/`占쏙옙 占쏙옙占쏙옙 占싱듸옙
- 占쏙옙占쏙옙 占쌉쇽옙 占쏙옙占쏙옙占쏙옙 占쏙옙占싫?옙構占? `scripts/` 占쏙옙占쏙옙占쏙옙占쏙옙占쏙옙 `argparse`占쏙옙 占심쇽옙 占쏙옙占쏙옙
- 환占썸별 占쏙옙灌占?`.env` 占실댐옙 占쏙옙占쏙옙 占쏙옙占싹뤄옙 占싻몌옙

## 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙

- 占쌕쏙옙 占쌘듸옙 占쏙옙占쏙옙占쏙옙 占싣뤄옙 占쏙옙占쏙옙占쏙옙占?占쏙옙占쏙옙
  - `python scripts/cli.py pipeline`
  - `python scripts/cli.py pipeline-alt`
  - `python scripts/cli.py q3`
  - `python scripts/cli.py q4`
  - `python scripts/cli.py plot`
  - `python scripts/cli.py plot-alt`

## 占쏙옙티占쏙옙트 占쏙옙占쏙옙(占쏙옙占쏙옙 占쌥울옙占쏙옙)

- 占쏙옙占쏙옙 占쏙옙占싱너몌옙/占쏙옙占쏙옙占싶댐옙 占쏙옙占쏙옙 占쏙옙管占?占싻몌옙

  - `data/` : `JBROI*`, `roi_box.gpkg`
  - `tools/` : `bfg-1.15.0.jar`
- 占쏙옙占쏙옙 占쏙옙크占쏙옙트占쏙옙 占쏙옙占쏙옙 占쏙옙占?占쌘듸옙 占쌔삼옙(`project_paths.py`)占쏙옙 `data/`占쏙옙 占쎌선 탐占쏙옙占싹듸옙占쏙옙 占쏙옙占쏙옙占쏙옙



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
