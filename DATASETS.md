# Dataset Layout

The old single-dataset project used `kongzhaopu`.  The current directory is organized for multi-dataset training:

```text
datasets/
  <dataset_name>/
    solar.csv
    rt_text1.jsonl
    rt_text2.jsonl
```

The dataset directory is kept at the project root as `datasets/` to avoid compound folder titles in the file explorer.

## Available Datasets

```text
adong
adong_2026
aping
daapeng
hebei_station00
hebei_station01
hebei_station02
hebei_station03
hebei_station04
hebei_station05
hebei_station06
hebei_station07
hebei_station08
hebei_station09
jiuzhai
kongzhaopu
yanfengdong
```

## Code Interface

Default single-dataset run:

```bash
python train_solar.py --mode paper --dataset_name kongzhaopu
```

Equivalent explicit paths:

```bash
python train_solar.py \
  --mode paper \
  --solar_csv ./datasets/kongzhaopu/solar.csv \
  --text1_path ./datasets/kongzhaopu/rt_text1.jsonl \
  --text2_path ./datasets/kongzhaopu/rt_text2.jsonl
```

Multi-dataset run:

```bash
python train_solar.py \
  --mode paper \
  --datasets kongzhaopu,adong,aping,daapeng,jiuzhai,yanfengdong
```

Multi-dataset mode does not concatenate raw time series.  Each station keeps an independent chronological split and independent text alignment.  The numeric scalers are fitted on the union of all training partitions.
