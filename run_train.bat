@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python train.py --data_root data --epochs 30 --batch_size 16 --no_fusion
pause
