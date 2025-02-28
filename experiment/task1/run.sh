#! /bin/bash 

source ~/.bashrc 

conda activate lpcode

date 

for lang in "c" "cpp" "java" "py"
do 
    python main.py --lang $lang
done

date 

rm -rf __pycache__