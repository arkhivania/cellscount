setlocal
@call 00_own_model_Name.bat
mkdir %modelName%
..\python.exe train.py --trainData %modelName%_Patches --outputModelFolder %modelName%