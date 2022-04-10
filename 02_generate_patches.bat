setlocal
@call 00_own_model_Name.bat
for /f %%f in ('dir /b %modelName%_Patches') do (
echo %%f
..\python.exe create_patches.py --input %modelName%_Patches\%%f --outputFolder %modelName%_Patches --channel %channel%
)