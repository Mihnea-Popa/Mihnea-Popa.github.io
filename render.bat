@echo off
echo Cleaning output folders with PDF resources...
if exist "docs\quant_research\monte_carlo_thorisson" rmdir /s /q "docs\quant_research\monte_carlo_thorisson"
if exist "docs\quant_research\bsm_monte_carlo" rmdir /s /q "docs\quant_research\bsm_monte_carlo"
if exist "docs\quant_research\tipe_somnolence" rmdir /s /q "docs\quant_research\tipe_somnolence"
if exist "docs\socio_economic_research" rmdir /s /q "docs\socio_economic_research"
echo Done. Starting quarto render...
quarto render
echo.
echo Render complete. Running git add, commit, push...
git add .
git commit -m "Update site"
git push
echo.
echo All done!
pause
