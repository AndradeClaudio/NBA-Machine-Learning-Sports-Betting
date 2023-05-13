# RunPythonScripts.ps1

$pythonExe = "python"
$scripts = @(
    @{
        Path = "C:\Python\NBA\NBA-Machine-Learning-Sports-Betting\src\Process-Data"
        Script = "Get_Data"
    },
    @{
        Path = "C:\Python\NBA\NBA-Machine-Learning-Sports-Betting\src\Process-Data"
        Script = "Get_Odds_Data"
    },
    @{
        Path = "C:\Python\NBA\NBA-Machine-Learning-Sports-Betting\src\Process-Data"
        Script = "Create_Games"
    },
    @{
        Path = "C:\Python\NBA\NBA-Machine-Learning-Sports-Betting\src\Train-Models"
        Script = "XGBoost_Model_ML"
    },
    @{
        Path = "C:\Python\NBA\NBA-Machine-Learning-Sports-Betting\src\Train-Models"
        Script = "XGBoost_Model_UO"
    }
)

foreach ($script in $scripts) {
    Set-Location $script.Path
    Write-Host "Executando o script: $($script.Script)"
    & $pythonExe -m $script.Script

    if ($LASTEXITCODE -ne 0) {
        Write-Host "O script '$($script.Script)' encontrou um erro. Código de saída: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "Todos os scripts foram executados com sucesso." -ForegroundColor Green
