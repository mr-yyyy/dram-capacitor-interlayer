$env:PYTHONIOENCODING = "utf-8"
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
Set-Location $PSScriptRoot
"CTE START $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
foreach ($m in @("AlN","HfN","ZrN","TaN_cubic","NbN_cubic","TiC","ZrO2_tet","ZrO2_mono")) {
  & $py -u cte.py $m 2>&1 | Where-Object { $_ -match 'alpha\(300K\)|Error|Traceback' } | Out-File -Append -Encoding utf8 run.log
}
"CTE DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
