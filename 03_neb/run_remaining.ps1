$env:PYTHONIOENCODING = "utf-8"
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
Set-Location $PSScriptRoot
$jobs = @(
  @("HfN","N","Oi"), @("ZrN","N","Oi"), @("TaC","C","Oi"), @("NbC","C","Oi"),
  @("Si3N4","N","Oi ON_VN"), @("SiC","C","Oi ON_VN"), @("Al2O3","O","Oi ON_VN"),
  @("HfO2","O","Oi ON_VN"), @("TiO2","O","Oi ON_VN"), @("Ta2O5","O","Oi ON_VN"),
  @("SiO2","O","Oi ON_VN"), @("Y2O3","O","Oi ON_VN"), @("Nb2O5","O","Oi ON_VN")
)
foreach ($j in $jobs) {
  $a = @("-u", "neb.py", $j[0], $j[1]) + $j[2].Split(' ')
  & $py @a 2>&1 | Where-Object { $_ -match 'Ea fwd|Error|Traceback' } | Out-File -Append -Encoding utf8 run_v3.log
}
"ALL DONE" | Out-File -Append -Encoding utf8 run_v3.log
