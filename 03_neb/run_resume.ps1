$env:PYTHONIOENCODING = "utf-8"
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
Set-Location $PSScriptRoot
"RESUME START $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run_v3.log
$jobs = @(
  @("Nb2O5","O","Oi ON_VN"),
  @("Al2O3","O","Oi ON_VN"), @("TiO2","O","Oi ON_VN"), @("Ta2O5","O","Oi ON_VN"),
  @("TaN_cubic","N","ON_VN")
)
foreach ($j in $jobs) {
  $a = @("-u", "neb.py", $j[0], $j[1]) + $j[2].Split(' ')
  & $py @a 2>&1 | Where-Object { $_ -match 'Ea fwd|Error|Traceback' } | Out-File -Append -Encoding utf8 run_v3.log
}
"RESUME DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run_v3.log
