# Waits for the YSZ MD (step 4) to finish, then runs step 6 (doped TiN) and step 7 (low-barrier MD).
$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:MD_SMOKE_TEST, Env:DOPED_DRY, Env:DOPED_FETCH_ONLY -ErrorAction SilentlyContinue
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $PSScriptRoot "run.log"
"WAIT for YSZ $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
while (-not (Select-String -Path (Join-Path $root "06_ysz_md\run.log") -Pattern "MD DONE" -Quiet)) { Start-Sleep 60 }

Set-Location (Join-Path $root "06_ysz_md")
& $py analyse.py *> analyse_out.txt
"YSZ analysed $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log

Set-Location (Join-Path $root "08_doped_tin")
"DOPED START $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
& $py doped.py *> doped_out.txt
"DOPED END $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log

Set-Location $PSScriptRoot
foreach ($m in "Al2O3", "HfO2", "Si3N4", "Nb2O5", "SiO2") {      # 3 temperatures concurrently per material
  $procs = foreach ($T in 1000, 1200, 1400) {
    Start-Process $py -ArgumentList "md_defect.py", $m, $T -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardError (Join-Path $PSScriptRoot "err_$($m)_$T.txt")
  }
  $procs | Wait-Process
  "$m finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
}
& $py analyse_defect.py *> analyse_out.txt
"ALL DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
