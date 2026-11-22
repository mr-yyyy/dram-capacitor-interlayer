Remove-Item Env:MD_SMOKE_TEST -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
Set-Location $PSScriptRoot
"V2 START $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
foreach ($seed in 1, 2) {
  $procs = foreach ($T in 800, 1000, 1200) {
    Start-Process $py -ArgumentList "md_v2.py", $T, $seed -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardError (Join-Path $PSScriptRoot "err_v2_T$($T)_s$seed.txt")
  }
  $procs | Wait-Process
  "v2 batch seed $seed finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
}
& $py analyse.py v2 *> analyse_v2_out.txt
Remove-Item (Join-Path $PSScriptRoot "v2_running.flag") -ErrorAction SilentlyContinue
"V2 DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
