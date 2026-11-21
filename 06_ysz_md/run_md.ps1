Remove-Item Env:MD_SMOKE_TEST -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:USERPROFILE\miniconda3\envs\m3gnet\python.exe"
Set-Location $PSScriptRoot
"MD START $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
foreach ($seed in 1, 2) {           # 3 concurrent runs per batch (GPU is call-overhead bound: 2.3x throughput)
  $procs = foreach ($T in 800, 1000, 1200) {
    Start-Process $py -ArgumentList "md.py", $T, $seed -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardError (Join-Path $PSScriptRoot "err_T$($T)_s$seed.txt")
  }
  $procs | Wait-Process
  "batch seed $seed finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
}
"MD DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 run.log
