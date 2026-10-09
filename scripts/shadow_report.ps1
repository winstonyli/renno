param([Parameter(Mandatory)][string]$Log, [int]$Samples = 3)
$total = 0; $dis = 0; $bad = 0
$siteN = @{}; $siteD = @{}; $cls = @{}
foreach ($line in [System.IO.File]::ReadLines($Log)) {
  $f = $line.Split("`t")
  if ($f.Count -lt 4 -or ($f[0] -ne 'A' -and $f[0] -ne 'D') -or ($f[0] -eq 'D' -and $f.Count -ne 11)) { $bad++; continue }
  $total++
  $siteN[$f[1]] = 1 + [int]$siteN[$f[1]]
  if ($f[0] -eq 'A') { continue }
  $dis++
  $siteD[$f[1]] = 1 + [int]$siteD[$f[1]]
  $c = $cls[$f[6]]
  if ($null -eq $c) { $c = @{ n = 0; tags = @{}; shapes = @{}; samples = New-Object System.Collections.ArrayList; seen = @{} }; $cls[$f[6]] = $c }
  $c.n++
  $c.tags[$f[7]] = 1
  $shape = "$($f[1]) old=$($f[2]) new=$($f[3]) cause=$($f[4]) kinds=$($f[5])"
  $c.shapes[$shape] = 1 + [int]$c.shapes[$shape]
  $key = "$($f[9]) => $($f[10])"
  if ($c.samples.Count -lt $Samples -and -not $c.seen.ContainsKey($key)) { $c.seen[$key] = 1; [void]$c.samples.Add("[$($f[7])@$($f[8])] $key") }
}
"Total lines: $total  disagreements: $dis  malformed: $bad"
""; "| Site | lines | disagreements |"; "|---|---|---|"
foreach ($s in ($siteN.Keys | Sort-Object)) { "| $s | $($siteN[$s]) | $([int]$siteD[$s]) |" }
""; "## Classes"
foreach ($name in ($cls.Keys | Sort-Object { -$cls[$_].n })) {
  $c = $cls[$name]
  ""; "### ${name}: $($c.n) lines, $($c.tags.Count) distinct tags"
  foreach ($s in ($c.shapes.Keys | Sort-Object { -$c.shapes[$_] })) { "- ${s}: $($c.shapes[$s])" }
  "Samples:"
  foreach ($s in $c.samples) { "- $s" }
}
