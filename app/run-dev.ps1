# 设置Visual Studio Build Tools环境变量
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcPath = "$vsPath\VC\Tools\MSVC"
$msvcVersion = (Get-ChildItem $vcPath | Sort-Object -Descending | Select-Object -First 1).Name

# Windows SDK路径
$winSdkPath = "C:\Program Files (x86)\Windows Kits\10"
$sdkVersion = (Get-ChildItem "$winSdkPath\bin" | Where-Object { $_.Name -match "^\d+\.\d+\.\d+" } | Sort-Object -Descending | Select-Object -First 1).Name

# 设置环境变量
$env:INCLUDE = "$vcPath\$msvcVersion\include;$winSdkPath\Include\$sdkVersion\ucrt;$winSdkPath\Include\$sdkVersion\um;$winSdkPath\Include\$sdkVersion\shared"
$env:LIB = "$vcPath\$msvcVersion\lib\x64;$winSdkPath\Lib\$sdkVersion\ucrt\x64;$winSdkPath\Lib\$sdkVersion\um\x64"
$env:LIBPATH = "$vcPath\$msvcVersion\lib\x64"

# 设置PATH
$binPath = "$vcPath\$msvcVersion\bin\Hostx64\x64"
$sdkBinPath = "$winSdkPath\bin\$sdkVersion\x64"
$env:Path = "$binPath;$sdkBinPath;$env:Path"

Write-Host "Environment set:"
Write-Host "MSVC Version: $msvcVersion"
Write-Host "SDK Version: $sdkVersion"
Write-Host "BIN Path: $binPath"
Write-Host "SDK BIN Path: $sdkBinPath"

# 运行Tauri开发
Set-Location "E:\Cursor Code\Chronos\app"
& npm run tauri:dev