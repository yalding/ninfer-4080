@echo off
setlocal EnableExtensions

rem Native Inkling-2-Mini, multimodal vision-from-RAM, 128K context on RTX 4060 Ti 16 GB.
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PATH=D:\Applications\Scoop\apps\cuda\13.4.3\bin\x64;%PATH%"
if not defined NINFER_VISION set "NINFER_VISION=1"
if not defined NINFER_ARTIFACT (
  if "%NINFER_VISION%"=="1" (
    set "NINFER_ARTIFACT=%ROOT%\models\inkling2_mini_q3g128.ninfer"
  ) else (
    set "NINFER_ARTIFACT=C:\LLM\inkling2_mini_q3g128.ninfer"
  )
)
if not defined NINFER_BIND set "NINFER_BIND=127.0.0.1"
if not defined NINFER_PORT set "NINFER_PORT=8001"
set "VISION_FLAGS="
if "%NINFER_VISION%"=="1" set "VISION_FLAGS=--vision --vision-offload on"
set "SERVER=%ROOT%\build-4060ti-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%ROOT%\build-windows\apps\ninfer-serve.exe"
if not exist "%SERVER%" (
  echo Could not locate ninfer-serve.exe. Build ninfer-serve first.
  exit /b 1
)
if not exist "%NINFER_ARTIFACT%" (
  echo Artifact not found: %NINFER_ARTIFACT%
  exit /b 1
)
echo Serving Inkling-2-Mini at http://%NINFER_BIND%:%NINFER_PORT%/v1
if defined NINFER_API_KEY (
  "%SERVER%" "%NINFER_ARTIFACT%" --host %NINFER_BIND% --port %NINFER_PORT% --api-key "%NINFER_API_KEY%" %VISION_FLAGS% --max-context 131072 --kv-capacity 131072 --kv-dtype rk4v4-e8 --prefill-chunk 1024 --host-kv-mib 4096 --max-concurrency 1 --max-pending-requests 16 %*
) else (
  "%SERVER%" "%NINFER_ARTIFACT%" --host %NINFER_BIND% --port %NINFER_PORT% --max-context 131072 --kv-capacity 131072 --kv-dtype rk4v4-e8 --prefill-chunk 1024 --host-kv-mib 4096 --max-concurrency 1 --max-pending-requests 16 %VISION_FLAGS% %*
)
exit /b %ERRORLEVEL%
