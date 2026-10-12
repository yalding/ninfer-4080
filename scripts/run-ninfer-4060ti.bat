@echo off
setlocal EnableExtensions

rem ============================================================================
rem  NInfer Local Windows Launcher for NVIDIA GeForce RTX 4060 Ti 16 GB (sm_89)
rem
rem  Serves the registered 3-bit GSQ artifact (Qwen3.8-27B) natively on Windows.
rem  Supports high-throughput DFlash2 (K=7, up to 90K context) and high-context MTP3.
rem
rem  Measured performance on RTX 4060 Ti 16 GB:
rem    * Prefill: 1097.4 tok/s (pp32768), 928.1 tok/s (pp65536, chunk 2688)
rem    * DFlash2 K=7: ~83.5 tok/s (structured), ~97.5 tok/s (code), ~89.7 tok/s (tiled tg512)
rem    * MTP3: ~55.6 tok/s (structured), ~61.5 tok/s (code), ~71.2 tok/s (pp8192+tg128)
rem    * No-spec baseline: ~19.3 - 20.2 tok/s
rem    * Memory slack: >1000 MiB free at 100K context + vision overlay + rk4v4-e8
rem ============================================================================

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

if not defined NINFER_ARTIFACT (
  if exist "C:\LLM\qwen3_8_27b_gsq3.ninfer" (
    set "NINFER_ARTIFACT=C:\LLM\qwen3_8_27b_gsq3.ninfer"
  ) else if exist "%ROOT%\out\qwen3_8_27b_gsq3.ninfer" (
    set "NINFER_ARTIFACT=%ROOT%\out\qwen3_8_27b_gsq3.ninfer"
  ) else if exist "%ROOT%\models\qwen3_8_27b_gsq3.ninfer" (
    set "NINFER_ARTIFACT=%ROOT%\models\qwen3_8_27b_gsq3.ninfer"
  )
)

if not defined NINFER_SPEC set "NINFER_SPEC=mtp"
if not defined NINFER_PORT set "NINFER_PORT=8080"
if not defined NINFER_BIND set "NINFER_BIND=127.0.0.1"
if not defined NINFER_KV_DTYPE set "NINFER_KV_DTYPE=rk4v4-e8"

if not defined NINFER_VISION set "NINFER_VISION=0"

if /i "%NINFER_SPEC%"=="mtp" (
  if not defined NINFER_DRAFT_TOKENS set "NINFER_DRAFT_TOKENS=3"
  if not defined NINFER_CONTEXT set "NINFER_CONTEXT=102400"
  set "EXTRA_SPEC_FLAGS=--ngram chain"
  set "EXTRA_VISION_FLAGS=--vision --vision-offload on"
) else (
  if not defined NINFER_DRAFT_TOKENS set "NINFER_DRAFT_TOKENS=7"
  if "%NINFER_VISION%"=="1" (
    if not defined NINFER_CONTEXT set "NINFER_CONTEXT=65536"
    set "EXTRA_VISION_FLAGS=--vision --vision-offload on"
  ) else (
    if not defined NINFER_CONTEXT set "NINFER_CONTEXT=90000"
    set "EXTRA_VISION_FLAGS="
  )
  set "EXTRA_SPEC_FLAGS="
)

rem Locate binary
set "SERVER="
if exist "%ROOT%\build-windows\apps\ninfer-serve.exe" (
  set "SERVER=%ROOT%\build-windows\apps\ninfer-serve.exe"
) else if exist "%ROOT%\build-4060ti-ninja\apps\ninfer-serve.exe" (
  set "SERVER=%ROOT%\build-4060ti-ninja\apps\ninfer-serve.exe"
) else if exist "%ROOT%\build-4060ti-clang-all\apps\ninfer-serve.exe" (
  set "SERVER=%ROOT%\build-4060ti-clang-all\apps\ninfer-serve.exe"
)

if not defined SERVER (
  echo Could not locate ninfer-serve.exe. Build with:
  echo   cmake --build build-windows --config Release --target ninfer-serve -j
  exit /b 1
)

if not exist "%NINFER_ARTIFACT%" (
  echo Artifact not found at: %NINFER_ARTIFACT%
  echo Please run scripts\download-qwen38-gsq3.bat
  exit /b 1
)

echo Starting NInfer on RTX 4060 Ti 16 GB at http://%NINFER_BIND%:%NINFER_PORT%/v1
echo Model: %NINFER_ARTIFACT%
echo Speculative Backend: %NINFER_SPEC% (Draft Tokens: %NINFER_DRAFT_TOKENS%)
echo Max Context: %NINFER_CONTEXT% (KV: %NINFER_KV_DTYPE%)

if defined NINFER_API_KEY (
  "%SERVER%" "%NINFER_ARTIFACT%" --host %NINFER_BIND% --port %NINFER_PORT% --api-key "%NINFER_API_KEY%" --max-context %NINFER_CONTEXT% --kv-capacity %NINFER_CONTEXT% --kv-dtype %NINFER_KV_DTYPE% --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096 --spec %NINFER_SPEC% --draft-tokens %NINFER_DRAFT_TOKENS% %EXTRA_SPEC_FLAGS% --lm-head-draft %EXTRA_VISION_FLAGS% --preserve-thinking
) else (
  "%SERVER%" "%NINFER_ARTIFACT%" --host %NINFER_BIND% --port %NINFER_PORT% --max-context %NINFER_CONTEXT% --kv-capacity %NINFER_CONTEXT% --kv-dtype %NINFER_KV_DTYPE% --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096 --spec %NINFER_SPEC% --draft-tokens %NINFER_DRAFT_TOKENS% %EXTRA_SPEC_FLAGS% --lm-head-draft %EXTRA_VISION_FLAGS% --preserve-thinking
)
