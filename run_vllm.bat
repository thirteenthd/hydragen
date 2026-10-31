@echo off
set "CONFIG_FILE=strata-iq3_xxs.json"
if not exist "%CONFIG_FILE%" (
    for %%f in (strata-*.json) do (
        set "CONFIG_FILE=%%f"
        goto :found
    )
)
:found
echo [hydragen] Starting server using config: %CONFIG_FILE%
python -m serve.server --engine vllm --config "%CONFIG_FILE%" --port 8080 --api-monitor --no-api-key %*
