Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

. "C:\Users\npaus\venv\Scripts\Activate.ps1"

uvicorn inference_server:app --host 0.0.0.0 --port 9000

Read-Host -Prompt "Press Enter to exit"
