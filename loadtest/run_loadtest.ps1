param([int]$Users = 50, [int]$Rate = 10, [string]$Duration = "60s")

New-Item -ItemType Directory -Force -Path results | Out-Null
foreach ($n in 1, 2, 3) {
    Write-Host "=== $n API container(s) ==="
    docker compose up -d --build --scale api=$n api nginx
    Start-Sleep -Seconds 30    # let TensorFlow load in each container
    python -m locust -f loadtest/locustfile.py --headless -u $Users -r $Rate -t $Duration `
        --host http://localhost:8000 --csv "results/containers_$n" --only-summary
}
python loadtest/summarize.py
