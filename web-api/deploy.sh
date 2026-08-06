#!/bin/bash
set Ceu
service_name=neomomo
gh run download --dir ./artifacts/ -n linux-dist
gcloud run deploy $service_name --region=asia-northeast1 --memory=256Mi --concurrency=1 --source=. --set-build-env-vars=GOOGLE_PYTHON_PACKAGE_MANAGER=uv
