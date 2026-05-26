#!/bin/bash
set Ceu
service_name=neomomo
gcloud run deploy $service_name --region=asia-northeast1 --memory=768Mi --concurrency=1 --source=. 
