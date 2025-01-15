#!/bin/bash

echo "Starting services..."
docker compose up -d

echo "Waiting for services to start..."
sleep 5

echo "Starting chat console..."
docker compose build console
docker compose run --rm console python console.py
