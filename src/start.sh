#!/bin/bash
gunicorn serve_model:app --bind 0.0.0.0:$PORT