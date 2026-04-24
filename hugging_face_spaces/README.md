---
title: GN Food Estimator
emoji: 🍱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
tags:
  - streamlit
  - food
  - computer-vision
  - sustainability
pinned: false
short_description: Streamlit app + trained models
license: mit
---

# GN Food Estimator

Estimate food volume and CO₂ emissions from GN (Gastronorm) containers
using computer vision and deep learning.

Built by [Food Be Good](https://www.foodbegood.app/) as part of a food
waste reduction initiative.

## How it works

1. Take a top-down photo of your GN container with an ArUco marker visible
2. The app detects the container, estimates fill level, identifies the food,
   and calculates the associated CO₂ emissions
3. Take a second photo after food collection to see how much CO₂ was saved

## Models

- **Model #1** — Faster R-CNN ResNet-50 FPN: container detection
- **Model #2** — EfficientNet-B0: fill level classification
- **Model #3** — nateraw/food (ViT, Food-101): food identification

## Notes

- An ArUco marker (DICT_4X4_50, ID 0, 94 mm side) must be visible in every photo
- Supported foods: rice, lentils
- CO₂ estimates based on German/Bavarian lifecycle data (ifeu 2020, KTBL)
