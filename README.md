# Photonic Sensing

The aim of this project is to gain a theoretical understanding of the changes in the properties of light as it travels through an element composed of metal nanoparticles. 
Light is passed through a transparent material composed of metal nanoparticles and its characteristics, such as reflection, transmission and dynamic range, are observed at the output.

## Table of contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)

## Features

If you need specific information on n1, n2, n3, f, d, TE, TM click on it specific tabview's case

- choose the wavelength range being studied
- choose if the light come with an angle (if yes : choose TM, write the angle on the pop-up and click on "angle")
- select the application ("color in glass" or "sensor")
- choose the host, the substrate and the metal(s) observed
- choose the variation to be observed (variation of f, n3, d) or the dynamic range
- click on calculation
- click on export data for make a csv file with the data of the last calcul
- click on show

## Prerequisites

- Python (3.10)
- Bibliothèques nécessaires :
  - `matplotlib`
  - `numpy`
  - `csv`
  - `math`
  - `cmath`
  - `colorsys`
  - `typing`
  - `customtkinter`
  - `PIL` (Pillow)

```bash
pip install matplotlib numpy customtkinter Pillow
