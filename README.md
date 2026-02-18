# LoRa RF Coverage Simulator (868 MHz)

Interaktivní Streamlit aplikace pro orientační odhad pokrytí a link budgetu pro LoRa / sub-GHz (např. 868 MHz).
Obsahuje modely šíření (FSPL + log-distance), vizualizace pokrytí (2D/3D), vyzařovací diagramy antén (2D řezy + 3D “koule”)
a volitelnou analýzu profilu terénu (LOS, Fresnel, difrakce).

> ⚠️ Výsledky jsou orientační. Neřeší multipath/fading, rušení, přesný clutter (budovy/les), polarizaci, ani regulatorní limity.

## Funkce

- **Model šíření**
  - FSPL (volný prostor)
  - Log-distance model (nastavitelné `n` a `d0`)
- **Link budget**
  - TX výkon, ztráty TX/RX, citlivost RX
  - výpočet PRX vs vzdálenost + odhad max dosahu (podle citlivosti)
- **Antény**
  - výběr typu (tvar vyzařování), zisk zadávaný ručně (TX i RX)
  - volitelně směrování (azimut) pro směrové antény
  - volitelně: u kolineáru možnost odvozovat tvar zisku (AUTO ON/OFF)
- **Vizualizace**
  - graf path loss vs vzdálenost
  - graf PRX vs vzdálenost
  - 2D heatmap pokrytí
  - 3D heatmap (surface) – vyžaduje Plotly
  - 2D řezy vyzařovacího diagramu (azimut + elevace)
  - 3D vyzařovací diagram (“koule”) – vyžaduje Plotly
- **Profil terénu (volitelně)**
  - upload CSV profilu (distance_m, elevation_m)
  - LOS + 1. Fresnelova zóna
  - jednoduchá difrakce (single knife-edge) jako přičtený útlum pro daný link

## Instalace

### 1) Vytvoření prostředí (doporučeno)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 2) Instalace závislostí
pip install -r requirements.txt

### 3) Spuštění
streamlit run app.py

Aplikace se otevře v prohlížeči (typicky http://localhost:8501).
Profil terénu: formát CSV
Aplikace očekává CSV se sloupci:
distance_m (0 … D)
elevation_m (nadmořská výška v metrech)

Příklad:
distance_m,elevation_m
0,312
50,313
100,315
...
1200,298


   








