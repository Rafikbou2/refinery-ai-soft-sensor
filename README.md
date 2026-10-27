# Refinery AI — Predictive Quality & Process Optimization

A soft sensor for the diesel side-draw of a crude distillation unit (CDU). Lab results for
**T95 (ASTM D86)** and **flash point** only come every 4 hours. This application:

1. **Receives operating data**: from the historian (CSV), from a file upload in the interface, or through a REST API.
2. **Predicts product quality** continuously, with a 90 % prediction interval and the probability of going off-spec.
3. **Shows KPIs**: on-spec percentage, diesel yield, quality giveaway in $/day, steam and energy intensity.
4. **Detects risks**: quality breach and its trend, unusual operation (Isolation Forest), inputs outside the model's training range, transmitter faults (frozen value or spikes), and drift against lab results.
5. **Recommends set-points** that increase margin while keeping both specs within the allowed risk, plus a What-if simulator.

The interface is bilingual (Arabic / English) and built for process engineers and operators: they upload
their usual Excel log and get the predictions, alerts and recommendations.

![Live quality — T95 approaching its limit](docs/screenshots/01_live_quality_en.png)

| Risks & alerts | Set-point recommendation |
|---|---|
| ![Risks and alerts](docs/screenshots/02_risks_alerts_en.png) | ![Optimization](docs/screenshots/03_optimization_en.png) |
| **Model performance** | **Arabic interface (RTL)** |
| ![Model](docs/screenshots/04_model_en.png) | ![Arabic interface](docs/screenshots/01_live_quality_ar.png) |

## Running it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python train.py            # generates 180 days of demo data and trains the model (~30 s)
.\.venv\Scripts\streamlit run app.py       # engineer interface  → http://localhost:8501
.\.venv\Scripts\uvicorn api:app --port 8000  # REST API → http://localhost:8000/docs
.\.venv\Scripts\python -m pytest -q        # tests
```

### Language
The interface is in **Arabic** by default, with right-to-left layout. Switch to English from the top of the sidebar. All
text is in [softsensor/i18n.py](softsensor/i18n.py); to add a language (French, for example), add a column
to the tables there.

### Using it with an Excel file
In the interface, choose **Upload Excel / CSV file**, then upload an `.xlsx`, `.xls` or `.csv` file.
- You can download a ready-to-fill template with the **⬇️ Excel template** button.
- Column names are accepted in Arabic, French or English (with or without units), and the header row does not have to be the first row.
- Date and time can be in one column or two separate columns, and decimal commas are accepted.
- Lab columns are optional.

Try it with the sample file [data/exemple_releve_CDU.xlsx](data/exemple_releve_CDU.xlsx).

To train on real data from your unit: `python train.py --file data/my_unit.xlsx`. The file must contain the
lab results (`t95_lab` / `flash_point_lab`, or `T95 labo` / `Point d'éclair`).

## Structure

| File | Role |
|---|---|
| [softsensor/config.py](softsensor/config.py) | Tags, operating limits, specs, costs |
| [softsensor/simulator.py](softsensor/simulator.py) | Demo historian data: crude changes, upsets, sensor faults |
| [softsensor/model.py](softsensor/model.py) | Ridge + Gradient Boosting on the residuals, conformal intervals, anomaly detection |
| [softsensor/risk.py](softsensor/risk.py) | KPIs, alerts, transmitter health |
| [softsensor/optimizer.py](softsensor/optimizer.py) | Set-point search within move limits |
| [app.py](app.py) | Streamlit interface (4 tabs) |
| [api.py](api.py) | `/predict`, `/recommend` and `/health` |

## Model performance (demo data, time-ordered split, latest 20 %)

| Target | RMSE | R² | 90 % interval coverage |
|---|---|---|---|
| T95 | 1.21 °C | 0.86 | 90 % |
| Flash point | 0.76 °C | 0.83 | 93 % |

The error is close to the lab measurement noise built into the simulator (1.0 and 0.6 °C).

> **Note:** the data and economic figures are synthetic, for demonstration. Before using the
> application on a real unit, retrain on its historian data and review the sensitivities and costs
> in `config.py`.
