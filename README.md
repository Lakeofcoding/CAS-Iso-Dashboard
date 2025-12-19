🧠 CAS-Dashboard – Medizinische Isolationen

Interaktives Dashboard zur Analyse und Visualisierung medizinischer Isolationsfälle in klinischen Einrichtungen.

🚀 Überblick

Dieses Projekt stellt einen funktionsfähigen Prototyp eines analytischen Dashboards dar, das medizinische Isolationsdaten strukturiert auswertet und visuell aufbereitet.
Der Fokus liegt auf transparenter Entscheidungsunterstützung, zeitlicher Dynamik und räumlicher Verteilung von Fällen.

Das Dashboard wurde mit Python, Streamlit und Plotly umgesetzt und ist vollständig interaktiv.

✨ Zentrale Funktionen

intelligente, heuristische Datums- und Intervallerkennung

mehrstufige Filterlogik (Station → Klinik → Zentrum)

Stichtagsbasierte Berechnung aktiver Fälle

KPI-Übersicht mit automatischer Risikoabschätzung

Detailansicht aller aktiven Isolationen (scrollbar)

interaktive Heatmap auf Grundrissbasis

Infektionsverteilung & Zeitreihenanalyse

modernes, professionelles UI

🧭 Navigation (Reiter)

Das Dashboard ist in vier Reiter unterteilt:

📊 Überblick

zentrale KPIs:

Gesamtfälle (gefiltert)

aktive Fälle am Stichtag

Anzahl Infektionsarten

offene Fälle (ohne Enddatum)

komprimierte Lageeinschätzung

🗺️ Heatmap & Infektionen

Balkendiagramm:

Verteilung nach Infektionstyp

Modus: alle Fälle oder nur aktive

Interaktive Grundriss-Heatmap

diffuse Heatmap oder Punktmarkierungen

automatische Parameteroptimierung

Filter nach Infektionstyp

visuelle Hotspot-Analyse auf Raumebene

📈 Zeitverlauf

Entwicklung aktiver Fälle über die Zeit

monatliche Aggregation

Modi:

Gesamtverlauf (eine Linie)

Aufschlüsselung nach Infektionsarten (Multi-Line)

dynamische Zeitachse basierend auf realen Start-/Stop-Daten
(kein künstlicher Abbruch mehr bei einzelnen Monaten)

📋 Detailansicht – Aktuelle Isolationen

„Aktuelle Isolationen im Detail“

zeigt die ersten 20 aktiven Fälle

weitere Einträge scrollbar

sortiert nach Startdatum (absteigend)

strukturierte Spalten:

Klinik

Zentrum

Station

Raum_ID

Infektion

Start / Stop

-----------------------------------------------
EINRICHTUNG
-----------------------------------------------
INFO: 

📁 Projektstruktur
CAS-Iso-Dashboard/
│
├── dashboard/
│   ├── app.py                # Hauptanwendung
│   └── assets/
│       ├── grundriss.png     # Original-Grundriss
│       └── derived/          # bereinigte / optimierte Assets
│
├── data/
│   ├── raw/                  # Original- / Dummy-Daten
│   └── processed/            # vorbereitete Datensätze
│
├── requirements.txt          # Python-Abhängigkeiten
└── README.md                 # Projektdokumentation



💾 Installation:


1️⃣ Repository klonen
git clone https://github.com/Lakeofcoding/CAS-Iso-Dashboard.git
cd CAS-Iso-Dashboard

(in diesem Fall erübrigt, da mittels ZIP Datei)

2️⃣ Virtuelle Umgebung erstellen
python -m venv .venv


Aktivieren:

Windows:

.venv\Scripts\activate


macOS / Linux:

source .venv/bin/activate

3️⃣ Abhängigkeiten installieren
pip install -r requirements.txt


START DES DASHBOARDS:


▶️ Dashboard starten
python -m streamlit run dashboard/app.py



Anschließend erreichbar unter:

http://localhost:8501



SONSTIGE INFORMATIONEN:

📂 Daten-Upload & Datenlogik
Unterstützte Formate

CSV

Excel (.xlsx, .xls)

Intelligente Verarbeitung

automatische Erkennung von:

Trennzeichen (, / ;)

Datumsfeldern

Start- und Endspalten


🧪 Erwartete Datenfelder (Beispiele)

Die App ist flexibel, erkennt aber typischerweise:

fallnummer

Infektion

Station

Zentrum

Klinik

Raum_ID

Startdatum (z. B. Startdatum Isolation)

Stopdatum (z. B. Stopdatum Isolation)

Datumsfelder werden heuristisch erkannt – exakte Spaltennamen sind nicht zwingend erforderlich.
