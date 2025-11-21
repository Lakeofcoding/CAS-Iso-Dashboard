🧠 CAS-Dashboard – Medizinische Isolationen

Interaktiver Prototyp für die Visualisierung medizinischer Isolationsfälle


🚀 Überblick

Dieses Projekt enthält einen funktionsfähigen Prototyp eines Dashboards zur Visualisierung von Isolationen in medizinischen Einrichtungen.
Der Fokus liegt auf:

interaktiven Filtern (Station, Zentrum, Klinik)

automatischer - und intelligenter - Datumsinterpretation

KPI-Übersicht

aktueller Fallliste nach Stichtag

Infektionsverteilung

Zeitreihenanalyse der aktiven Fälle

modernem UI-Design

Das Dashboard ist in Python mit Streamlit und Plotly implementiert.

📁 Projektstruktur
CAS-Iso-Dashboard/
│
├── dashboard/
│   ├── app.py            # Hauptanwendung
│   ├── assets/
│   │     └── style.css   # Layout & Farbtheme
│
├── data/
│   ├── raw/              # Originaldaten / Dummy-Daten
│   └── processed/        # vorbereitete Datensätze
│
├── requirements.txt      # benötigte Python-Pakete
└── README.md             # (diese Datei)

💾 Installation
1. Repository klonen
git clone git@github.com:Lakeofcoding/CAS-Iso-Dashboard.git
cd CAS-Iso-Dashboard

2. Python-Umgebung erstellen
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

3. Abhängigkeiten installieren
pip install -r requirements.txt

▶️ Dashboard starten
python -m streamlit run dashboard/app.py


Das Dashboard öffnet sich dann unter:

http://localhost:8501

📊 Funktionsumfang
🔹 Daten-Upload

CSV / Excel

automatische Erkennung von Semikolon/Komma

automatische Erkennung von Datumsfeldern

Erkennung von Start/Stop-Spalten

🔹 Filter

Station

Zentrum (abhängig von Station)

Klinik (abhängig von Station/Zentrum)

Zurücksetzen mit einem Klick

🔹 Kennzahlen

Anzahl Fälle

Aktive Fälle am Stichtag

Unterschiedliche Infektionen (aktiv)

Offene Fälle (Enddatum fehlt)

🔹 Tabellenansicht

alle aktiven Fälle zum Stichtag

sortiert nach Startdatum absteigend

logisch strukturierte Spalten

🔹 Fälle pro Infektion

Modus: gesamt oder aktiv am Stichtag

modernes Bar-Chart (Plotly)

🔹 Zeitverlauf

aktive Fälle pro Monat

Modus:

eine Linie gesamt

mehrere Linien nach Infektionsart

Multiselect für Infektionen

🖌️ Design & Farben

Ein modernes Farbschema ist hinterlegt in:

dashboard/assets/style.css


Streamlit lädt dieses Design beim Start automatisch.
Das Theme wurde neutral-professionell gehalten (Blau-Grautöne).

🔍 Datenbasis

Die App erwartet Spalten wie:

fallnummer

Infektion

Station

Zentrum

Klinik

Startdatum Isolation

Stopdatum Isolation

Raum_ID


Datumsfelder werden heuristisch erkannt.
