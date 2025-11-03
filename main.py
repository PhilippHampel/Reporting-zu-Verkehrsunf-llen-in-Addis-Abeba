import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os

# =======================================================================================
# 1) CSV einlesen
# =======================================================================================
df = pd.read_csv("Addis_Ababa_city_RTA.csv")

REQ_COLS = {"Accident_severity", "Cause_of_accident"}
missing = REQ_COLS - set(df.columns)
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}")

# Übersetzungsfunktion für Unfallschwere
def translate_severity(severity):
    translations = {
        "Slight Injury": "leichter Unfall",
        "Serious Injury": "schwerer Unfall",
        "Fatal injury": "tödlicher Unfall"
    }
    return translations.get(severity, severity)

# Übersetzungsfunktion für Unfallursachen
def translate_cause(cause):
    translations = {
        "No distancing": "Kein Abstand",
        "Changing lane to the right": "Spurwechsel nach rechts",
        "Changing lane to the left": "Spurwechsel nach links",
        "Driving carelessly": "Unachtsames Fahren",
        "No priority to vehicle": "Keine Vorfahrt für Fahrzeuge",
        "Moving Backward": "Rückwärtsfahren",
        "No priority to pedestrian": "Keine Vorfahrt für Fußgänger",
        "Other": "Sonstiges",
        "Overtaking": "Überholen",
        "Driving under the influence of drugs": "Fahren unter Drogeneinfluss",
        "Driving to the left": "Nach links fahren",
        "Driving at high speed": "Fahren mit hoher Geschwindigkeit",
        "Getting off the vehicle improperly": "Unsachgemäßes Aussteigen",
        "Overturning": "Umkippen",
        "Turnover": "Überschlag",
        "Overspeed": "Überhöhte Geschwindigkeit",
        "Overloading": "Überladung",
        "Drunk driving": "Trunkenheit am Steuer",
        "Improper parking": "Falsches Parken",
        "Unknown": "Unbekannt"
    }
    return translations.get(cause, cause)

# Übersetzungsfunktion für alle anderen Spalteninhalte
def translate_value(value, column=None):
    translations = {
        # Bildungsgrad
        "Above high school": "Über Gymnasium",
        "Elementary school": "Grundschule",
        "High school": "Gymnasium",
        "Illiterate": "Analphabetisch",
        "Junior high school": "Mittelstufe",
        "Writing & reading": "Lesen & Schreiben",
        
        # Kreuzungstypen
        "No junction": "Keine Kreuzung",
        "Out of junction": "Außerhalb Kreuzung",
        
        # Geschlecht
        "Female": "Weiblich",
        "Male": "Männlich",
        
        # Kollisionstypen
        "Collision with animals": "Kollision mit Tieren",
        "Collision with pedestrians": "Kollision mit Fußgängern",
        "Collision with roadside objects": "Kollision mit Straßenobjekten",
        "Collision with roadside-parked vehicles": "Kollision mit geparkten Fahrzeugen",
        "Fall from vehicles": "Sturz vom Fahrzeug",
        "Rollover": "Überschlag",
        "Vehicle with vehicle collision": "Fahrzeug-Fahrzeug-Kollision",
        "With Train": "Mit Zug",
        
        # Fahrzeugtypen (falls vorhanden)
        "Automobile": "PKW",
        "Public (> 45 seats)": "Öffentlich (> 45 Sitze)",
        "Public (13-45 seats)": "Öffentlich (13-45 Sitze)",
        "Public (12 seats)": "Öffentlich (12 Sitze)",
        "Lorry (41-100Q)": "LKW (41-100Q)",
        "Lorry (11-40Q)": "LKW (11-40Q)",
        "Lorry (<=10Q)": "LKW (<=10Q)",
        "Long lorry": "Langer LKW",
        "Taxi": "Taxi",
        "Motorcycle": "Motorrad",
        "Special vehicle": "Spezialfahrzeug",
        "Bajaj": "Bajaj",
        "Bicycle": "Fahrrad",
        
        # Straßentypen
        "Double carriageway (median)": "Zweispurig (mit Mittelstreifen)",
        "One way": "Einbahnstraße",
        "Two-way (divided with broken lines road marking)": "Zweispurig (unterbrochene Linien)",
        "Two-way (divided with solid lines road marking)": "Zweispurig (durchgezogene Linien)",
        "Undivided Two way": "Ungeteilte zweispurige Straße",
        
        # Allgemeine Begriffe
        "Other": "Sonstiges",
        "Unknown": "Unbekannt",
        "NA": "k.A."
    }
    
    # Konvertiere zu String für die Übersetzung
    value_str = str(value)
    return translations.get(value_str, value_str)

# Übersetzungsfunktion für Spaltennamen
def translate_column_name(col_name):
    translations = {
        "Drivers_gender": "Geschlecht des Fahrers",
        "Educational_level": "Bildungsniveau",
        "Number_of_vehicles_involved": "Anzahl beteiligter Fahrzeuge",
        "Types_of_junction": "Kreuzungstyp",
        "Types_of_Junction": "Kreuzungstyp",  # Mit großem J
        "Type_of_collision": "Kollisionstyp",
        "Accident_severity": "Unfallschwere",
        "Cause_of_accident": "Unfallursache",
        "Day_of_week": "Wochentag",
        "Time_of_day": "Tageszeit",
        "Age_band_of_driver": "Altersgruppe des Fahrers",
        "Sex_of_driver": "Geschlecht des Fahrers",
        "Type_of_vehicle": "Fahrzeugtyp",
        "Owner_of_vehicle": "Fahrzeughalter",
        "Service_year_of_vehicle": "Fahrzeugalter",
        "Defect_of_vehicle": "Fahrzeugmangel",
        "Area_accident_occured": "Unfallgebiet",
        "Road_surface_type": "Straßenbelag",
        "Road_surface_conditions": "Straßenzustand",
        "Light_conditions": "Lichtverhältnisse",
        "Weather_conditions": "Wetterbedingungen",
        "Lanes_or_Medians": "Fahrspuren oder Mittelstreifen",
        "Vehicle_movement": "Fahrzeugbewegung",
        "Casualty_class": "Opferklasse",
        "Sex_of_casualty": "Geschlecht des Opfers",
        "Age_band_of_casualty": "Altersgruppe des Opfers",
        "Casualty_severity": "Verletzungsschwere",
        "Work_of_casuality": "Beruf des Opfers",
        "Fitness_of_casuality": "Gesundheitszustand des Opfers",
        "Pedestrian_movement": "Fußgängerbewegung"
    }
    return translations.get(col_name, col_name)

# Matte Farben für Unfallschwere (heller gemacht)
severity_colors = {
    "Slight Injury": "#B8B8B8",    # Helleres Grau (vorher #8C8C8C)
    "Serious Injury": "#F4D03F",   # Helleres Gelb (vorher #D4A017)
    "Fatal injury": "#E74C3C"      # Helleres Rot (vorher #B22222)
}

# Hauptordner + Unterordner anlegen
base_dir = "charts"
folders = {
    "pie": os.path.join(base_dir, "1_unfallschwere_verteilung"),
    "stacked": os.path.join(base_dir, "2_unfallursachen_nach_schweregrad"),
    "top10": os.path.join(base_dir, "3_top10_allcols"),
    "spalten": os.path.join(base_dir, "4_spalten_verteilungen"),
    "spalten_compare": os.path.join(base_dir, "5_spalten_vergleich"),
}
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# =======================================================================================
# 2) Diagramm: Verteilung der Unfallschwere (Tortendiagramm, größere Labels)
# =======================================================================================
severity_counts = df["Accident_severity"].value_counts().sort_index()
total = severity_counts.sum()

# Tortendiagramm ohne Beschriftungen
plt.figure(figsize=(8, 8))
colors = [severity_colors.get(sev, "#808080") for sev in severity_counts.index]

wedges, texts = plt.pie(
    severity_counts,
    labels=None,  # Keine Labels am Diagramm
    colors=colors,
    startangle=90,
    wedgeprops={"edgecolor": "black", "linewidth": 1.5},
)

# Prozentwerte - größer und alle in schwarz
for i, wedge in enumerate(wedges):
    pct = severity_counts.iloc[i] / total * 100
    angle = (wedge.theta2 + wedge.theta1) / 2.0
    angle_rad = np.deg2rad(angle)
    
    # Position: außerhalb bei <5%, sonst innerhalb
    if pct < 5:
        r = 1.15  # Außerhalb der Torte
    else:
        r = 0.7  # Innerhalb der Torte
    
    # Einheitliche große Schriftgröße für alle
    fontsize = 22
    
    x = r * np.cos(angle_rad)
    y = r * np.sin(angle_rad)
    plt.text(x, y, f"{pct:.1f}%", ha="center", va="center",
             fontsize=fontsize, weight="bold", color="black")  # Alle in schwarz

# Kein Titel
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(folders["pie"], "unfallschwere_verteilung.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================================================================================
# 2b) Separate Legende als eigene Grafik
# =======================================================================================
fig, ax = plt.subplots(figsize=(5, 2.5))  # Kleinere Grafik für die einfachere Legende
ax.axis('off')

# Erstelle Legende mit Farben und deutschen Bezeichnungen (ohne Zahlen)
legend_elements = []
for severity in severity_counts.index:
    label = translate_severity(severity)  # Nur die Bezeichnung, keine Zahlen
    color = severity_colors.get(severity, "#808080")
    legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

legend = ax.legend(handles=legend_elements, 
                   loc='center', 
                   fontsize=14,
                   title="Unfallschwere",
                   title_fontsize=16,
                   frameon=True,
                   fancybox=True,
                   shadow=True)

plt.tight_layout()
plt.savefig(os.path.join(folders["pie"], "unfallschwere_legende.png"), dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# =======================================================================================
# 3) Gruppieren: Counts & Prozent pro (Schwere, Ursache)
# =======================================================================================
grouped = (
    df.groupby(["Accident_severity", "Cause_of_accident"])
    .size()
    .reset_index(name="count")
)
grouped["percent"] = grouped.groupby("Accident_severity")["count"].transform(
    lambda x: x / x.sum() * 100
)

# =======================================================================================
# 4) Textdatei mit exakten Werten
# =======================================================================================
txt_path = os.path.join(folders["stacked"], "gestapeltes_balkendiagramm_daten.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Daten für gestapeltes Balkendiagramm (Verteilung der Unfallursachen pro Unfallschwere)\n")
    f.write("=" * 100 + "\n\n")
    for severity in grouped["Accident_severity"].unique():
        sub = grouped[grouped["Accident_severity"] == severity].sort_values("count", ascending=False)
        f.write(f"Unfallschwere: {translate_severity(severity)}\n")
        f.write("-" * 100 + "\n")
        for _, row in sub.iterrows():
            f.write(f"{translate_cause(row['Cause_of_accident']):<60}  Count: {int(row['count']):>6}   Anteil: {row['percent']:>6.2f}%\n")
        f.write("\n")

# =======================================================================================
# 5) Farben konsistent je Ursache
# =======================================================================================
global_cause_order = df["Cause_of_accident"].value_counts().index.tolist()
palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
color_map = {cause: palette[i % len(palette)] for i, cause in enumerate(global_cause_order)}
other_color = "#cccccc"

# =======================================================================================
# 6) Gestapeltes Diagramm mit <5%-Zusammenfassung ("Other")
# =======================================================================================
severities = list(grouped["Accident_severity"].unique())
severities_deutsch = [translate_severity(sev) for sev in severities]
x = np.arange(len(severities))

plt.figure(figsize=(16, 10))  # Größer für die verdoppelten Schriften
legend_labels = set()

for i, severity in enumerate(severities):
    sub = grouped[grouped["Accident_severity"] == severity].copy()
    small_mask = sub["percent"] < 5
    other_sum = sub.loc[small_mask, "percent"].sum()
    sub = sub.loc[~small_mask].sort_values("percent", ascending=False)
    if other_sum > 0:
        sub = pd.concat([
            sub,
            pd.DataFrame({
                "Accident_severity": [severity],
                "Cause_of_accident": ["Other (<5%)"],
                "percent": [other_sum]
            })
        ], ignore_index=True)

    bottom = 0.0
    for _, row in sub.iterrows():
        cause = row["Cause_of_accident"]
        pct = row["percent"]
        color = other_color if cause == "Other (<5%)" else color_map.get(cause, "gray")

        plt.bar(x[i], pct, bottom=bottom, color=color, edgecolor="black")
        plt.text(x[i], bottom + pct / 2, f"{pct:.1f}%",
                 ha="center", va="center", fontsize=16, color="black", fontweight="bold")  # Verdoppelt (war 8)
        bottom += pct
        legend_labels.add(cause)

plt.xticks(x, severities_deutsch, rotation=45, ha="right", fontsize=20)  # Verdoppelt (ursprünglich 10)
plt.ylabel("Anteil (%)", fontsize=24)  # Verdoppelt (ursprünglich 12)
plt.title("Verteilung der Unfallursachen pro Unfallschwere (in %)",
          fontsize=28, weight="bold")  # Verdoppelt (ursprünglich 14)
plt.ylim(0, 100)

# Y-Achsen-Tick-Labels vergrößern
ax = plt.gca()
ax.tick_params(axis='y', labelsize=20)  # Verdoppelt (Standard ~10)

# Legende entfernt vom Hauptdiagramm

plt.tight_layout()
plt.savefig(os.path.join(folders["stacked"], "unfallursachen_nach_schweregrad_prozent_sortiert_alles.png"), dpi=300)
plt.close()

# =======================================================================================
# 6c) Separate Legende für erstes gestapeltes Balkendiagramm (mit Zusammenfassung)
# =======================================================================================
fig_legend1, ax_legend1 = plt.subplots(figsize=(6, 8))  # Schmaler aber höher für eine Spalte
ax_legend1.axis('off')

# Erstelle Legende mit Ursachen aus dem ersten Diagramm
legend_elements_1 = []
for cause in global_cause_order:
    if cause in legend_labels:
        label = "Sonstiges (<5%)" if cause == "Other (<5%)" else translate_cause(cause)
        color = other_color if cause == "Other (<5%)" else color_map.get(cause, "gray")
        legend_elements_1.append(Patch(facecolor=color, edgecolor="black", label=label))
if "Other (<5%)" in legend_labels and "Other (<5%)" not in global_cause_order:
    legend_elements_1.append(Patch(facecolor=other_color, edgecolor="black", label="Sonstiges (<5%)"))

# Erstelle die Legende in der Mitte des Bildes mit nur einer Spalte
legend1 = ax_legend1.legend(handles=legend_elements_1, 
                           loc='center',
                           title="Unfallursachen - Legende\n(Zusammengefasst)",
                           title_fontsize=24,
                           fontsize=18,
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           ncol=1)  # Nur eine Spalte - alles untereinander

plt.tight_layout()
plt.savefig(os.path.join(folders["stacked"], "unfallursachen_legende_zusammengefasst.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================================================================================
# 6b) Zusätzliches Diagramm: alle Klassen (ohne Zusammenfassen), nur >=5% beschriftet
# =======================================================================================
plt.figure(figsize=(12, 8))  # Angepasst für bessere Proportionen
legend_labels_all = set()

for i, severity in enumerate(severities):
    sub_all = grouped[grouped["Accident_severity"] == severity].sort_values("percent", ascending=False)
    bottom = 0.0
    for _, row in sub_all.iterrows():
        cause = row["Cause_of_accident"]
        pct = row["percent"]
        color = color_map.get(cause, "gray")

        plt.bar(x[i], pct, bottom=bottom, color=color, edgecolor="black")
        if pct >= 5:
            plt.text(x[i], bottom + pct / 2, f"{pct:.1f}%",
                     ha="center", va="center", fontsize=16, color="black", fontweight="bold")  # Verdoppelt (war 8)
        bottom += pct
        legend_labels_all.add(cause)

plt.xticks(x, severities_deutsch, rotation=45, ha="right", fontsize=20)  # Verdoppelt (ursprünglich 10)
plt.ylabel("Anteil (%)", fontsize=24)  # Verdoppelt (ursprünglich 12)

# HIER IST DIE ÄNDERUNG: pad=20 erhöht den Abstand zwischen Titel und Diagramm
plt.title("Verteilung der Unfallursachen pro Unfallschwere (in %)",
          fontsize=28, weight="bold", pad=20)  # pad=20 für mehr Abstand nach oben

plt.ylim(0, 100)

# Y-Achsen-Tick-Labels vergrößern
ax = plt.gca()
ax.tick_params(axis='y', labelsize=20)  # Verdoppelt (Standard ~10)

# Legende mit deutschen Bezeichnungen
legend_handles_all = []
for cause in global_cause_order:
    if cause in legend_labels_all:
        legend_handles_all.append(Patch(facecolor=color_map.get(cause, "gray"), edgecolor="black", 
                                        label=translate_cause(cause)))

# Legende entfernt vom Hauptdiagramm

plt.tight_layout()
plt.savefig(os.path.join(folders["stacked"], "unfallursachen_nach_schweregrad_prozent_sortiert_alle_klassen.png"), 
           dpi=300, bbox_inches='tight', pad_inches=0.3)  # pad_inches hinzugefügt gegen Abschneiden
plt.close()

# =======================================================================================
# 6d) Separate Legende für zweites gestapeltes Balkendiagramm (alle Klassen)
# =======================================================================================
fig_legend2, ax_legend2 = plt.subplots(figsize=(6, 12))  # Schmaler aber höher für eine Spalte mit vielen Einträgen
ax_legend2.axis('off')

# Erstelle Legende mit Ursachen aus dem zweiten Diagramm (alle Klassen)
legend_elements_2 = []
for cause in global_cause_order:
    if cause in legend_labels_all:
        label = translate_cause(cause)
        color = color_map.get(cause, "gray")
        legend_elements_2.append(Patch(facecolor=color, edgecolor="black", label=label))

# Erstelle die Legende in der Mitte des Bildes mit nur einer Spalte
legend2 = ax_legend2.legend(handles=legend_elements_2, 
                           loc='center',
                           title="Unfallursachen - Legende\n(Alle Klassen)",
                           title_fontsize=24,
                           fontsize=18,
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           ncol=1)  # Nur eine Spalte - alles untereinander

plt.tight_layout()
plt.savefig(os.path.join(folders["stacked"], "unfallursachen_legende_alle_klassen.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================================================================================
# 7) Top-10-Klassen über alle Spalten (15 Diagramme)
# =======================================================================================
print("🔍 Erstelle 15 kombinierte Top-10-Diagramme ...")
other_cols = [c for c in df.columns if c not in ["Accident_severity", "Cause_of_accident"]]

for severity in severities:
    top5_causes = (
        grouped[grouped["Accident_severity"] == severity]
        .sort_values("count", ascending=False)["Cause_of_accident"]
        .head(5)
        .tolist()
    )

    for cause in top5_causes:
        subset = df[(df["Accident_severity"] == severity) & (df["Cause_of_accident"] == cause)]
        if subset.empty:
            continue

        all_values = []
        for col in other_cols:
            all_values.extend(subset[col].dropna().astype(str).tolist())

        value_counts = pd.Series(all_values).value_counts(normalize=True).head(10) * 100
        if value_counts.empty:
            continue

        # Übersetze die Werte für die Anzeige
        translated_index = [translate_value(val) for val in value_counts.index]
        value_counts.index = translated_index

        plt.figure(figsize=(8, 5))
        value_counts_sorted = value_counts.sort_values()
        value_counts_sorted.plot(kind="barh", color="steelblue", edgecolor="black")
        plt.xlabel("Anteil (%)")
        plt.ylabel("Klassen (über alle Spalten)")
        plt.title(f"{translate_severity(severity)} – {translate_cause(cause)}\nTop 10 Klassen aus allen Spalten (in %)")

        for idx, value in enumerate(value_counts_sorted):
            if value >= 3:
                plt.text(value / 2, idx, f"{value:.1f}%", va="center", ha="center",
                         color="black", fontweight="bold", fontsize=8)
            else:
                plt.text(value + 0.3, idx, f"{value:.1f}%", va="center", ha="left",
                         color="black", fontweight="bold", fontsize=8)

        plt.tight_layout()
        safe_cause = str(cause).replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_sev = str(severity).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_name = os.path.join(folders["top10"], f"{safe_sev}_{safe_cause}_top10_allcols.png")
        plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

print("✅ 15 kombinierte Diagramme gespeichert.")

# =======================================================================================
# 8) Verteilungen aller restlichen Spalten pro Unfallschwere (Ordner 4)
# =======================================================================================
print("📊 Erstelle Verteilungen aller restlichen Spalten pro Unfallschwere (Ordner 4) ...")

for severity in severities:
    df_sev = df[df["Accident_severity"] == severity]
    for col in other_cols:
        counts = df_sev[col].value_counts(normalize=True) * 100
        if counts.empty:
            continue

        # Übersetze die Werte für die Anzeige
        translated_index = [translate_value(val) for val in counts.index]
        counts.index = translated_index

        plt.figure(figsize=(8, 5))
        counts_sorted = counts.sort_values(ascending=True)
        counts_sorted.plot(kind="barh", color="teal", edgecolor="black")
        plt.xlabel("Anteil (%)")
        plt.ylabel(translate_column_name(col))  # Übersetzte Spaltenbezeichnung
        plt.title(f"{translate_severity(severity)} – Verteilung von {translate_column_name(col)} (in %)")  # Übersetzte Spaltenbezeichnung

        for idx, value in enumerate(counts_sorted):
            if value >= 3:
                plt.text(value / 2, idx, f"{value:.1f}%", va="center", ha="center",
                         color="black", fontweight="bold", fontsize=8)
            else:
                plt.text(value + 0.3, idx, f"{value:.1f}%", va="center", ha="left",
                         color="black", fontweight="bold", fontsize=8)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)  # Manuelle Anpassung statt tight_layout
        safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_sev = str(severity).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_name = os.path.join(folders["spalten"], f"{safe_sev}_{safe_col}_verteilung.png")
        plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

print("✅ Verteilungsdiagramme gespeichert in:", folders["spalten"])

# =======================================================================================
# 9) NEU: Vergleichsdiagramme pro Spalte (Ordner 5) – gruppierte Balken
# =======================================================================================
print("📊 Erstelle Vergleichsdiagramme (alle Unfallschweren in einer Grafik) ...")

for col in other_cols:
    classes = sorted(df[col].dropna().astype(str).unique())
    if len(classes) == 0:
        continue

    # Berechne die maximalen Prozentwerte für jede Klasse über alle Schweregrade
    max_percentages = {}
    data_original = {}
    
    for severity in severities:
        df_sev = df[df["Accident_severity"] == severity]
        counts = (df_sev[col].dropna().astype(str).value_counts(normalize=True) * 100).to_dict()
        data_original[severity] = counts
        
        # Update maximale Prozente
        for c in classes:
            pct = counts.get(c, 0)
            if c not in max_percentages:
                max_percentages[c] = pct
            else:
                max_percentages[c] = max(max_percentages[c], pct)
    
    # Trenne Klassen in über/unter 3% basierend auf Maximum
    classes_above_3_temp = [c for c in classes if max_percentages[c] >= 3]
    classes_below_3 = [c for c in classes if max_percentages[c] < 3]
    
    # Prüfe ob "Other" oder "Sonstiges" bereits in den Klassen über 3% existiert
    existing_other_classes = [c for c in classes_above_3_temp if c in ["Other", "Sonstiges"]]
    
    # Entferne "Other" und "Sonstiges" aus den Klassen über 3%
    classes_above_3 = [c for c in classes_above_3_temp if c not in ["Other", "Sonstiges"]]
    
    # Füge existierende "Other"/"Sonstiges" zu den zusammenzufassenden Klassen hinzu
    classes_to_combine = classes_below_3 + existing_other_classes
    
    # Wenn es Klassen zum Zusammenfassen gibt (unter 3% ODER existierendes "Other"/"Sonstiges")
    if classes_to_combine:
        # Übersetze die Namen der zusammenzufassenden Klassen
        translated_to_combine = [translate_value(c) for c in classes_to_combine]
        
        # Neue Klassenliste mit einfachem "Sonstiges"
        new_classes_unsorted = classes_above_3 + ["_OTHER_"]
        
        # Berechne Daten für alle Klassen
        data_unsorted = {}
        for severity in severities:
            data_unsorted[severity] = []
            for c in classes_above_3:
                data_unsorted[severity].append(data_original[severity].get(c, 0))
            
            # Summiere alle zusammenzufassenden Klassen (inkl. existierendes "Other"/"Sonstiges")
            other_sum = sum(data_original[severity].get(c, 0) for c in classes_to_combine)
            data_unsorted[severity].append(other_sum)
        
        # Sortiere nach "Fatal injury" Werten (absteigend)
        if "Fatal injury" in data_unsorted:
            fatal_values = data_unsorted["Fatal injury"]
        else:
            # Fallback: nimm den ersten Schweregrad
            fatal_values = data_unsorted[severities[0]]
        
        # Erstelle Sortier-Indizes
        sorted_indices = sorted(range(len(fatal_values)), key=lambda i: fatal_values[i], reverse=True)
        
        # Sortiere Klassen und Daten
        new_classes = [new_classes_unsorted[i] for i in sorted_indices]
        translated_classes_unsorted = [translate_value(c) for c in classes_above_3] + ["Sonstiges"]
        translated_classes = [translated_classes_unsorted[i] for i in sorted_indices]
        
        data = {}
        for severity in severities:
            data[severity] = [data_unsorted[severity][i] for i in sorted_indices]
            
        # Erstelle separate Legende für "Sonstiges"
        if translated_to_combine:  # Nur wenn es tatsächlich Klassen gibt
            fig_legend, ax_legend = plt.subplots(figsize=(6, min(3, 1 + len(translated_to_combine) * 0.15)))
            ax_legend.axis('off')
            
            # Formatiere mit Zeilenumbruch nach jeweils 2 Klassen
            legend_lines = ["Sonstiges:"]
            for i in range(0, len(translated_to_combine), 2):
                if i + 1 < len(translated_to_combine):
                    legend_lines.append(f"{translated_to_combine[i]}, {translated_to_combine[i+1]}")
                else:
                    legend_lines.append(translated_to_combine[i])
            
            legend_text = "\n".join(legend_lines)
            
            ax_legend.text(0.5, 0.5, legend_text, 
                          transform=ax_legend.transAxes,
                          ha='center', va='center',
                          fontsize=11,
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray", alpha=0.9))
            
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Manuelle Anpassung
            safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")
            legend_name = os.path.join(folders["spalten_compare"], f"{safe_col}_sonstiges_legende.png")
            plt.savefig(legend_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
    else:
        # Keine Zusammenfassung nötig - aber trotzdem sortieren
        data_unsorted = {}
        for severity in severities:
            data_unsorted[severity] = [data_original[severity].get(c, 0) for c in classes]
        
        # Sortiere nach "Fatal injury" Werten (absteigend)
        if "Fatal injury" in data_unsorted:
            fatal_values = data_unsorted["Fatal injury"]
        else:
            # Fallback: nimm den ersten Schweregrad
            fatal_values = data_unsorted[severities[0]]
        
        # Erstelle Sortier-Indizes
        sorted_indices = sorted(range(len(fatal_values)), key=lambda i: fatal_values[i], reverse=True)
        
        # Sortiere Klassen und Daten
        new_classes = [classes[i] for i in sorted_indices]
        translated_classes = [translate_value(classes[i]) for i in sorted_indices]
        
        data = {}
        for severity in severities:
            data[severity] = [data_unsorted[severity][i] for i in sorted_indices]

    n_classes = len(new_classes)
    n_sev = len(severities)
    x = np.arange(n_classes)
    width = 0.8 / n_sev

    plt.figure(figsize=(max(40, n_classes * 3.2), 14.4))  # Höhe nochmals um 20% erhöht

    # Verwende die definierten matten Farben für die gruppierten Balken
    for idx, severity in enumerate(severities):
        offsets = x - 0.4 + width/2 + idx * width
        bar_color = severity_colors.get(severity, "#808080")  # Matte Farben
        plt.bar(offsets, data[severity], width=width, 
                label=translate_severity(severity),  # Deutsche Labels in Legende
                color=bar_color, edgecolor="black")

    # Bestimme für jede Klasse, ob mindestens ein Wert <10% ist
    classes_with_small_values = []
    for class_idx in range(n_classes):
        has_small_value = False
        for severity in severities:
            if data[severity][class_idx] < 10 and data[severity][class_idx] > 0:
                has_small_value = True
                break
        classes_with_small_values.append(has_small_value)

    # Prozentzahlen platzieren
    for idx, severity in enumerate(severities):
        offsets = x - 0.4 + width/2 + idx * width
        
        for class_idx, (xx, val) in enumerate(zip(offsets, data[severity])):
            if val > 0:
                # Bei 7 oder mehr Klassen etwas kleinere Schrift verwenden
                if n_classes >= 7:
                    base_size = 30  # Dreifach (ursprünglich 10)
                else:
                    base_size = 42  # Dreifach (ursprünglich 14)
                
                # Wenn diese Klasse mindestens einen kleinen Wert hat, alle Werte über den Balken
                if classes_with_small_values[class_idx]:
                    # Über dem Balken, vertikal, mit festem Abstand
                    # Verwende die Y-Position des Balkens plus einen festen Offset
                    y_position = val + 2.0  # Fester Abstand von 2.0 Einheiten über dem Balken
                    plt.text(xx, y_position, f"{val:.1f}%",
                            ha="center", va="bottom",
                            fontsize=base_size, color="black", fontweight="bold", 
                            rotation=90)  # Vertikal über dem Balken
                else:
                    # Normal im Balken platzieren
                    if val < 10:
                        plt.text(xx, val / 2, f"{val:.1f}%", 
                                ha="center", va="center",
                                fontsize=base_size, color="black", fontweight="bold", 
                                rotation=90)
                    elif val < 20:
                        plt.text(xx, val / 2, f"{val:.1f}%", 
                                ha="center", va="center",
                                fontsize=max(base_size-3, 27), color="black", fontweight="bold", 
                                rotation=90)
                    else:
                        plt.text(xx, val / 2, f"{val:.1f}%", 
                                ha="center", va="center",
                                fontsize=max(base_size-6, 24), color="black", fontweight="bold", 
                                rotation=90)

    # Formatiere X-Achsen-Labels mit Zeilenumbrüchen nach jedem Wort
    formatted_labels = []
    for label in translated_classes:
        # Ersetze Leerzeichen durch Zeilenumbrüche
        formatted_label = label.replace(" ", "\n")
        formatted_labels.append(formatted_label)
    
    # Setze X-Ticks ohne Labels (wir fügen sie manuell hinzu)
    plt.xticks(x, [''] * len(x))
    
    # Zeichne die X-Achsen-Labels manuell mit versetzten Positionen
    ax = plt.gca()
    
    # Entferne die Standard-Tick-Markierungen
    ax.tick_params(axis='x', which='both', length=0)
    
    for i, (pos, label) in enumerate(zip(x, formatted_labels)):
        # Abwechselnd hoch und tief
        if i % 2 == 0:
            # Hohe Position (näher an der Achse)
            y_offset = -0.02  # Sehr nah an der Achse
            # Keine Tick-Linie für hohe Positionen
        else:
            # Tiefe Position (weiter weg von der Achse)
            y_offset = -0.12  # Weiter weg von der Achse
            tick_length = 0.08  # Längere Tick-Linie
        
        # Füge den Text hinzu
        trans = ax.get_xaxis_transform()
        ax.text(pos, y_offset, label, 
                transform=trans,
                ha='center', va='top',
                fontsize=36, fontweight='normal')  # Gleiche Größe wie Y-Achse (36pt)
        
        # Zeichne die Tick-Markierung nur für tiefe Positionen
        if i % 2 == 1:
            ax.plot([pos, pos], [0, -tick_length], 
                    transform=trans,
                    color='black', linewidth=1, clip_on=False)
    
    # Mehr Platz unter dem Plot für die versetzten Labels
    # wird automatisch durch bbox_inches='tight' beim Speichern optimiert
    
    plt.ylabel("Anteil (%)", fontsize=44)  # Auf 44pt gesetzt
    plt.title(f"Verteilung von {translate_column_name(col)} nach Unfallschwere", fontsize=44, fontweight='bold')  # Auf 44pt gesetzt
    plt.legend(fontsize=44)  # Bleibt bei 44pt
    
    # Y-Achsen-Tick-Labels verdreifachen
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=36)  # Verdreifacht (Standard war ~12)
    # tight_layout entfernt wegen Konflikten mit manueller Positionierung

    safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_name = os.path.join(folders["spalten_compare"], f"{safe_col}_vergleich.png")
    plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("✅ Vergleichsdiagramme gespeichert in:", folders["spalten_compare"])
print("\n🎉 Alle Diagramme und Dateien wurden erfolgreich erzeugt und strukturiert gespeichert.")