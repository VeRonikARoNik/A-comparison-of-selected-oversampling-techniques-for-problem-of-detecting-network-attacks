import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
file_path = "F:\\iot_data\\rt-iot2022\\output\\multi_steps_results.xlsx"

# Lista arkuszy do przetworzenia
sheets = ["NoOversampling", "SMOTE_ADASYN", "SMOTE_ADASYN_GAN"]

# Lista metryk do wizualizacji
numeric_columns = ["Accuracy", "Precision1", "Recall1", "F11", "AUC", "CV_Accuracy"]

for sheet in sheets:
    # Wczytanie danych z danego arkusza
    df = pd.read_excel(file_path, sheet_name=sheet)
    # Ujednolicenie nazw kolumn (usunięcie nawiasów, spacji)
    df.columns = df.columns.str.replace("[()]", "", regex=True).str.replace(" ", "_")
    # Konwersja wybranych kolumn na typ numeryczny
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Sortowanie po kolumnie Fraction
    df = df.sort_values(by=["Fraction"])
    
    # -----------------------------
    # 1. Arkusz NoOversampling
    # -----------------------------
    if sheet == "NoOversampling":
        # Dla arkusza NoOversampling – wykres zbiorczy
        for metric in numeric_columns:
            plt.figure(figsize=(14, 7))
            title = f"No Oversampling - {metric}"
            x_axis = "Fraction"
            hue = "Classifier"
            ax = sns.barplot(data=df, x=x_axis, y=metric, hue=hue)
            
            # Dodanie wartości liczbowych na słupkach
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.4f}", 
                            (p.get_x() + p.get_width() / 2., p.get_height() / 2.), 
                            ha='center', va='center', fontsize=9, color='black', rotation=90)
            
            plt.title(title)
            plt.xlabel(x_axis)
            plt.ylabel(metric)
            plt.legend(title=hue)
            plt.xticks(rotation=0, ha="center")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            plt.figtext(
                0.5, -0.2, 
                f"Wykres przedstawia porównanie metryki {metric} dla różnych klasyfikatorów "
                f"z frakcją w arkuszu {sheet}.",
                wrap=True, 
                horizontalalignment='center', 
                fontsize=10
            )
            plt.show()
    
    # ----------------------------------------
    # 2. Arkusze SMOTE_ADASYN i SMOTE_ADASYN_GAN
    # ----------------------------------------
    elif sheet in ["SMOTE_ADASYN", "SMOTE_ADASYN_GAN"]:
        # Jeśli arkusz zawiera kolumny 'Oversampler' i 'Ratio', łączymy je w jedną etykietę
        if "Oversampler" in df.columns and "Ratio" in df.columns:
            df["Oversampling_Info"] = df["Oversampler"] + "\nRatio: " + df["Ratio"].astype(str)
        else:
            df["Oversampling_Info"] = "Brak_inf."
        
        # Pętla po każdej metryce
        for metric in numeric_columns:
            # Zamiast pojedynczego wykresu z pętlą po Fraction
            # wykorzystujemy catplot, aby automatycznie stworzyć facet-grid:
            g = sns.catplot(
                data=df,
                x="Oversampling_Info",   # Kategoria na osi X
                y=metric,                # Metryka do wizualizacji
                hue="Classifier",        # Różne kolory słupków dla każdego klasyfikatora
                col="Fraction",          # Podział na kolumny dla każdej Fraction
                kind="bar",
                col_wrap=3,             # Maks. liczba kolumn w jednym wierszu (jeśli Fraction jest więcej)
                sharey=False,           # Każda kolumna może mieć własną oś Y
                height=4,               # Wysokość każdego wykresu (w calach)
                aspect=1.2              # Stosunek szerokości do wysokości
            )
            
            # Tytuł całej siatki
            g.fig.suptitle(f"{sheet} - {metric}", y=1.05, fontsize=14)
            
            # Dodanie wartości liczbowych na słupkach
            # (w catplot / FacetGrid musimy przejść po każdej osi w g.axes.flat)
            for ax in g.axes.flat:
                for container in ax.containers:
                    # Utworzenie listy etykiet z wysokości słupków
                    labels = [f"{bar.get_height():.4f}" for bar in container]
                    # bar_label (Matplotlib 3.4+) umożliwia szybkie dodanie etykiet
                    ax.bar_label(
                        container, 
                        labels=labels, 
                        label_type='center', 
                        rotation=90, 
                        color='black', 
                        fontsize=8
                    )
            
            # Obracamy etykiety na osi X dla czytelności
            for ax in g.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            
            plt.show()
