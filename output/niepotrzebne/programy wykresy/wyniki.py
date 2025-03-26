import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych z pliku Excel dla arkuszy NoOversampling i SMOTE_ADASYN
file_path = "F:\\iot_data\\rt-iot2022\\output\\multi_steps_results.xlsx"



# Lista arkuszy do analizy
sheets = ["NoOversampling", "SMOTE_ADASYN"]

# Wczytanie danych dla każdego arkusza i generowanie wykresów
for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Poprawa nazw kolumn
    df.columns = df.columns.str.replace("[()]", "", regex=True).str.replace(" ", "_")

    # Lista metryk do wizualizacji
    numeric_columns = ["Accuracy", "Precision1", "Recall1", "F11", "AUC", "CV_Accuracy"]

    # Konwersja wartości numerycznych na float
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Tworzenie dodatkowej kolumny do oznaczenia oversamplingu
    if sheet == "SMOTE_ADASYN":
        df["Oversampling_Info"] = df["Oversampler"] + " | Ratio: " + df["Ratio"].astype(str) + " | Frac: " + df["Fraction"].astype(str)

    # Tworzenie wykresów dla każdej metryki
    for metric in numeric_columns:
        plt.figure(figsize=(16, 7))

        if sheet == "NoOversampling":
            title = f"No Oversampling - {metric}"
            x_axis = "Fraction"
            hue = "Classifier"
        else:
            title = f"{sheet} - {metric}"
            x_axis = "Oversampling_Info"
            hue = "Classifier"

        ax = sns.barplot(data=df, x=x_axis, y=metric, hue=hue)

        # Dodanie wartości nad słupkami
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5), 
                        textcoords='offset points')

        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(metric)
        plt.legend(title=hue)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Opis pod wykresem
        plt.figtext(0.5, -0.2, f"Wykres przedstawia porównanie metryki {metric} dla różnych klasyfikatorów "
                    f"w zależności od wartości {x_axis} w arkuszu {sheet}. "
                    f"Dla {sheet}, uwzględniono rodzaj oversamplingu (SMOTE / ADASYN), Ratio (np. 0.25, 0.5, 1) oraz Frakcję.", 
                    wrap=True, horizontalalignment='center', fontsize=10)

        # Wyświetlenie wykresu
        plt.show()
