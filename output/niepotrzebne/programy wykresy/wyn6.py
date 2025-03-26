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
            
            plt.tight_layout()
            plt.show()
    
    elif sheet in ["SMOTE_ADASYN", "SMOTE_ADASYN_GAN"]:
        if "Oversampler" in df.columns and "Ratio" in df.columns:
            df["Oversampling_Info"] = df["Oversampler"] + "\nRatio: " + df["Ratio"].astype(str)
        else:
            df["Oversampling_Info"] = "Brak_inf."
        
        # Tworzymy wykresy w pętli po metrykach przy pomocy catplot
        for metric in numeric_columns:
            g = sns.catplot(
                data=df,
                x="Oversampling_Info",   
                y=metric,                
                hue="Classifier",        
                col="Fraction",          
                kind="bar",
                col_wrap=3,         
                sharey=False,       
                height=5,           # większa wysokość
                aspect=1.5          # większa szerokość (poprawia czytelność)
            )
            
            g.fig.suptitle(f"{sheet} - {metric}", y=1.02, fontsize=14)
            
            # Dodanie wartości liczbowych na słupkach
            for ax in g.axes.flat:
                # zmniejszenie czcionki etykiet
                ax.tick_params(axis='x', labelsize=8)  
                
                for container in ax.containers:
                    labels = [f"{bar.get_height():.4f}" for bar in container]
                    ax.bar_label(
                        container, 
                        labels=labels, 
                        label_type='center', 
                        rotation=90, 
                        color='black', 
                        fontsize=8
                    )
                
                # Obracamy etykiety na osi X dla czytelności
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            
            # Zwiększamy dolny margines, aby etykiety się mieściły
            g.fig.subplots_adjust(bottom=0.25)
            
            plt.show()
