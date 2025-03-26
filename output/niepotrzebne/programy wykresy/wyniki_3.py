import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
# Wczytanie danych z pliku Excel dla arkuszy NoOversampling i SMOTE_ADASYN
file_path = "F:\\iot_data\\rt-iot2022\\output\\multi_steps_results.xlsx"
sheets = ["NoOversampling", "SMOTE_ADASYN"]

# Lista metryk do wizualizacji
numeric_columns = ["Accuracy", "Precision1", "Recall1", "F11", "AUC", "CV_Accuracy"]

for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.replace("[()]", "", regex=True).str.replace(" ", "_")
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    if sheet == "SMOTE_ADASYN":
        df["Oversampling_Info"] = df["Oversampler"] + "\nRatio: " + df["Ratio"].astype(str)
    
    df = df.sort_values(by=["Fraction"])
    
    for metric in numeric_columns:
        if sheet == "NoOversampling":
            plt.figure(figsize=(14, 7))
            title = f"No Oversampling - {metric}"
            x_axis = "Fraction"
            hue = "Classifier"
            ax = sns.barplot(data=df, x=x_axis, y=metric, hue=hue)
            
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
            
            plt.figtext(0.5, -0.2, f"Wykres przedstawia porównanie metryki {metric} dla różnych klasyfikatorów "
                        f"z frakcją w arkuszu {sheet}.", 
                        wrap=True, horizontalalignment='center', fontsize=10)
            plt.show()
        
        else:
            for fraction in sorted(df["Fraction"].unique()):
                plt.figure(figsize=(14, 7))
                df_subset = df[df["Fraction"] == fraction]
                title = f"{sheet} - {metric} (Fraction {fraction})"
                x_axis = "Oversampling_Info"
                hue = "Classifier"
                
                ax = sns.barplot(data=df_subset, x=x_axis, y=metric, hue=hue)
                
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
                
                plt.figtext(0.5, -0.2, f"Wykres przedstawia porównanie metryki {metric} dla różnych klasyfikatorów "
                            f"z frakcją {fraction} w arkuszu {sheet}. "
                            f"Dla {sheet}, uwzględniono rodzaj oversamplingu (SMOTE / ADASYN) oraz Ratio.", 
                            wrap=True, horizontalalignment='center', fontsize=10)
                plt.show()
