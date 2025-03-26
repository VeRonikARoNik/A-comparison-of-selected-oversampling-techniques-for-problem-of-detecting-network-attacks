import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "F:\\iot_data\\rt-iot2022\\output\\multi_steps_results.xlsx"
sheets = ["NoOversampling", "SMOTE_ADASYN", "SMOTE_ADASYN_GAN"]
numeric_columns = ["Accuracy", "Precision1", "Recall1", "F11", "AUC", "CV_Accuracy"]

for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.replace("[()]", "", regex=True).str.replace(" ", "_")
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    df = df.sort_values(by=["Fraction"])
    
    if sheet == "NoOversampling":
        for metric in numeric_columns:
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
            plt.tight_layout()
            plt.show()
    
    elif sheet in ["SMOTE_ADASYN", "SMOTE_ADASYN_GAN"]:
        if "Oversampler" in df.columns and "Ratio" in df.columns:
            df["Oversampling_Info"] = df["Oversampler"] + "\nRatio: " + df["Ratio"].astype(str)
        else:
            df["Oversampling_Info"] = "Brak_inf."
        
        for metric in numeric_columns:
            for fraction in sorted(df["Fraction"].unique()):
                df_subset = df[df["Fraction"] == fraction]
                
                g = sns.catplot(
                    data=df_subset,
                    x="Oversampling_Info",
                    y=metric,
                    hue="Classifier",
                    kind="bar",
                    height=5,
                    aspect=1.5
                )
                # Nieco niższa wartość y i dodatkowe zmniejszenie top:
                g.fig.suptitle(
                    f"{sheet} - {metric}\nFraction = {fraction}", 
                    y=1.02, 
                    fontsize=14
                )
                
                # Dodanie wartości liczbowych
                for ax in g.axes.flat:
                    for container in ax.containers:
                        labels = [f"{bar.get_height():.4f}" for bar in container]
                        ax.bar_label(container, labels=labels, label_type='center',
                                     rotation=90, color='black', fontsize=8)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                
                # Dodatkowe regulacje marginesów:
                g.fig.subplots_adjust(top=0.85, bottom=0.3)
                
                plt.show()
