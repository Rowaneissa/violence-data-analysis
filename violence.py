import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = kagglehub.dataset_download("whenamancodes/violence-against-women-girls")

df=pd.read_csv(r'C:\Users\carnival\.cache\kagglehub\datasets\whenamancodes\violence-against-women-girls\versions\1\\Violence Against Women  Girls Data.csv')
df=df.dropna(subset=['Value'])
df['Value'] = df['Value'].fillna(df['Value'].mean())
x = df.groupby('Gender')['Value'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=x.index, y=x.values, palette=["#FF69B4", "#87CEFA"])
plt.title(
    "Voices on Violence: \n Gender Differences in Justifying Abuse",
    fontsize=14,
    fontweight='bold',
    color='darkblue')
plt.xlabel("Gender",fontweight='bold')
plt.ylabel("Average",fontweight='bold')
plt.show()
area = ['Urban', 'Rural']
area_df = df[df['Demographics Response'].isin(area)]
avg_by_area = area_df.groupby('Demographics Response')['Value'].mean().sort_values(ascending=False)
sns.barplot(x=avg_by_area.index, y=avg_by_area.values)
plt.title("Justification of Violence by Area", fontsize=16, fontweight='bold', color='darkgreen')
plt.xlabel("Area Type", fontsize=12, fontweight='bold')
plt.ylabel("Average Agreement (%)", fontsize=12, fontweight='bold')
plt.show()
edu=['Higher','No education','Primary','Secondary']
edu_df=df[df['Demographics Response'].isin(edu)]
avgedu=edu_df.groupby('Demographics Response')['Value'].mean()
sns.barplot(x=avgedu.index, y=avgedu.values,palette="BuPu")
plt.title("Justification of Violence by Education Level", fontsize=16, fontweight='bold', color='violet')
plt.xlabel("Education Level", fontsize=12, fontweight='bold')
plt.ylabel("Average Agreement (%)", fontsize=12, fontweight='bold')
plt.xticks(rotation=15)
plt.show()