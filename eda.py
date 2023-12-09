import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_csv('heart_disease_data.csv')
data=data.dropna(how='any')
data.shape
data.columns
data.describe()


fig = plt.figure(figsize=(18,15))
gs = fig.add_gridspec(3,3)
#gs.update(wspace=0.5, hspace=0.25)
#ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])
ax6 = fig.add_subplot(gs[2,0])
ax7 = fig.add_subplot(gs[2,1])
ax8 = fig.add_subplot(gs[2,2])
color_palette = ["#800000","#8000ff","#6aac90","#5833ff","#da8829"]

#Sex
sns.countplot(ax=ax1,data=data,x='sex',palette=color_palette)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Exang count
sns.countplot(ax=ax2,data=data,x='exang',palette=color_palette)
ax2.set_xlabel("")
ax2.set_ylabel("")

# Ca count
sns.countplot(ax=ax3,data=data,x='ca',palette=color_palette)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Cp count
sns.countplot(ax=ax4,data=data,x='cp',palette=color_palette)
ax4.set_xlabel("")
ax4.set_ylabel("")

# Fbs count
sns.countplot(ax=ax5,data=data,x='fbs',palette=color_palette)
ax5.set_xlabel("")
ax5.set_ylabel("")

# Restecg count
sns.countplot(ax=ax6,data=data,x='restecg',palette=color_palette)
ax6.set_xlabel("")
ax6.set_ylabel("")

# Slope count
sns.countplot(ax=ax7,data=data,x='slope',palette=color_palette)
ax7.set_xlabel("")
ax7.set_ylabel("")

# Thal count
sns.countplot(ax=ax8,data=data,x='thal',palette=color_palette)
ax8.set_xlabel("")
ax8.set_ylabel("")



i = ['age', 'trestbps', 'chol','thalach','oldpeak']
for j in i:
    plt.hist(data[j],bins = 60, color = 'slateblue')
    plt.title(j)
    plt.show()


dt=pd.read_csv('heart_disease_data.csv')
dt.rename(columns={'sex':'gender'},inplace=True)
dt.head()
#change the gender into categorical data
dt['gender'][dt['gender']==0]='female'
dt['gender'][dt['gender']==1]='male'
dt.head()
#bar plot of target by gender
sns.barplot(data=dt,x='gender',y='target')



print("Percentage of females heart disease:",dt["target"][dt["gender"]=="female"].value_counts(normalize=True)[1]*100)
print("Percentage of males heart disease:",dt["target"][dt["gender"]=="male"].value_counts(normalize=True)[1]*100)