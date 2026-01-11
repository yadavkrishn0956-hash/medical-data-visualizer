import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("medical_examination.csv")


# ---------------------------
# Add overweight column
# ---------------------------
# BMI = weight (kg) / (height (m))^2
df['overweight'] = (
    df['weight'] / ((df['height'] / 100) ** 2) > 25
).astype(int)


# ---------------------------
# Normalize cholesterol & glucose
# 0 = good, 1 = bad
# ---------------------------
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# ---------------------------
# Categorical Plot
# ---------------------------
def draw_cat_plot():
    # Prepare data for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group and count
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Draw catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    return fig


# ---------------------------
# Heat Map
# ---------------------------
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Correlation matrix
    corr = df_heat.corr()

    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5}
    )

    return fig
