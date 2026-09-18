import matplotlib.pyplot as plt
import seaborn as sns


def plot_risk_distribution(df):
    plt.figure()
    sns.countplot(data=df, x="risk_level")
    plt.title("Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Students")
    plt.show()


def plot_correlation(df):
    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title("Feature Correlation Matrix")
    plt.show()
