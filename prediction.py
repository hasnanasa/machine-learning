{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMDYS7SQPFsLCKnrYXPoyE7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hasnanasa/machine-learning/blob/main/prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Prediction des prix immobiliers for Morocco**"
      ],
      "metadata": {
        "id": "T3ItYR7W-FNe"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n2gTBsbK-ASi",
        "outputId": "eab2f21a-6dc7-4cfd-e065-daa64f99216d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 324 entries, 0 to 323\n",
            "Data columns (total 8 columns):\n",
            " #   Column         Non-Null Count  Dtype  \n",
            "---  ------         --------------  -----  \n",
            " 0   price_£        324 non-null    float64\n",
            " 1   proprety type  324 non-null    object \n",
            " 2   surface        324 non-null    float64\n",
            " 3   bedroom        324 non-null    int64  \n",
            " 4   bathroom       324 non-null    int64  \n",
            " 5   address        324 non-null    object \n",
            " 6   city           324 non-null    object \n",
            " 7   principale     141 non-null    object \n",
            "dtypes: float64(2), int64(2), object(4)\n",
            "memory usage: 20.4+ KB\n",
            "None\n",
            "     price_£ proprety type   surface  bedroom  bathroom  \\\n",
            "0   239769.0     Apartment   1399.31        2         2   \n",
            "1  1798271.0         House  14531.28        6         6   \n",
            "2   434582.0         House   4305.56        4         6   \n",
            "3   129875.0     Apartment    893.40        2         2   \n",
            "4   199808.0     Apartment   1367.02        3         2   \n",
            "\n",
            "                                             address        city  \\\n",
            "0    Ennakhil-(Palmeraie), Marrakech, Marrakesh-Safi   Marrakech   \n",
            "1    Ennakhil-(Palmeraie), Marrakech, Marrakesh-Safi   Marrakech   \n",
            "2      Ménara, Marrakech, Marrakesh-Tensift-El Haouz   Marrakech   \n",
            "3  Guéliz, Marrakech, Marrakesh-Tensift-El Haouz ...   Marrakech   \n",
            "4                Anfa, Casablanca, Casablanca-Settat  Casablanca   \n",
            "\n",
            "                   principale  \n",
            "0              Marrakesh-Safi  \n",
            "1              Marrakesh-Safi  \n",
            "2  Marrakesh-Tensift-El Haouz  \n",
            "3  Marrakesh-Tensift-El Haouz  \n",
            "4           Casablanca-Settat  \n",
            "proprety type\n",
            "House        246\n",
            "Apartment     77\n",
            "Rural          1\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Load data\n",
        "df = pd.read_csv('housing_sales_ma_.csv')\n",
        "\n",
        "# Initial inspection\n",
        "print(df.info())\n",
        "print(df.head())\n",
        "print(df['proprety type'].value_counts())"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**data cleaning**"
      ],
      "metadata": {
        "id": "2W1uZeA9-2UN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Rename columns for ease\n",
        "df.rename(columns={\n",
        "    'price_£': 'price',\n",
        "    'proprety type': 'property_type',\n",
        "    'surface': 'surface_m2',\n",
        "    'bedroom': 'bedrooms',\n",
        "    'bathroom': 'bathrooms',\n",
        "    'address': 'address',\n",
        "    'city': 'city',\n",
        "    'principale': 'region'\n",
        "}, inplace=True)\n",
        "\n",
        "# 2. Keep only apartments\n",
        "df = df[df['property_type'] == 'Apartment'].copy()\n",
        "\n",
        "# 3. Remove rows with missing critical values\n",
        "df.dropna(subset=['price', 'surface_m2', 'address', 'city'], inplace=True)\n",
        "\n",
        "# 4. Convert price and surface to numeric (already float, but ensure)\n",
        "df['price'] = pd.to_numeric(df['price'], errors='coerce')\n",
        "df['surface_m2'] = pd.to_numeric(df['surface_m2'], errors='coerce')\n",
        "df.dropna(subset=['price', 'surface_m2'], inplace=True)\n",
        "\n",
        "# 5. Remove unrealistic values (outliers)\n",
        "# Surface: keep between 20 m² and 500 m² for apartments (typical in Morocco)\n",
        "df = df[(df['surface_m2'] >= 20) & (df['surface_m2'] <= 500)]\n",
        "\n",
        "# Price: remove below 50k MAD and above 10M MAD (reasonable for apartments)\n",
        "df = df[(df['price'] >= 50000) & (df['price'] <= 10000000)]\n",
        "\n",
        "# 6. Extract quartier (neighborhood) from address\n",
        "# Address format: \"Quartier, City, Region\" or \"Quartier, City, Region postal\"\n",
        "def extract_quartier(addr):\n",
        "    if pd.isna(addr):\n",
        "        return 'Unknown'\n",
        "    parts = str(addr).split(',')\n",
        "    return parts[0].strip()\n",
        "\n",
        "df['quartier'] = df['address'].apply(extract_quartier)\n",
        "\n",
        "# 7. Clean city names (some are 'Unknown' or misspelled)\n",
        "df['city'] = df['city'].replace('Unknown', np.nan)\n",
        "df.dropna(subset=['city'], inplace=True)\n",
        "df['city'] = df['city'].str.strip().str.title()\n",
        "\n",
        "# 8. Remove rows with too few samples per quartier/city (optional)\n",
        "quartier_counts = df['quartier'].value_counts()\n",
        "df = df[df['quartier'].isin(quartier_counts[quartier_counts >= 3].index)]\n",
        "\n",
        "city_counts = df['city'].value_counts()\n",
        "df = df[df['city'].isin(city_counts[city_counts >= 5].index)]\n",
        "\n",
        "# Final cleaned dataset\n",
        "print(f\"Cleaned dataset shape: {df.shape}\")\n",
        "print(df[['price', 'surface_m2', 'quartier', 'city']].head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "g2P9-fjb-kV8",
        "outputId": "acde1aaa-d540-4605-f575-62e9a1bb8a34"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cleaned dataset shape: (8, 9)\n",
            "        price  surface_m2 quartier       city\n",
            "63   267743.0       166.0   Guéliz  Marrakech\n",
            "84   259750.0        65.0    Agdal  Marrakech\n",
            "117  209798.0       160.0   Guéliz  Marrakech\n",
            "118  362651.0       165.0    Agdal  Marrakech\n",
            "135  649376.0       410.0    Agdal  Marrakech\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**data visualization**"
      ],
      "metadata": {
        "id": "qIn8Kkgt-tQh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Set style\n",
        "sns.set_style('whitegrid')\n",
        "plt.figure(figsize=(12, 8))\n",
        "\n",
        "# Distribution of prices\n",
        "plt.subplot(2, 2, 1)\n",
        "sns.histplot(df['price'], bins=50, kde=True)\n",
        "plt.title('Distribution of Apartment Prices (MAD)')\n",
        "plt.xlabel('Price')\n",
        "\n",
        "# Distribution of surfaces\n",
        "plt.subplot(2, 2, 2)\n",
        "sns.histplot(df['surface_m2'], bins=30, kde=True)\n",
        "plt.title('Distribution of Surface Area (m²)')\n",
        "plt.xlabel('Surface m²')\n",
        "\n",
        "# Price vs Surface scatter\n",
        "plt.subplot(2, 2, 3)\n",
        "sns.scatterplot(data=df, x='surface_m2', y='price', alpha=0.5)\n",
        "plt.title('Price vs Surface')\n",
        "plt.xlabel('Surface m²')\n",
        "plt.ylabel('Price (MAD)')\n",
        "\n",
        "# Average price by top 10 cities\n",
        "plt.subplot(2, 2, 4)\n",
        "city_price = df.groupby('city')['price'].mean().sort_values(ascending=False).head(10)\n",
        "sns.barplot(x=city_price.values, y=city_price.index)\n",
        "plt.title('Average Price by City')\n",
        "plt.xlabel('Average Price (MAD)')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Average price by top 15 quartiers\n",
        "plt.figure(figsize=(12, 6))\n",
        "quartier_price = df.groupby('quartier')['price'].mean().sort_values(ascending=False).head(15)\n",
        "sns.barplot(x=quartier_price.values, y=quartier_price.index)\n",
        "plt.title('Average Price by Quartier (Top 15)')\n",
        "plt.xlabel('Average Price (MAD)')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "X3T9m1Hd-ncy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "\n",
        "# Assume df is your cleaned dataset with columns: price, surface_m2, quartier, city\n",
        "\n",
        "# Compute target encoding for quartier and city\n",
        "global_mean_price = df['price'].mean()\n",
        "\n",
        "# For quartier: map each quartier to its mean price, fallback to city mean then global\n",
        "quartier_mean = df.groupby('quartier')['price'].mean().to_dict()\n",
        "city_mean = df.groupby('city')['price'].mean().to_dict()\n",
        "\n",
        "def encode_quartier(q, c):\n",
        "    if q in quartier_mean:\n",
        "        return quartier_mean[q]\n",
        "    elif c in city_mean:\n",
        "        return city_mean[c]\n",
        "    else:\n",
        "        return global_mean_price\n",
        "\n",
        "def encode_city(c):\n",
        "    if c in city_mean:\n",
        "        return city_mean[c]\n",
        "    else:\n",
        "        return global_mean_price\n",
        "\n",
        "# Apply encoding\n",
        "df['quartier_encoded'] = df.apply(lambda row: encode_quartier(row['quartier'], row['city']), axis=1)\n",
        "df['city_encoded'] = df['city'].apply(encode_city)\n",
        "\n",
        "# Features: surface + encoded quartier + encoded city\n",
        "X = df[['surface_m2', 'quartier_encoded', 'city_encoded']]\n",
        "y = df['price']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict for a new apartment with unseen quartier and city\n",
        "def predict_price(surface, quartier, city):\n",
        "    # Encode quartier and city using the same fallback logic\n",
        "    q_enc = encode_quartier(quartier, city)\n",
        "    c_enc = encode_city(city)\n",
        "    X_new = pd.DataFrame([[surface, q_enc, c_enc]], columns=['surface_m2', 'quartier_encoded', 'city_encoded'])\n",
        "    return model.predict(X_new)[0]\n",
        "\n",
        "# Example: completely new quartier 'NewDistrict' and new city 'NewCity'\n",
        "price_pred = predict_price(3, 'Nasser', 'Temara')\n",
        "print(f\"Predicted price: {price_pred:,.0f} MAD\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "km1qLiaFBZsP",
        "outputId": "10e1c240-cde7-4514-baab-7b99c3c38972"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Predicted price: 179,649 MAD\n"
          ]
        }
      ]
    }
  ]
}