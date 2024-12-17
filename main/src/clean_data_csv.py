import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
# Fonction pour remplacer les noms de pays par leurs noms correspondants
def replace_country(c_str):
    # Dictionnaire de correspondance des noms de mois avec leurs numéros
    falseCountries_to_correct = {
        'usa': "états-unis d'amérique", 'brooklyn': "états-unis d'amérique", "etats-unis": "états-unis d'amérique",
        'deutschland': 'allemagne', 'malta': 'malte', 'france métropolitaine': 'france', 'malaysia': 'malaisie',
        'south africa': 'afrique de sud', 'nederlanad': 'pays-bas', 'united states': "états-unis d'amérique",
        'germany': 'allemagne', 'spain': 'espagne', 'belgium': 'belgique',
        'sweden': 'suède', 'italy': 'italie', 'hungary': 'hongrie', 'norway': 'norvège', 'Croatia': 'croatie',
        'new zealand': 'nouvelle-zélande', 'nederland': 'pays-bas', 'saudi arabia': 'arabie saoudite',
        'switzerland': 'suisse', 'denmark': 'danemark', 'czech republic': 'république tchèque',
        'poland': 'pologne', "greece": "grèce", "mexico": "mexique", "irlande (eire)": "irlande", "japan": "japon",
        "thailand": "thailande", "russia": "russie"
    }
    for c, cc in falseCountries_to_correct.items():
        if not pd.isna(c_str) and c in c_str:
            return c_str.replace(c, cc)
    return c_str  # Retourne la chaîne inchangée si aucun remplacement n'est effectué


# Fonction pour remplacer les noms de mois par leurs numéros correspondants
def replace_month(date_str):
    # Dictionnaire de correspondance des noms de mois avec leurs numéros
    month_to_num = {
        'Janvier': '01', 'janvier': '01', 'JANV': '01', 'Février': '02', 'Vévrier': '02', 'FEV': '02', 'Mars': '03',
        'MARS': '03', 'mars': '03', 'Avril': '04', 'Mai': '05', 'MAI': '05', 'MA': '05', 'Juin': '06', 'June': '06',
        'JUIN': '06', 'JUN': '06', 'Jun': '06',
        'Juillet': '07', 'Jullet': '07', 'Juill': '07', 'Aout': '08', 'Août': '08', 'Septembre': '09',
        'September': '09', 'Sept': '09', 'sept': '09', 'Saptembre': '09', 'Spt': '09', 'Octobre': '10', 'OCT': '10',
        'Novembre': '11',
        'Novembe': '11', 'NOV': '11', 'Décembre': '12', 'December': '12', 'DECEMBRE': '12', 'DEC': '12'
    }
    for month, num in month_to_num.items():
        if month in date_str:
            return date_str.replace(month, num)
    return date_str  # Retourne la chaîne inchangée si aucun remplacement n'est effectué


# Fonction pour reformater la date
def reformat_date(date_str):
    date_str = re.sub(r'\s{2,}', ' ', date_str).strip()
    return date_str.strip().replace(' ', '-')


def imput_data(df):
    # remplissage
    # Define the mapping for 'CodeProduit' and their corresponding 'Pays' values
    product_country_map = {
        '1 trans': 'france',
        '1 trans invoiced': 'united kingdom',
        '1x': 'france',
        '1x nat': 'france',
        '2 rouge': 'USA',
        '2 trans': 'USA',
        '2 trans  invoiced': 'united kingdom',
        '2x': 'france',
        '3x': 'USA',
        'aviary': 'USA',
        'bag': 'france',
        'big red': 'USA',
        'bssd': 'USA',
        'dsmlt': 'USA',
        'lot': 'france',
        'moulinet': 'france',
        'new red': 'USA',
        'oliv': 'france',
        'quail': 'USA',
        'rge': 'USA',
        'rouge': 'france',
        'slid': 'USA',
        'smlt': 'USA',
        'smlt invoiced': 'united kingdom'
    }
    # Update 'Pays' based on the mapping for non-null 'Pays'
    for product, country in product_country_map.items():
        df.loc[(df['CodeProduit'] == product) & (df['Pays'].isnull()), 'Pays'] = country
    # Drop the rows where 'CodeProduit' is '2 bags', '2dsmlt', or '3x invoiced',
    # and 'Pays' is null in a single step for efficiency
    df = df[~((df['CodeProduit'] == '2 bags') & (df['Pays'].isnull()))]
    df = df[~((df['CodeProduit'] == '2dsmlt') & (df['Pays'].isnull()))]
    df = df[~((df['CodeProduit'] == '3x   invoiced') & (df['Pays'].isnull()))]
    df = df[~((df['CodeProduit'] == 'goatskin invoiced') & (df['Pays'].isnull()))]
    df = df[~((df['CodeProduit'] == '2smlt') & (df['Pays'].isnull()))]
    return df


def clean_data(df):
    # Application de la fonction sur la colonne "Date"
    df['Pays'] = df['Pays'].apply(replace_country)
    # Application de la fonction sur la colonne "Date"
    df['Date'] = df['Date'].apply(replace_month)

    # Application de la fonction sur la colonne "Date"
    df['Date'] = df['Date'].apply(reformat_date)
    df['Pays'] = df['Pays'].replace('états-unis d\'amérique', 'USA')
    df = imput_data(df)
    df['sales_number'] = df.groupby(['CodeProduit', 'Pays', 'Date'])['CodeProduit'].transform('count').astype(int)
    df = df.drop_duplicates()
    # Group by 'CodeProduit' and calculate total sales
    total_sales = df.groupby('CodeProduit')['sales_number'].sum()

    # Filter for products with total sales >= 30
    valid_products = total_sales[total_sales >= 30].index
    # Keep only rows with valid CodeProduit
    df = df[df['CodeProduit'].isin(valid_products)]

    df = df.reset_index(drop=True)
    df.to_excel('Cleaned_Sales_Data.xlsx', index=False)
    print('Cleaning Sales Data is Done !')
    return df
