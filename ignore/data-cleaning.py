import pandas as pd 
from sklearn.impute import SimpleImputer

# Modify mental health data set for merging 
def normalize_mh_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["Code", "index"], inplace=True)
    df.dropna(inplace=True)
    df_copy = df.copy() 
    df_copy = df_copy.rename(columns={"Entity":"country"})
    df_copy["Year"] = df_copy["Year"].astype(int)
    df_copy[['Schizophrenia (%)', 'Bipolar disorder (%)', 'Anxiety disorders (%)', "Eating disorders (%)"]] = df_copy[['Schizophrenia (%)', 'Bipolar disorder (%)', 'Anxiety disorders (%)','Eating disorders (%)']].astype(float)
    df_copy.columns = df_copy.columns.str.replace(" (%)", "").str.lower().str.replace(" ", "_")
    # Download the data 
    df_copy.to_csv("mh_data.csv", index=False)
    return df_copy 

# Modify world happiness index for merging 
def normalize_wh_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns= ["Positive affect", "Negative affect"], inplace=True)
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={"Country name":"country", "Life Ladder":"happiness"})
    df_copy = df_copy.rename(columns={"Log GDP per capita":"gdp_per_capita", 
                                                    "Social support":"social_support", 
                                                    "Healthy life expectancy at birth":"life_expectancy",
                                                    "Freedom to make life choices":"freedom", 
                                                    "Generosity":"generosity", 
                                                    "Perceptions of corruption":"corruption"})
    df_copy.columns= df_copy.columns.str.lower().str.replace(" ", "_")
    df_copy.dropna(inplace=True)
    df_copy.to_csv("wh_data.csv", index=False)
    return df_copy 

# Rename countries for merging and cultural sensitivity
def rename_countries(mental_health: pd.DataFrame, world_happiness:pd.DataFrame):
    mental_health.replace({"Macedonia":"North Macedonia"},inplace=True)
    world_happiness.replace({ "Palestinian Territories": "Palestine",
                    "Congo (Kinshasa)": "Democratic Republic of the Congo",
                    "Taiwan Province of China": "Taiwan",
                    "Ivory Coast": "Cote d'Ivoire",
                    "Congo (Brazzaville)": "Congo"},
                    inplace=True)
    return (mental_health, world_happiness)

def merge_data(mental_health: pd.DataFrame, world_happiness: pd.DataFrame) -> pd.DataFrame:
    mental_health, world_happiness = rename_countries(mental_health, world_happiness)
    mental_health = mental_health[mental_health.country.isin(world_happiness.country)]
    data_merged = mental_health.merge(world_happiness, on=["country", "year"], how="right")
    data_merged.to_csv("merged_data.csv", index=False)
    return data_merged 

def get_quantile_score(data: pd.DataFrame, disorder: str, disorder_rate: float) -> int: 
    """This function returns a score from 1-4, based on the quantile that the 
    disorder rate falls into. These scores will later be summed to create an 
    overall mental health score, which will then be used to create categorical
    labels for the mental health outcomes.
    
    (This prompt was written with the help of Copilot.)

    Keyword arguments:
    data -- the data frame that contains the mental health data
    disorder -- the name of the disorder that we want to obtain the quantile score for
    disorder_rate -- the rate of the disorder in a specific row (country-year pair)
    """
    if (disorder_rate < data[disorder].quantile(0.25)):
        return 1 
    elif (disorder_rate < data[disorder].median()):
        return 2
    elif (disorder_rate < data[disorder].quantile(0.75)):
        return 3 
    else:
        return 4
    
def get_prevalence_category(data_imputed: pd.DataFrame, score: int) -> str:
    """This function returns a categorical label based on the total mental health
    score of a country-year pair. (This prompt was written with the help of Copilot.)

    Keyword arguments:
    score -- the total mental health score of a country-year pair
    """
    total_col = data_imputed["total_score"]
    if (score < total_col.quantile(0.25)):
        return "Low Prevalence"
    elif (score < total_col.median()):
        return "Below Average Prevalence"
    elif (score < total_col.quantile(0.75)):
        return "Above Average Prevalence"
    else:
        return "High Prevalence"

def impute_data(data_merged):
    # impute NaN values in disorder columns 
    disorders_subset = data_merged[["alcohol_use_disorders", "schizophrenia", 
                                "bipolar_disorder", "anxiety_disorders", 
                                "eating_disorders", "depression", 
                                "drug_use_disorders"]]
    disorders_labels = disorders_subset.columns
    # Subset remaining data that we don't want to impute 
    data_other = data_merged.drop(columns=disorders_labels)
    num_imputer = SimpleImputer(strategy="median")
    imputed = num_imputer.fit_transform(disorders_subset)
    data_num_imputed = pd.DataFrame(imputed, columns=disorders_labels)
    # Concatenate the two subsetted data frames back together. 
    data_imputed = pd.concat([data_other, data_num_imputed], axis=1)
    # Compute mental health scores 
    score_columns = [] # Save column names for later. 
    for disorder in disorders_labels: 
        data_imputed[disorder + "_score"] = data_imputed[disorder].\
            apply(lambda x: get_quantile_score(data_imputed, disorder, x))
        score_columns.append(disorder + "_score")
    data_imputed["total_score"] = data_imputed[score_columns].sum(axis=1)
    data_imputed["mental_health_prevalence"] = data_imputed["total_score"].\
        apply(lambda x: get_prevalence_category(data_imputed, x))
    data_imputed = data_imputed.drop(columns = score_columns)
    data_imputed.drop(columns=["total_score","country","year","alcohol_use_disorders","schizophrenia","bipolar_disorder","anxiety_disorders","eating_disorders","depression","drug_use_disorders"],inplace=True)
    return data_imputed 

mental_health_data = normalize_mh_data(pd.read_csv("ignore/original_data/worldwide_mh_data.csv", low_memory=False))
world_happiness_data = normalize_wh_data(pd.read_csv("ignore/original_data/world-happiness-report.csv"))
merged_data = merge_data(mental_health_data, world_happiness_data)
imputed_data = impute_data(merged_data)
imputed_data.to_csv("imputed_data.csv", index=False)