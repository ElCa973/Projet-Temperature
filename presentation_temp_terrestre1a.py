import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import mean_squared_error, r2_score
#from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
#import plotly.express as px
#from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import warnings
from sklearn.exceptions import ConvergenceWarning
#import datetime
 
with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning) 

world_df=pd.read_csv("world_df.csv", parse_dates=['year'])
world_df['year'] = world_df['year'].dt.year.astype(str)
df=pd.read_csv('owid-co2-data.csv', parse_dates=['year'])
df_nasa=pd.read_csv('ZonAnn_Ts_dSST.csv',  parse_dates=['Year'])

# Sommaire
st.sidebar.title("Sommaire")
pages = ["Page de garde", "Contexte du projet", "Exploration des données", "Analyse des données", "Modélisation", "Séries Temporelles", "Conclusion"]
page = st.sidebar.radio("Sélectionnez une page : ", pages)

# Pages
if page == pages [0]:
    # Contenu de la première page
    st.markdown("<h1 style='color: white;'>Page de garde</h1>", unsafe_allow_html=True)
    #st.title("Page de garde")
    st.image("Page de garde présentation.png")

elif page == pages [1]:
    # Contenu de la première page
    st.title("TEMPERATURE TERRESTRE")

    st.write("Ce projet a été réalisé dans le cadre du cursus de Data Analyst chez Datascientest.")
    st.write("L'objectif principal de ce projet est de constater le réchauffement (et le dérèglement) climatique à l'échelle de la planète sur les derniers siècles et dernières décennies. Nous avons analysé l'évolution des températures au niveau mondial et régional liée aux émissions des gaz à effet de serre. Ici nous allons présenter les résultats de notre étude. Les données que nous avons étudiées proviennt des deux bases de données suivantes : ")
    st.write("1. L'augmentation de la température au niveau mondial est-elle homogène partout sur la planète ?")
    st.write("2. Quelle est l’évolution générale des émissions de gaz à effet de serre dans différents pays et régions au fil du temps (1850-2021) ?")
    st.write("3. Quelle est la contribution des changements d'utilisation des terres aux émissions de CO2 à l'échelle mondiale ?")
    st.write("4. Comment la part des émissions cumulatives de CO2 diffère-t-elle d'un pays à l'autre ?")
    st.write("5. Quelle est la corrélation entre les émissions de gaz à effet de serre et les changements de température globale ?")
    st.write(" - Une base de données des émissions GES et de changement de température GitHub - https://github.com/owid/co2-data")
    st.write("- Une base de données de l'analyse de la température terrestre de la NASA - https://data.giss.nasa.gov/gistemp/")
    st.image('climate-change-1908381_1280.png')

elif page == pages [2]:
    # Contenu de la deuxième page
    st.title("Exploration et préparation des données")

    st.write("Voici les 5 premières lignes de la base de données du Github")
    
    

    df=pd.read_csv('owid-co2-data.csv', parse_dates=['year'])
    df['year'] = df['year'].dt.year.astype(str)
    st.dataframe(df.head())
    st.write("La première base de données nécessitait beaucoup de travail de préparation et de nettoyage des données")
    option = st.selectbox(
       "Afficher les paramètres de la table",
       ("Dimension initiale", "Nb de Nan", "Nb de doublons"),
       index=None,
       placeholder="Selectionner paramètre...",
      )
    
    if option == "Dimension initiale":
    # Affichez les dimensions initiales de la table
        st.write("Dimension initiale :", df.shape)
    elif option == "Nb de Nan":
        st.write("Nb de NaN :", df.isnull().sum().sum())
    elif option == "Nb de doublons":
        st.write("Nb de doublons :", df.duplicated().sum())
    
        
     

    st.write("La base de données de la NASA est beaucoup plus petite, ne présente pas de Nan, ni de doublons.")
    df_nasa=pd.read_csv('ZonAnn_Ts_dSST.csv',  parse_dates=['Year'])
    df_nasa['Year'] = df_nasa['Year'].dt.year.astype(str)
    df_nasa=df_nasa.rename(columns={"Year" : "year"})
    st.dataframe(df_nasa.head())

    
    option = st.selectbox(
       "Choisir les paramètres à afficher",
       ("Dimension initiale", "Nb de Nan", "Nb de doublons"),
       index=None,
       placeholder="Selectionner paramètre...",
      )
    nb_doublons = df_nasa.duplicated().sum()
    if option == "Dimension initiale":
    # Affichez les dimensions initiales de la table
        st.write("Dimension initiale :", df_nasa.shape)
    elif option == "Nb de Nan":
        st.write("Nb de NaN :", df_nasa.isnull().sum().sum())
    elif option == "Nb de doublons":
        st.write("Nb de doublons :", df.duplicated().sum())
    
    st.write("Après avoir étudié le dataframe du GitHub, nous avons décidé de supprimer les colonnes ayant plus de '30%' des valeurs manquantes, ainsi que les lignes avant l'année 1850.")
    

    df['year'] = df['year'].astype(int)

     # Filtrer le DataFrame
    df_1850 = df[df['year'] >= 1850]
    msno.bar(df_1850)

    # Utilisez Streamlit pour afficher le graphique
    st.pyplot(plt)
    
    seuil = 0.7 * len(df_1850)

    # Supprimer les colonnes ayant plus de 30% de NaN
    df_1850 = df_1850.dropna(axis=1, thresh=seuil)
    df_1850 = df_1850.drop('iso_code', axis=1)



    st.write("Nous avons gardé 13 variables dont le nombre de valeurs valides est entre 70%' et 100%, en excluant la variable 'iso_code' que nous avons jugée non pertinente.")

    st.write("Dimension après la réduction des données :", df_1850.shape)
    
    st.write("Pour la gestion des valeurs extrêmes, nous avons procédé à les identifier à l'aide des boîtes à moustaches pour chaque variable, ainsi que la méthode d'identification statistique, puis nous les avons remplacées par les limites inférieures et supérieures de l'IQR. Puis les valeurs manquantes ont été imputées à l'aide de la stratégie 'médiane'")

      
    
    # Sélectionnez les variables à imputer
    variables = ['population', 'cumulative_luc_co2', 'land_use_change_co2', 'land_use_change_co2_per_capita',
             'share_global_cumulative_luc_co2', 'share_global_luc_co2',
             'share_of_temperature_change_from_ghg', 'temperature_change_from_ch4',
             'temperature_change_from_co2', 'temperature_change_from_ghg',
             'temperature_change_from_n2o']


    # Chargez vos données dans df_1850

    variables = ['population', 'cumulative_luc_co2', 'land_use_change_co2', 'land_use_change_co2_per_capita',
             'share_global_cumulative_luc_co2', 'share_global_luc_co2',
             'share_of_temperature_change_from_ghg', 'temperature_change_from_ch4',
             'temperature_change_from_co2', 'temperature_change_from_ghg',
             'temperature_change_from_n2o']

   
    # Créez un objet SimpleImputer
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    # Appliquez l'imputation aux colonnes sélectionnées
    df_1850[variables] = simple_imputer.fit_transform(df_1850[variables])
    
    # Affichez le DataFrame mis à jour
    show_result = st.checkbox("Afficher le nombre de valeurs manquantes")

    # Affichez le DataFrame mis à jour
    if show_result:
        st.dataframe(df_1850.isnull().sum())
    st.write("Nous n'avons gardé que 2 variables dans la base de données de la Nasa : 'year' et 'Glob' - la variable cible ")
    st.write("Après avoir préparé les deux bases de données, nous les avons fusionnées pour avoir une base de données qui contient les valeurs explicatives et la valeur cible.")
    col = world_df.pop('Glob')
    world_df.insert(1, 'Glob', col)
    
    show_result = st.checkbox("Afficher le dataframe fusionné")

    # Affichez le DataFrame mis à jour
    if show_result:
        st.dataframe(world_df)


    
    
elif page == pages [3]:
    # Contenu de la troisième page
    st.title("Analyse des données")
    st.write("Dans cette partie de notre présentation nous montrerons les analyses statistiques effectué sur nos variables explicatives et sur la variable cible")
    
    
    df=pd.read_csv('owid-co2-data.csv')
    #world_df = df[df['country'] == 'World']
    df_nasa=pd.read_csv('ZonAnn_Ts_dSST.csv', index_col='Year')
        
   

    fig1 =sns.displot(x="Glob", data=df_nasa,kde=True)
    plt.title("Distribution de la variable cible: Température mondiale (°C)")
    st.pyplot(fig1)
    st.write("En observant sa densité du noyau, nous constatons visuellement qu'elle ne suit pas une distribution normale.")
    
    variables = [ 'population',
        'cumulative_luc_co2',
        'land_use_change_co2',
        'land_use_change_co2_per_capita', 'temperature_change_from_ghg',
         'temperature_change_from_n2o']
    fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    for i, variable in enumerate(variables):
       sns.histplot(world_df[variable], kde=True, ax=axes[i])
       axes[i].set_title(f' {variable}')
       axes[i].set_xlabel('')
    plt.tight_layout()
    plt.suptitle("Distributions des variables explicatives avec leurs densités de fonction", y=1.02, fontsize=18)
    st.pyplot(fig2)
    
    st.write("Nous observons que la normalité n'est pas vérifiée pour nos variables explicatives non plus. Par la suit le test de SHAPIRO a confirmé que la plupart des variables ne suivent pas la distribution normale, sauf 3 variables explicatives.") 
    st.write("\n")
    st.write("le TEST de SPEARMAN peut être appliqué pour étudier la nature des corrélations entres les variables, car pour ce type de test les données n'ont pas besoin d'être normalement distribuées.")   
   
    features_stat_descp_no_share =['population', 'cumulative_luc_co2', 'land_use_change_co2',
       'land_use_change_co2_per_capita', 'temperature_change_from_ch4', 'temperature_change_from_co2',
       'temperature_change_from_ghg', 'temperature_change_from_n2o']


    # Créez votre matrice de corrélation
    correlation_matrix = world_df[features_stat_descp_no_share].corr()

    # Créez la heatmap
    fig3, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.xticks(rotation=90)
    plt.title("Corrélations entre les Variables explicatives sauf les 'share_'", fontsize=10)
    plt.tight_layout()

    # Affichez la figure dans Streamlit
    st.pyplot(fig3)
    
    


elif page == pages [4]:
    # Contenu de la quatrième page
    st.title("Modélisation")
    st.write("Modélisation avec données réduites et optimisées.")

    world_dfREG=pd.read_csv("world_dfREG.csv")
    X = world_dfREG[['land_use_change_co2_per_capita','population','temperature_change_from_ch4', 'temperature_change_from_ghg']]
    y =  world_dfREG['Glob']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sc = StandardScaler()
    numer = ['land_use_change_co2_per_capita','population','temperature_change_from_ch4', 'temperature_change_from_ghg']

    X_train[numer] = sc.fit_transform(X_train[numer])
    X_test[numer] = sc.transform(X_test[numer])

    linear_reg_reduit_grid = LinearRegression(fit_intercept= True, n_jobs = 1, positive = True)
    linear_reg_reduit_grid.fit(X_train[numer], y_train)

    y_pred_linear_reduit_grid = linear_reg_reduit_grid.predict(X_test[numer])
    rmse_linear_reduit_grid = mean_squared_error(y_test, y_pred_linear_reduit_grid, squared=False)
    r2_linear_reduit_grid = r2_score(y_test, y_pred_linear_reduit_grid)

    decision_tree_reg_reduit_grid = DecisionTreeRegressor(random_state=42, 
                                                 criterion ='friedman_mse', 
                                                 max_depth=10, 
                                                 min_samples_leaf= 2, 
                                                 min_samples_split=2, 
                                                 splitter ='random')
    decision_tree_reg_reduit_grid.fit(X_train[numer], y_train)

    y_pred_tree_reduit_grid = decision_tree_reg_reduit_grid.predict(X_test[numer])
    rmse_tree_reduit_grid = mean_squared_error(y_test, y_pred_tree_reduit_grid, squared=False)
    r2_tree_reduit_grid = r2_score(y_test, y_pred_tree_reduit_grid)

    random_forest_reg_reduit_grid = RandomForestRegressor(random_state=42, max_depth= 7, n_estimators= 200)
    random_forest_reg_reduit_grid.fit(X_train[numer], y_train)

    y_pred_forest_reduit_grid = random_forest_reg_reduit_grid.predict(X_test[numer])
    rmse_forest_reduit_grid = mean_squared_error(y_test, y_pred_forest_reduit_grid, squared=False)
    r2_forest_reduit_grid = r2_score(y_test, y_pred_forest_reduit_grid)

    
   

    model_choisi = st.selectbox(label = "Modèle", options = ['Regression Linéaire', 'Random Forest', 'Decision Tree'])

    def train_model(model_choisi) :
        if model_choisi =='Regression Linéaire' :
            y_pred = y_pred_linear_reduit_grid
            
        elif model_choisi == 'Random Forest' :
            y_pred=y_pred_forest_reduit_grid
            
        elif model_choisi == 'Decision Tree' :
            y_pred = y_pred_tree_reduit_grid
            
        r2 = r2_score(y_test, y_pred)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return r2, rmse
    r2, rmse = train_model(model_choisi)


    
    st.write("Coefficient de détermination:", r2)  
    st.write("RMSE (Root Mean Squared Error) :", rmse) 


    
elif page == pages [5]:
    # Contenu de la quatrième page
    st.title("Séries Temporelles")
    st.write("Dans le cadre de notre projet nous avons été amenés à étudier un module complémentaire qui n'était pas inclus dans notre programme.")
    st.write("En travaillant avec des séries temporelles, il est recommandé d'avoir la date comme index pour faciliter les opérations temporelles et l'analyse.")
    world_dfTIMESERIES = pd.read_csv("ZonAnn_Ts_dSST.csv", parse_dates=[0], index_col= 0, header=0).squeeze()
    world_df_TS = world_dfTIMESERIES['Glob']
     
    show_result = st.checkbox("Afficher le DataFrame pour une analyse en séries temporelles")

    # Affichez le DataFrame mis à jour
    if show_result:
        st.dataframe(world_df_TS)
    
    plt.plot(world_df_TS, label='Température', c = 'red')
    plt.xlabel('Années')
    plt.ylabel('Température °C')
    plt.title('Variation des températures dans le monde') 
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write("Pour analyser notre série temporelle nous avons commencé par la décomposition saisonnière qui permet de décomposer une série en plusieurs composantes : la tendance (trend), la saisonnalité (seasonal), et les résidus (residuals).")
    seasonal = seasonal_decompose(world_df_TS)
    seasonal.plot()
    st.pyplot(plt)

    st.write("Après avor analysé les composantes obtenues, il a fallu vérifier la stationnarité de notre série, car l'absence de saisonnalité ne signifie pas que la série est stationnaire.")
    st.write("La vérification de la stationnarité peut être effectuée de deux manières différentes : visualisation graphique et tests statistiques(le test Dickey-Fuller augmenté (ADF) ou le test de Phillips-Perron)")


    st.write("Nous avons effectué le test ADF et obtenu le résultat suivant : ")
    time_series = world_df_TS


    # Effectuez le test de Dickey-Fuller Augmenté
    result = adfuller(time_series)

    # Examinez la statistique du test et la p-valeur
    adf_statistic = result[0]
    p_value = result[1]

    if p_value <= 0.05:
        st.write("La série est stationnaire (p-valeur =", p_value, ")")
    else:
        st.write("La série n'est pas stationnaire (p-valeur =", p_value, ")")

    st.write("La série n'est pas stationnaire, donc il faut la stationnariser à l'aide de la différenciation")
    st.write("Le graphique ci-dessous nous montre que la stationnarisation a bien fonctionné")
    world_df_TS_diff = world_df_TS.diff().dropna() #différenciation
    fig, ax = plt.subplots()
    ax.plot(world_df_TS_diff)
    st.pyplot(fig)

    st.write("Maintenant on peut procéder à la modélisation, en séparant notre jeu de données en X_train et X_test au préalable.")
    split_index = round(len(world_df_TS_diff) * 0.8)
    X_train = world_df_TS_diff.iloc[:split_index]
    X_test = world_df_TS_diff.iloc[split_index:]
    
    

    model=sm.tsa.SARIMAX(world_df_TS_diff,order=(4,2,0))
    sarima=model.fit()
    st.write(sarima.summary())

    st.write("Voici les métriques de performance pour évaluer notre modèle.")
    m_sarimax_foca420 = sarima.get_forecast(steps=28).predicted_mean
    st.write("\n====================== METRICS ==============================")
    st.write('MSE:', mean_squared_error(X_test, m_sarimax_foca420))
    st.write('RMSE:', np.sqrt(mean_squared_error(X_test, m_sarimax_foca420)))
    st.write('MAE:', mean_absolute_error(X_test, m_sarimax_foca420))
    st.write("\n============================================================")

    st.write("Nous pouvons maintenant procéder aux prédictions.")

    


    option = st.selectbox(
       "Choisissez la durée des prédictions",
       ("Prédictions jusqu'en 2030", "Prédictions jusqu'en 2042"),
       index=None,
       placeholder="Selectionner paramètre...",
      )
    
    if option == "Prédictions jusqu'en 2030":
        pred = np.exp(sarima.predict(143, 151))# 20 ans (de 2022 à 2042), d'où les 143 jusqu'à 163                                    

        world_df_TS_pred = pd.concat([world_df_TS, pred])#Concaténation des prédictions
        st.line_chart(world_df_TS_pred, use_container_width=True) #Visualisation

           
        


    elif option == "Prédictions jusqu'en 2042":
        pred = np.exp(sarima.predict(143, 163))
            
        world_df_TS_pred = pd.concat([world_df_TS, pred])
        st.line_chart(world_df_TS_pred)

elif page == pages[6]:
    # Contenu de la cinquième page
    st.title("CONCLUSION")
    st.write("•	En conclusion, notre analyse confirme le consensus scientifique écrasant selon lequel le changement climatique est réel, largement provoqué par les activités humaines et qu'il pose des défis importants à notre planète.") 

    st.write("•	Les données et les analyses présentées dans ce rapport soulignent l'ampleur du réchauffement climatique.")

    st.write("•	D'après nos prédictions futures, il est essentiel de reconnaître que la hausse de la température mondiale va se poursuivre.")

    st.write("•	Si rien n’est fait, la température moyenne augmentera de 1,04°C dans les prochaines années.")

    st.write("•	Les conséquences du changement climatique sont étendues, impactant les écosystèmes, l'agriculture, mais aussi notre vie quotidienne.")

    st.write("•	Les régions vulnérables, comme l'Afrique, sont en danger, même si elles contribuent moins aux émissions de gaz à effet de serre.")
    st.image("climate-change-new.png")

    
