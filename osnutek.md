# Napovedovanje cen stanovanj v Amesu s pomočjo naprednih regresijskih tehnik  
*Spletno tekmovanje na spletni strani Kaggle*

## Opis problema  
Cene nepremičnin so odvisne od kompleksne kombinacije dejavnikov, ki presegajo osnovne parametre, kot so število sob ali lokacija. V tem projektu analiziramo nabor podatkov o 2,900 stanovanjih v Amesu (Iowa, ZDA), ki vsebuje 79 opisnih spremenljivk - od dimenzij kleti do bližine železniških prog. Glavni izziv je identifikacija najpomembnejših dejavnikov vpliva na ceno in razvoj natančnega napovednega modela.

**Ključna vprašanja**:  
1. Kateri arhitekturni in lokacijski dejavniki najbolj vplivajo na ceno stanovanj?  
2. Kako učinkovito obdelati manjkajoče podatke in kategorikalne spremenljivke?  
3. Kateri modeli (XGBoost, naključni gozdovi, nevronske mreže) dosežejo najnižjo RMSE napako?  
4. Kako izboljšati natančnost z optimizacijo hiperparametrov in ansambelskimi metodami?  

## Vir in oblika podatkov  
Podatki so dostopni prek tekmovanja [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course/data) na spletni strani Kaggle. 

**Struktura podatkov**:  
- `train.csv` (1.46 MB): Učni niz z 1,460 primeri in 81 stolpci (vključno s ciljno spremenljivko SalePrice)  
- `test.csv` (1.42 MB): Testni niz z 1,459 primeri za validacijo modelov  
- `data_description.txt`: Detajlne definicije vseh spremenljivk  
- Primeri podatkov: kombinacija numeričnih (npr. 'LotArea', 'YearBuilt') in kategorikalnih ('Neighborhood', 'RoofStyle') spremenljivk  

**Značilnosti podatkovnega niza**:  
- Heterogenost: 23 numerične in 43 kategorikalne spremenljivke  
- Manjkajoče vrednosti (npr. v stolpcih 'PoolQC', 'Alley')  
- Hierarhične kategorije (npr. 'OverallQual' z ocenami 1-10)  

## Načrt analize  
1. Eksplorativna analiza (EDA) z vizualizacijo korelacij in distribucij  
2. Predprocesiranje:  
   - Imputacija manjkajočih vrednosti  
   - Kodiranje kategorikalnih spremenljivk (Target Encoding, One-Hot)  
   - Normalizacija numeričnih spremenljivk  
3. Feature engineering: kombinacija novih spremenljivk (npr. skupna površina)  
4. Implementacija več modelov in primerjava njihove uspešnosti  
5. Optimizacija hiperparametrov z GridSearch/RandomSearch  
6. Interpretacija modelov s SHAP vrednotami  

Ciljna metrika: **RMSE na logaritmirani cenovni skali** (po kaggleovem kriteriju)

Člani: Adam Zeggai, Matej Pavli, Matic Rape, Miha Kastelic, Rok Rihar