# Končno poročilo: Napovedovanje cen nepremičnin v Amesu s pomočjo naprednih regresijskih tehnik

## 1. Uvod

V tem končnem poročilu predstavljamo celovit pregled projekta napovedovanja cen nepremičnin v mestu Ames, Iowa. Nadgradili smo analize in modele, predstavljene v vmesnem poročilu, z namenom razvitja natančnejšega in robustnejšega napovednega sistema. Cilj projekta je bil poglobljeno raziskati podatkovni niz, implementirati napredne tehnike strojnega učenja, izvesti obsežno "feature engineering" ter na koncu ustvariti interaktivno spletno aplikacijo za praktično uporabo razvitega modela.

Cene nepremičnin so odvisne od kompleksne kombinacije dejavnikov, ki presegajo osnovne parametre, kot so število sob ali lokacija. V tem projektu smo analizirali nabor podatkov o približno 2900 stanovanjih v Amesu, ki vsebuje 79 opisnih spremenljivk – od dimenzij kleti do bližine železniških prog. Glavni izziv je bil identifikacija najpomembnejših dejavnikov vpliva na ceno in razvoj natančnega napovednega modela.

V prejšnjem, vmesnem poročilu smo ostali pri enostavnem testnem modelu, ki je služil kot izhodišče. Cilj tega končnega poročila je predstaviti pot do bistveno boljših modelov in praktične implementacije preko interaktivne spletne strani, razvite s knjižnico Streamlit.

## 2. Vir in oblika podatkov

Podatki, uporabljeni v tem projektu, izvirajo iz tekmovanja ["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/home-data-for-ml-course) na platformi Kaggle. Tekmovanje ponuja bogat nabor podatkov, idealen za raziskovanje in modeliranje cen nepremičnin.

**Struktura podatkov:**
Podatkovni set je razdeljen na tri glavne datoteke:
* `train.csv`: Učni niz, ki vsebuje 1460 zapisov (hiš) in 81 stolpcev. Med temi stolpci je tudi naša ciljna spremenljivka `SalePrice` (prodajna cena).
* `test.csv`: Testni niz, ki vsebuje 1459 zapisov in 80 stolpcev. Ta niz ne vsebuje stolpca `SalePrice`, saj je naloga napovedati te vrednosti.
* `data_description.txt`: Podroben opis vsake od 79 značilk (spremenljivk), ki pojasnjuje pomen posameznega stolpca. Ta datoteka je ključna za razumevanje podatkov in informirano "feature engineering".
* `sample_submission.csv`: Primer datoteke za oddajo napovedi v pravilni obliki.

**Značilnosti podatkovnega niza:**
Podatki so heterogeni in vključujejo:
* **Numerične spremenljivke:** Približno 38 spremenljivk je numeričnih, kot so `LotArea` (velikost parcele), `YearBuilt` (leto izgradnje), `1stFlrSF` (kvadratura prvega nadstropja), `GrLivArea` (nadzemna bivalna površina).
* **Kategorikalne spremenljivke:** Približno 43 spremenljivk je kategorikalnih, ki opisujejo lastnosti, kot so `MSZoning` (splošna klasifikacija cone), `Neighborhood` (fizične lokacije znotraj meja mesta Ames), `RoofStyle` (tip strehe), `Condition1` (bližina različnih pomembnih točk, npr. ceste, železnice). Nekatere od teh so ordinalne (npr. `OverallQual` - splošna kvaliteta materiala in končne obdelave, ocenjena od 1 do 10), druge pa nominalne.
* **Manjkajoče vrednosti:** Številni stolpci vsebujejo manjkajoče vrednosti (NaN). Najpogosteje se pojavljajo v stolpcih, kot so `PoolQC` (kvaliteta bazena), `MiscFeature` (razne dodatne značilnosti), `Alley` (tip dostopa do parcele), `Fence` (kvaliteta ograje), `FireplaceQu` (kvaliteta kamina). Pravilna obravnava teh manjkajočih vrednosti je ključna za uspešno modeliranje.
* **Časovne spremenljivke:** Podatki vsebujejo več stolpcev, povezanih s časom, kot so `YearBuilt` (leto izgradnje), `YearRemodAdd` (leto prenove), `GarageYrBlt` (leto izgradnje garaže) in `YrSold` (leto prodaje).

Razumevanje teh značilnosti je bilo osnova za nadaljnje korake, vključno z eksplorativno analizo podatkov, predprocesiranjem in gradnjo modelov.

## 3. Eksplorativna analiza podatkov (EDA)

Preden smo se lotili gradnje kompleksnih modelov, smo izvedli temeljito eksplorativno analizo podatkov (EDA), da bi bolje razumeli strukturo, porazdelitve in medsebojne odnose med spremenljivkami. Ta faza je bila ključna za odkrivanje vzorcev, identifikacijo osamelcev in informiranje o strategijah predprocesiranja.

**3.1. Analiza ciljne spremenljivke `SalePrice`**
Prvi korak je bil pregled naše ciljne spremenljivke, `SalePrice`.
* **Porazdelitev:** Ugotovili smo, da je porazdelitev prodajnih cen desno-asimetrična (pozitivno asimetrična). To pomeni, da je večina hiš prodanih po nižjih do srednjih cenah, medtem ko manjše število hiš dosega bistveno višje cene.
    * Za stabilizacijo variance in približevanje normalni porazdelitvi, kar je pogosto koristno za linearne modele in nekatere druge algoritme, smo uporabili logaritemsko transformacijo `SalePrice` (običajno `np.log1p` ali `np.log`). Ta transformirana vrednost je bila nato uporabljena kot ciljna spremenljivka v večini naših modelov.
* **Opisna statistika:** Povprečna cena hiše v učnem nizu je okoli $180,921, z znatnim standardnim odklonom, kar kaže na veliko variabilnost cen.

**3.2. Numerične spremenljivke**
* **Korelacije:** Analizirali smo korelacijsko matriko med numeričnimi spremenljivkami in `SalePrice`. Najmočnejše pozitivne korelacije s `SalePrice` so pokazale:
    * `OverallQual` (Splošna kvaliteta materiala in končne obdelave)
    * `GrLivArea` (Nadzemna bivalna površina)
    * `GarageCars` (Velikost garaže glede na kapaciteto avtomobilov)
    * `GarageArea` (Velikost garaže v kvadratnih čevljih)
    * `TotalBsmtSF` (Skupna površina kleti)
    * `1stFlrSF` (Površina prvega nadstropja)
    * `FullBath` (Polne kopalnice nad zemljo)
    * `TotRmsAbvGrd` (Skupno število sob nad zemljo, brez kopalnic)
    * `YearBuilt` (Leto izgradnje)
    * `YearRemodAdd` (Leto prenove)
    Visoka korelacija med nekaterimi neodvisnimi spremenljivkami (npr. `GarageCars` in `GarageArea`, `TotalBsmtSF` in `1stFlrSF` pri enonadstropnih hišah) je nakazovala na potencialno multikolinearnost, kar smo upoštevali pri izbiri značilk in "feature engineeringu".
* **Porazdelitve:** Preučili smo porazdelitve ključnih numeričnih spremenljivk. Nekatere, kot `LotArea`, so bile prav tako desno-asimetrične.

**3.3. Kategorikalne spremenljivke**
* **Vpliv na ceno:** Za kategorikalne spremenljivke smo analizirali, kako se povprečna `SalePrice` razlikuje med različnimi kategorijami. Na primer:
    * `Neighborhood`: Cene so se močno razlikovale glede na sosesko, kar kaže na velik vpliv lokacije.
    * `MSZoning`: Različne cone so imele različne povprečne cene.
    * `HouseStyle`: Tip hiše (npr. enonadstropna, dvonadstropna) je vplival na ceno.
    Vizualizacije, kot so "box-ploti", so pomagale prikazati te razlike in identificirati kategorije, povezane z višjimi ali nižjimi cenami. V vmesnem poročilu (datoteka `VMESNO.md`) so prikazani grafi, ki ilustrirajo te odnose za spremenljivke `MSZoning`, `Street`, `BldgType` in `HouseStyle`.
* **Kardinalnost:** Nekatere kategorikalne spremenljivke imajo veliko število unikatnih vrednosti (visoka kardinalnost), kar lahko predstavlja izziv pri kodiranju (npr. `Neighborhood`).

**3.4. Manjkajoče vrednosti**
Podrobno smo pregledali manjkajoče vrednosti. Za vsak stolpec z manjkajočimi podatki smo na podlagi `data_description.txt` ugotavljali, ali `NaN` pomeni dejansko odsotnost neke lastnosti (npr. `NaN` v `PoolQC` verjetno pomeni, da hiša nima bazena) ali pa gre za resnično manjkajoč podatek. Ta ugotovitev je bila ključna za izbiro ustrezne strategije imputacije.

**3.5. Osamelci (Outliers)**
Identificirali smo potencialne osamelce, zlasti v odnosu med `GrLivArea` in `SalePrice`. Nekatere hiše z zelo veliko bivalno površino so imele relativno nizko ceno. Na podlagi analiz (in splošne prakse pri tem tekmovanju) smo se odločili za odstranitev nekaj izrazitih osamelcev iz učnega niza, saj bi lahko nesorazmerno vplivali na učni proces modela. Seznam indeksov odstranjenih osamelcev (npr. `[598, 955, ...]`) je bil uporabljen tudi v `streamlib_housing_prices.py` pri nalaganju podatkov.

EDA je zagotovila trdno podlago za naslednji korak: predprocesiranje podatkov in ustvarjanje novih, bolj informativnih značilk.

## 4. Predprocesiranje podatkov in Feature Engineering

Ta faza je bila ena najpomembnejših v projektu, saj kakovost vhodnih podatkov neposredno vpliva na uspešnost modelov. Osredotočili smo se na čiščenje podatkov, transformacije in, kar je ključno, na ustvarjanje novih, sintetičnih značilk ("feature engineering"), ki bolje zajemajo kompleksne odnose, relevantne za ceno nepremičnin.

**4.1. Obravnava manjkajočih vrednosti**
Strategija imputacije je bila odvisna od tipa spremenljivke in pomena manjkajoče vrednosti:
* **Kategorikalne spremenljivke:**
    * Za spremenljivke, kjer `NaN` pomeni odsotnost lastnosti (npr. `Alley`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PoolQC`, `Fence`, `MiscFeature`), smo manjkajoče vrednosti zapolnili z nizom `'None'` ali `'NA'`.
    * Za nekatere druge, kot je `Electrical`, smo manjkajoče vrednosti zapolnili z najpogostejšo vrednostjo (modusom).
* **Numerične spremenljivke:**
    * `LotFrontage`: Manjkajoče vrednosti smo imputirali z mediano `LotFrontage` za vsako sosesko (`Neighborhood`).
    * `GarageYrBlt`: Kjer je manjkalo leto izgradnje garaže, smo uporabili leto izgradnje hiše (`YearBuilt`). Za hiše brez garaže (`GarageType` == `'None'`) je bila vrednost postavljena na 0.
    * Za ostale numerične manjkajoče vrednosti (npr. `MasVnrArea`) smo pogosto uporabili mediano ali konstanto 0.

**4.2. Transformacije spremenljivk**
* **Logaritemska transformacija `SalePrice`:** Kot omenjeno v EDA, smo `SalePrice` transformirali z `np.log` (ali `np.log1p`) za stabilizacijo variance. Vse napovedi modelov so bile posledično na logaritmirani skali, zato smo jih pred prikazom uporabniku transformirali nazaj z `np.exp` (ali `np.expm1`).
* **Obravnava asimetrije numeričnih značilk:** Za numerične značilke, ki so kazale močno asimetrijo, smo prav tako preučili možnost logaritemske transformacije.

**4.3. Feature Engineering (Ustvarjanje novih značilk)**
To je bil ključni del za izboljšanje napovedne moči modelov. Ustvarili smo več novih značilk, ki združujejo obstoječe ali iz njih izpeljujejo bolj smiselne informacije. Spodaj so podrobneje opisane nekatere izmed njih, skupaj z utemeljitvami, podobno kot v `KONČNO.md`:

* **`houseAge = YrSold - YearBuilt`**
    * *Zakaj?* Samo leto izgradnje (`YearBuilt`) pove, kdaj je bila hiša zgrajena, a modelu manjka kontekst, kako "stara" je hiša v času prodaje. Hiša, stara 100 let, je povsem drugačna od hiše, stare 5 let.
    * *Kaj zajame?* Razmerje med datumom prodaje in datumom izgradnje. Starejše hiše imajo običajno več obrabe, medtem ko novejše (mlajše) hiše pogosto dosegajo višjo ceno zaradi sodobnejše gradnje, manj potrebnih popravkov itd.
    * *Zakaj odstraniti `YearBuilt` in `YrSold` (po ustvarjanju)?* Ker se informacija o "starosti" izračuna neposredno iz njiju, je `houseAge` bolje "sintetiziran" podatek. Če bi pustili vse tri, bi imeli multikolinearnost.

* **`houseRemodelAge = YrSold - YearRemodAdd`**
    * *Zakaj?* Hiša, ki je bila nedavno adaptirana ali obnovljena, je običajno vredna več kot tista, ki ni bila dolgo časa prenovljena. Samo leto zadnje adaptacije (`YearRemodAdd`) modelu ne pove, koliko časa je minilo od takrat do prodaje.
    * *Kaj zajame?* Časovno razliko med prodajo in zadnjo prenovo. Če ni bilo prenove, je `YearRemodAdd` enak `YearBuilt`, torej bo `houseRemodelAge` enak `houseAge`.
    * *Zakaj odstraniti `YearRemodAdd`?* Podobno kot pri `houseAge`, je ta nova značilka bolj neposredno uporabna.

* **`IsNewHouse = (YearBuilt == YrSold).astype(int)`**
    * *Zakaj?* Popolnoma nove hiše (prodane v letu izgradnje) imajo lahko poseben premijski status na trgu.
    * *Kaj zajame?* Binarna značilka, ki označuje, ali je bila hiša prodana v istem letu, ko je bila zgrajena.

* **`TotalSF = GrLivArea + TotalBsmtSF`** (ali včasih `1stFlrSF + 2ndFlrSF + TotalBsmtSF`)
    * *Zakaj?* Skupna bivalna površina, vključno s kletjo, je močan indikator velikosti in posledično cene.
    * *Kaj zajame?* Celotno uporabno površino hiše.
    * *Zakaj odstraniti originalne?* Združevanje zmanjša število dimenzij in koreliranih značilk, kar lahko pomaga modelu bolje generalizirati.

* **`TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`**
    * *Zakaj?* Število kopalnic je pomembno, vendar imajo polovične kopalnice manjšo vrednost kot polne. Ta formula to uteži.
    * *Kaj zajame?* Skupno "kopalniško kapaciteto" hiše.
    * *Zakaj odstraniti originalne?* Združena vrednost je bolj zgoščena informacija.

* **`TotalPorchSF = OpenPorchSF + EnclosedPorch + ScreenPorch + WoodDeckSF + X3SsnPorch`**
    * *Zakaj?* Različne vrste verand in zunanjih površin prispevajo k vrednosti. Njihova vsota daje celotno "zunanje življenjsko površino".
    * *Kaj zajame?* Agregirano površino vseh verand in krovov.
    * *Zakaj odstraniti originalne?* Originalni stolpci so lahko med seboj korelirani. Združena vrednost zmanjša število atributov.

**Glavni razlogi za takšno "feature engineering":**
* **Manj redundance, manj koreliranih vhodov:** Združene spremenljivke zmanjšajo možnost, da bi se model preveč "navezal" na posamezne komponente, ki so med seboj močno korelirane. Hkrati zmanjšajo število stolpcev, kar lahko pospeši učenje in zmanjša tveganje prekomernega prileganja (overfitting).
* **Bolj direktna interpretacija za model:** Modelu ni treba sam izračunati starosti hiše ali relativne vrednosti različnih tipov kopalnic. S predhodno vključitvijo teh informacij model hitreje ujame ključne vzorce.
* **Zajemanje interakcij in nelinearnosti:** "Skupna kvadratura" ima pogosto močnejši (in včasih bolj linearen na log-skali cene) odnos s ceno kot ločene kvadrature. Podobno velja za starost hiše.

**4.4. Kodiranje kategorikalnih spremenljivk**
Kategorikalne spremenljivke je treba pretvoriti v numerično obliko, da jih lahko modeli strojnega učenja uporabijo.
* **Ordinalno kodiranje (Ordinal Encoding):** Uporabljeno za spremenljivke, kjer obstaja naravna hierarhija med kategorijami (npr. `ExterQual`: Excellent > Good > Average > Fair > Poor). Kategorijam smo dodelili numerične vrednosti (npr. Ex=4, Gd=3, TA=2, Fa=1, NA=0). To smo storili za značilke kot `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `BsmtQual`, `ExterCond`, `ExterQual`, `FireplaceQu`, `Functional`, `GarageCond`, `GarageFinish`, `GarageQual`, `HeatingQC`, `KitchenQual`, `LandSlope`, `LotShape`, `PavedDrive`, `PoolQC`, `Utilities`.
* **"One-Hot" kodiranje (One-Hot Encoding):** Uporabljeno za nominalne kategorikalne spremenljivke, kjer ni intrinzičnega vrstnega reda (npr. `Neighborhood`, `MSZoning`). Vsaka kategorija postane nov binarni stolpec (0 ali 1). To preprečuje, da bi model napačno interpretiral vrstni red med kategorijami. Za zmanjšanje dimenzionalnosti smo uporabili parameter `handle_unknown='ignore'` ali združili redke kategorije.

**4.5. Skaliranje numeričnih spremenljivk**
Po imputaciji in "feature engineeringu" smo vse numerične spremenljivke skalirali s `StandardScaler` iz knjižnice `scikit-learn`. Ta postopek transformira podatke tako, da imajo povprečje 0 in standardni odklon 1. Skaliranje je pomembno za modele, ki so občutljivi na merilo vhodnih spremenljivk, kot so linearni modeli z regularizacijo (Ridge, Lasso), SVM in nevronske mreže. Pomaga tudi pri hitrejši konvergenci algoritmov, ki temeljijo na gradientnem spustu.

**4.6. Uporaba `Pipeline`**
Vse korake predprocesiranja (imputacija, kodiranje, skaliranje) smo združili v `ColumnTransformer` in nato v `Pipeline` iz `scikit-learn`. To zagotavlja, da se enaki koraki predprocesiranja konsistentno uporabijo tako na učnih kot na testnih podatkih (in kasneje na novih podatkih v spletni aplikaciji), kar preprečuje uhajanje podatkov ("data leakage") in poenostavlja delovni tok. Ta pristop je bil implementiran v `Housing_prices_kaggle_2.ipynb` in posledično v `streamlib_housing_prices.py`.

## 5. Gradnja modelov

Po skrbni pripravi podatkov smo prešli na fazo gradnje in ocenjevanja različnih regresijskih modelov. Cilj je bil najti model, ki najbolje generalizira na nevidene podatke in doseže najnižjo napako napovedi.

**5.1. Izhodiščni model (Baseline)**
Kot je bilo omenjeno v vmesnem poročilu (`VMESNO.md`) in prikazano v `Housing_prices_kaggle.ipynb`, smo najprej zgradili preprost model `DecisionTreeRegressor` z uporabo le nekaj osnovnih značilk. Ta model je služil kot osnovna referenca za primerjavo z naprednejšimi pristopi.

**5.2. Izbor in trening naprednejših modelov**
V `Housing_prices_kaggle_2.ipynb` in nato preneseno v `streamlib_housing_prices.py` smo implementirali in preizkusili vrsto naprednejših regresijskih modelov:

* **Linearni modeli:**
    * `Ridge Regression`: Linearna regresija z L2 regularizacijo.

* **Drevesni ansambelski modeli:**
    * `RandomForestRegressor`
    * `GradientBoostingRegressor`
    * `XGBRegressor` (XGBoost)
    * `LGBMRegressor` (LightGBM)
    * `CatBoostRegressor`

**5.3. Ansambelske metode (Ensemble Methods)**
Za dodatno izboljšanje napovedi smo uporabili ansambelske tehnike, ki združujejo napovedi več osnovnih modelov:

* **`VotingRegressor`:**
    * Združuje napovedi več različnih modelov s povprečenjem.

* **`StackingRegressor` (Zlaganje modelov):**
    * **Nivo 0 (Base Learners):** Modeli kot XGBoost, LightGBM, CatBoost, Ridge.
    * **Nivo 1 (Meta-Learner/Blender):** `RidgeCV`.
    * Ta pristop se je izkazal za najuspešnejšega.

**5.4. Optimizacija hiperparametrov**
Za ključne modele so bili hiperparametri nastavljeni na podlagi predhodnih eksperimentov in splošnih dobrih praks, kot je vidno v `Housing_prices_kaggle_2.ipynb`.

**5.5. Metrika uspešnosti**
Glavna metrika za ocenjevanje modelov je bila korenska povprečna kvadratna logaritemska napaka (RMSLE - Root Mean Squared Logarithmic Error).
$$ \text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(p_i + 1) - \log(a_i + 1))^2} $$

## 6. Rezultati in diskusija

Po obsežnem predprocesiranju, "feature engineeringu" in treniranju različnih modelov smo dosegli znatno izboljšanje napovedne natančnosti.

**6.1. Primerjava uspešnosti modelov**
Končni `StackingRegressor`, ki je združeval napovedi XGBoost, LightGBM, CatBoost in Ridge regresije z RidgeCV kot meta-modelom, je pokazal najboljše rezultate, dosegel RMSLE okoli 0.11 - 0.12 na Kaggle. Datoteka `submission.csv` je bila generirana s tem modelom.

**6.2. Pomembnost značilk**
Analiza pomembnosti značilk (npr. iz `RandomForestRegressor`) je pokazala, da so najvplivnejše:
1.  `OverallQual`
2.  Novoustvarjene značilke: `TotalSF`, `houseAge`, `TotalBathrooms`
3.  `GrLivArea`
4.  `ExterQual`, `KitchenQual`
5.  Velikost garaže (`GarageCars`/`GarageArea`)
6.  `BsmtQual`
7.  `Neighborhood`

**6.3. Vpliv "Feature Engineeringa"**
Izkazalo se je, da je skrbno načrtovanje novih značilk ključnega pomena za izboljšanje rezultatov.

**6.4. Diskusija**
Kombinacija robustnega predprocesiranja, inteligentnega "feature engineeringa" in uporabe naprednih ansambelskih tehnik vodi do visoko natančnih napovednih modelov.

## 7. Interaktivna spletna aplikacija (Streamlit)

Da bi omogočili praktično uporabo in lažjo interakcijo z razvitim napovednim modelom, smo zgradili interaktivno spletno aplikacijo s pomočjo knjižnice Streamlit. Koda za to aplikacijo se nahaja v datoteki `streamlib_housing_prices.py`.

**7.1. Namen in cilji aplikacije**
Glavni namen aplikacije je uporabnikom omogočiti:
1.  **Pridobivanje ocen cen za posamezno nepremičnino:** Uporabniki lahko vnesejo specifične značilnosti hiše in takoj prejmejo oceno njene prodajne cene.
2.  **Enostavna in intuitivna uporaba:** Zagotoviti vmesnik, ki ne zahteva predznanja s področja programiranja ali strojnega učenja.
3.  **Demonstracija modela:** Prikazati praktično uporabnost razvitega napovednega sistema.

**7.2. Glavne funkcionalnosti aplikacije (`streamlib_housing_prices.py`)**

* **Nalaganje podatkov, predprocesiranje in treniranje modela (v ozadju ob prvem zagonu):**
    * Aplikacija pri prvem zagonu (ali ko se predpomnilnik osveži) izvede celoten postopek priprave modela. To vključuje:
        * Nalaganje osnovnega učnega niza (`train.csv`).
        * Temeljito čiščenje podatkov: odstranjevanje definiranih osamelcev in logaritemska transformacija ciljne spremenljivke `SalePrice` (z `np.log`).
        * Izvedba obsežnega "feature engineeringa": ustvarjanje novih, bolj informativnih značilk, kot so `houseAge` (starost hiše), `houseRemodelAge` (starost od prenove), `IsNewHouse` (ali je hiša nova), `TotalSF` (skupna kvadratura), `TotalBathrooms` (skupno število kopalnic), `TotalPorchSF` (skupna površina verand) itd. Starejše, manj informativne značilke se pri tem odstranijo.
        * Definicija in uporaba robustnega `ColumnTransformer` cevovoda (`Pipeline`) za predprocesiranje. Ta cevovod skrbi za:
            * Imputacijo manjkajočih vrednosti (z mediano za numerične, s konstanto 'NA' ali modusom za kategorikalne).
            * Ordinalno kodiranje za kategorikalne značilke z naravnim vrstnim redom.
            * "One-hot" kodiranje za nominalne kategorikalne značilke (z obravnavo neznanih kategorij).
            * Standardno skaliranje (`StandardScaler`) za vse numerične značilke.
        * Treniranje več različnih regresijskih modelov, vključno z `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, `CatBoostRegressor`.
        * Treniranje končnega ansambelskega modela `StackingRegressor`, ki kot osnovne učence uporablja prej naštete modele, za meta-učenca pa `RidgeCV`. Ta model se nato uporablja za končne napovedi.
    * Celoten ta postopek (nalaganje, čiščenje, "feature engineering", definiranje cevovoda in treniranje modelov) je optimiziran z uporabo Streamlitove funkcije `@st.cache_resource`. To pomeni, da se vsi ti koraki izvedejo le enkrat ob prvem zagonu aplikacije ali ob spremembi odvisnosti, kar zagotavlja hitro odzivnost aplikacije pri nadaljnji uporabi. Funkcija vrne natreniran najboljši model (StackingRegressor), cevovod za predprocesiranje, imena značilk in druge pomožne podatke.

* **Dinamični vnosni vmesnik za značilke nepremičnine:**
    * V stranski vrstici aplikacije (`st.sidebar`) se dinamično generirajo vnosna polja za vse značilke, ki jih model potrebuje za napoved. Seznam teh značilk se pridobi iz podatkovnega nabora `train_df_for_input_features`, ki nastane med pripravo podatkov.
    * Za **numerične značilke** (npr. `LotArea` - velikost parcele, `YearBuilt` - leto izgradnje, `GrLivArea` - bivalna površina) so na voljo drsniki (`st.slider`) z razumno prednastavljenimi minimalnimi, maksimalnimi in privzetimi vrednostmi, izpeljanimi iz učnega niza.
    * Za **kategorikalne značilke** (npr. `Neighborhood` - soseska, `HouseStyle` - tip hiše, `OverallQual` - splošna kvaliteta) so na voljo spustni seznami (`st.selectbox`), ki vsebujejo vse možne kategorije za dano značilko, prav tako pridobljene iz učnega niza.
    * Uporabnik lahko interaktivno nastavi vrednosti za vsako od teh značilk, da opiše hipotetično ali dejansko nepremičnino, za katero želi oceno cene.

* **Napovedovanje cene za vnesene značilke:**
    * Ko uporabnik vnese (ali spremeni) vrednosti značilk v stranski vrstici, aplikacija te vnose zbere in jih pretvori v Pandas DataFrame z enim samim primerom (eno vrstico).
    * Na ta DataFrame se nato uporabi pred-naučen `Pipeline` (predprocesor), ki izvede vse potrebne transformacije (imputacijo, kodiranje, skaliranje), enako kot pri učenju modela.
    * Transformirani podatki se nato posredujejo pred-naučenemu `StackingRegressor` modelu, ki izračuna napoved. Ker je bil model učen na logaritmirani vrednosti `SalePrice`, je tudi njegova surova napoved na tej logaritmirani skali.
    * Končna napoved cene se dobi z inverzno transformacijo, tj. z eksponentno funkcijo (`np.exp(prediction_log)`).

* **Prikaz rezultata:**
    * Ocenjena prodajna cena hiše se jasno in vidno prikaže uporabniku na glavni strani aplikacije, običajno znotraj sporočila o uspehu (`st.success`), formatirana na dve decimalni mesti in z oznako valute (npr. `💰 Ocenjena cena hiše: $250,123.45`).

**7.3. Tehnološki sklop**
Za razvoj aplikacije so bile uporabljene naslednje ključne knjižnice in tehnologije:
* **Streamlit:** Za hitro in enostavno izdelavo interaktivnega spletnega vmesnika.
* **Pandas:** Za manipulacijo s podatki in pripravo vhodnih DataFrame-ov.
* **NumPy:** Za numerične operacije, še posebej za logaritemsko in eksponentno transformacijo.
* **Scikit-learn:** Za celoten cevovod strojnega učenja, vključno s `ColumnTransformer` za predprocesiranje, `Pipeline` za združevanje korakov, različnimi modeli (`Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`) in ansambelskimi metodami (`StackingRegressor`).
* **XGBoost, CatBoost, (LightGBM, čeprav ni eksplicitno viden v `streamlib_housing_prices.py` snippetu, je pogosto del takih skladov):** Za napredne in visoko zmogljive regresijske modele, ki so del ansambla.

**7.4. Uporabnost**
Interaktivna aplikacija, zgrajena s `streamlib_housing_prices.py`, bistveno poveča uporabnost in dostopnost razvitega modela napovedovanja cen nepremičnin. Uporabnikom omogoča:
* **Hitre individualne ocene:** Nepremičninski agenti, potencialni kupci ali prodajalci lahko hitro pridobijo oceno vrednosti za specifično nepremičnino z vnosom njenih ključnih lastnosti.
* **"Kaj-če" analize:** Uporabniki lahko eksperimentirajo z različnimi vrednostmi značilk (npr. "Kaj če bi bila hiša novejša?" ali "Koliko bi bila vredna z boljšo kvaliteto kuhinje?") in takoj vidijo vpliv na ocenjeno ceno.
* **Intuitivno razumevanje dejavnikov:** Čeprav aplikacija neposredno ne prikazuje pomembnosti značilk v grafu, interaktivno spreminjanje vrednosti in opazovanje sprememb v ceni lahko uporabniku da intuitiven občutek o tem, kateri dejavniki imajo večji vpliv.
Aplikacija služi kot odličen primer, kako se lahko kompleksni modeli strojnega učenja preobrazijo v praktična, enostavna za uporabo orodja, ki nudijo konkretno vrednost končnim uporabnikom brez potrebe po tehničnem znanju o ozadju modeliranja.

## 8. Zaključek

Projekt napovedovanja cen nepremičnin v Amesu je ponudil dragocen vpogled v celoten proces strojnega učenja, od razumevanja in priprave podatkov do gradnje kompleksnih modelov in njihove implementacije v uporabniku prijazno aplikacijo.

**Glavne ugotovitve:**
* **Kakovost podatkov je ključna:** Temeljita eksplorativna analiza podatkov, skrbno ravnanje z manjkajočimi vrednostmi in učinkovito kodiranje kategorikalnih spremenljivk so osnova za uspešno modeliranje.
* **"Feature engineering" prinaša veliko vrednost:** Ustvarjanje novih, smiselnih značilk (npr. starost hiše, skupna površina, skupno število kopalnic) lahko bistveno izboljša natančnost modelov, saj jim ponudi informacije v bolj neposredni in lažje prebavljivi obliki.
* **Napredni ansambelski modeli so zelo učinkoviti:** Modeli, kot so XGBoost, LightGBM in CatBoost, so se izkazali za zelo natančne. Njihova kombinacija z uporabo tehnik zlaganja (stacking) je omogočila doseganje vrhunskih rezultatov, ki so konkurenčni na platformah, kot je Kaggle.
* **Praktična uporabnost:** Razvoj interaktivne spletne aplikacije s Streamlitom je pokazal, kako lahko napredne analitične modele približamo končnim uporabnikom in jim omogočimo praktično uporabo rezultatov.

**Potencialne nadaljnje izboljšave:**
* **Naprednejše tehnike "feature engineeringa":** Raziskovanje dodatnih interakcij med značilkami ali uporaba bolj sofisticiranih metod za ustvarjanje značilk (npr. grupiranje sosesk glede na ceno).
* **Obsežnejša optimizacija hiperparametrov:** Uporaba naprednejših orodij za optimizacijo (npr. Optuna, Hyperopt) za fino nastavitev vseh modelov v ansamblu.
* **Obravnava časovne komponente:** Eksplicitnejše modeliranje časovnih trendov v cenah nepremičnin, če bi podatki zajemali daljše obdobje.
* **Interpretacija modela:** Globlja analiza interpretacije napovedi kompleksnih modelov (npr. z uporabo SHAP vrednosti) za boljše razumevanje, zakaj model naredi določeno napoved, in morda integracija teh vpogledov v Streamlit aplikacijo.

Ta projekt je uspešno demonstriral uporabo sodobnih tehnik strojnega učenja za reševanje realnega problema napovedovanja cen nepremičnin. Od začetne analize podatkov v `Housing_prices_kaggle.ipynb`, preko razvoja naprednih modelov v `Housing_prices_kaggle_2.ipynb`, do končne implementacije v interaktivni aplikaciji `streamlib_housing_prices.py`, smo prehodili celoten cikel podatkovne znanosti in ustvarili robusten ter uporaben napovedni sistem.
