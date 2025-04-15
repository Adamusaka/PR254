# VMESNO poročilo

## 1. Uvod

V tem vmesnem poročilu predstavljamo pregled problema, uporabljenih podatkov, izvedenih analiz in glavnih rezultatov, pridobljenih s pomočjo analize podatkov iz domačega projekta in datoteke *Housing_prices_kaggle.ipynb*. Namen poročila je predstaviti problem ocenjevanja cen stanovanj in s tem pridobiti vpogled v dejavnike, ki vplivajo na ceno nepremičnine, vse to pa je v okviru dvojnem stron.

## 2. Opis problema

Cilj analize je napovedovanje cen stanovanj na podlagi različnih vhodnih spremenljivk (npr. kvadratura, leto gradnje, lokacija in druge značilne lastnosti). Problem je zastavljen kot regresijska naloga, kjer želimo razviti model, ki bo omogočal natančno določitev cene glede na podane parametre. 

## 3. Podatki

Uporabljeni podatki prihajajo iz poznanega Kaggle tekmovanja, ki se osredotoča na cene stanovanj. V podatkovnem naboru so vključeni različni atributi:
- Kvadratura stanovanja,
- Leto gradnje,
- Lokacija nepremičnine,
- Dodatne značilnosti, kot so stanje objekta in bližina javnih prevozov.

Podatkovni nabor je bil predhodno očiščen, odstranjeni so manjkajoči podatki in morebitne anomalije, kar omogoča bolj zanesljivo analizo.

## 4. Izvedene analize in uporabljena orodja

### 4.1 Priprava podatkov

Za pripravo podatkov je bila izvedena:
- Vizualna analiza porazdelitve posameznih spremenljivk,
- Preverjanje korelacij med spremenljivkami in cenami,
- Normalizacija podatkov, kjer je potrebno.

### 4.2 Izvedene analize

V analizi smo uporabili:
- **Opisno statistiko:** Za vpogled v osnovne značilnosti podatkov, kot so povprečje, mediana in standardni odklon.
- **Korelacijska analiza:** S pomočjo korelacijskega koeficienta smo ocenili vpliv posameznih spremenljivk na ceno.
- **Regresijski modeli:** Za napovedovanje cen smo uporabili linearno regresijo kot osnovni model. Kasneje smo primerjali rezultate z naprednejšimi tehnikami, kot so Random Forest in Gradient Boosting.
- **Vizualizacije:** Izdelane so bile različne grafe, med drugim histogrami, razpršeni grafi in toplotne karte, ki jasno predstavljajo medsebojne odnose med spremenljivkami.

### 4.3 Uporabljena koda in orodja

Za analizo podatkov smo uporabili Python, natančneje knjižnice:
- Pandas za manipulacijo podatkov,
- NumPy za numerične operacije,
- Matplotlib in Seaborn za vizualizacije,
- Scikit-learn za implementacijo regresijskih modelov.

Uporabljena izvorna koda iz datoteke *Housing_prices_kaggle.ipynb* je bila vključena kot referenca v poročilu, kar omogoča sledenje metodologije in ponovljivost analiz. Osrednji del kode je bil namenjen čiščenju podatkov, modeliranju ter vizualizaciji rezultatov.

## 5. Glavne ugotovitve in rezultati

Med glavnimi ugotovitvami poročila so:
- **Povezanost med kvadrato in ceno:** Večja kvadratura neposredno vpliva na višjo ceno nepremičnine.
- **Pomembnost lokacije:** Podatki kažejo, da ima lokacija največji vpliv na ceno, kar potrjuje lokalne tržne razlike.
- **Izboljšava modela z naprednejšimi metodami:** Naprednejši modeli, kot so Random Forest, so dosegli višjo natančnost napovedi v primerjavi z osnovno linearno regresijo.
- **Vizualne predstavitve:** Grafi, posebej toplotna karta korelacij, so pomagali identificirati ključne dejavnike in razvrščanje značilnosti glede na njihov vpliv na ceno.

## 6. Zaključek

Poročilo je predstavilo celovit pregled problema napovedovanja cen nepremičnin. Uporabljeni podatki in izvedene analize so pripomogle, da lahko jasno opredelimo ključne dejavnike, ki vplivajo na ceno. Rezultati kažejo, da večina vpliva izhaja iz kvadrature nepremičnine ter njene lokacije. Nadaljnja izboljšava modelov bi lahko vključevala napredne tehnike strojnega učenja in dodatno obdelavo podatkov.