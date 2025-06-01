To je dokumentaijca za končno poročilo (rabmo 1300 besed)

# Končno poročilo

V prejšnem poročilu smo ostali pri enostavnem testnem modelu. Cilj bo zdaj ustavirti boljši modele in nakoncu še interaktivno spletno stran.

houseAge = YrSold - YearBuilt
*Zakaj? Samo leto izgradnje (“YearBuilt”) sicer pove, kdaj je bila hiša zgrajena, a modelu manjka kontekst, kako “stara” je hiša v času prodaje. Če hiša stoji 100 let, je to povsem drugače kot hiša stare 5 let.
*Kaj zajame? Razmerje med datumom prodaje in datumom izgradnje. Starejše hiše imajo običajno več obrabe, medtem ko novejše (mlajše) hiše pogosto dosegajo višjo ceno, saj je v njih modernejša gradnja, manj popravkov itd.
*Zakaj odstraniti YearBuilt? Ker se informacija o “starosti” izračuna direktno iz nje, je houseAge bolje “sintetiziran” podatek. Če bi pustili oba, bi imeli multikolinearnost (močna korelacija med stolpcema).

houseRemodelAge = YrSold - YearRemodAdd
-Zakaj? Hiša, ki je bila nedavno adaptirana ali obnovljena, je običajno vredna več kot tista, ki ni bila dolgo časa prenovljena. Samo leto zadnje adaptacije (“YearRemodAdd”) modelu ne pove, koliko časa je minilo od takrat do prodaje.
-Kaj zajame? Čas (število let), kolikokrat in kako dolgo je hiša “stara” po zadnji večji obnovi. Denimo, če je bila hiša zgrajena leta 1950, vendar obnovljena leta 2005, bo houseAge 70 let, a houseRemodelAge recimo 20 let (če se prodaja leta 2025).
-Zakaj odstraniti YearRemodAdd? Ker je v praksi pomembneje “koliko časa je minilo od obnove”, ne natanko leto obnove. Tako en stolpec (houseRemodelAge) ujame bistvo in zmanjša redundantnost.

totalSF = 1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2
-Zakaj? Ločeni podatki o kvadraturi pritličja, nadstropja in kletnih površin (t. i. “Finished Basement”) so sicer relevantni, a modelu lažje razume, da je skupna uporabna površina (1. in 2. nadstropje + urejene kletne sobe) ključna pri določanju cene.
-Kaj zajame? Celotno “notranjo” bivalno površino (v kvadratnih metrih). Tipično velja, da večja notranja kvadratura pomeni višjo ceno.
-Zakaj odstraniti posamezne stolpce? Ko združiš vse v totalSF, večina informacij o velikosti hiše ostane ohranjena (in model ne rabi “razmišljati” oštirih ločenih komponentah). Hkrati zmanjšaš število stolpcev in preprečiš, da bi se model preveč navezal na en sam nivo hiše ali kleti.

totalArea = GrLivArea + TotalBsmtSF
-Zakaj? Podobno kot pri totalSF je tukaj namen pokazati vso kvadraturno površino, ki jo je mogoče bivalno izkoristiti:
--GrLivArea = “Above grade” (nadpovršinska živa površina),
--TotalBsmtSF = skupna kvadratura celotne kleti (tudi neurejenih delov).
-Kaj zajame? Kombinacijo bivalne površine nad tlemi in celotne kletne površine, kar pogosto bolje korelira s tržno ceno kot vsak stolpec posebej.
-Zakaj odstraniti originalna polja? Podobno kot pri totalSF, eno združeno število lažje zajame vpliv “celotne kvadrature” in zmanjša število vhodnih atributov.

totalBaths = BsmtFullBath + FullBath + 0.5*(BsmtHalfBath + HalfBath)
-Zakaj? Štetje kopalnic (polnih in polovičnih) ločeno ne pove takoj, koliko kopalnic dejansko “uporabnik” dobi. Polovična kopalnica (“HalfBath”) je pri oceni vrednosti manj vredna od polne.
-Kaj zajame? Enotno število “ekvivalentnih polnih kopalnic”. Eno polovično šteje kot 0,5, ker nima tuša/ščepec in zato pri ceni ne šteje v celoti kot polna.
-Zakaj odstraniti posamezne stolpce? Ker je smiselnejše za model graditi neposredno na združeni oceni števila kopalnic v enotah “polnih kopalnic”, ne pa imeti štiri ločene kolone (BsmtFullBath, FullBath, BsmtHalfBath, HalfBath), ki vsebujejo redundantne informacije.

totalPorchSF = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
-Zakaj? Vse površine zunanjih prostorov (verande, terase, zimski vrt, pokrite lože, dr.) pomembno vplivajo na tržno vrednost. Posamezni tipi (“OpenPorchSF” ipd.) povedo modelu, a če jih kombiniramo, dobimo skupno zunanjo kvadrature, ki je pogosto tista, ki se upošteva pri cenitvah.
-Kaj zajame? Skupno kvadrato vseh zunanjih “dodatkov”, kar za model pomeni, da razume “celotno zunanje življenjsko površino” kot eno samostojno še posebej pomembno merilo.
-Zakaj odstraniti originalne stolpce? Ker so vsi med sabo visoko korelirani (večja veranda = višja vrednost, kakršnakoli), združena vrednost zmanjša število atributov in olajša učenje.


Glavni razlogi za takšno feature engineering:
Manj redundance, manj koreliranih vhodov
– Združene spremenljivke odstranijo možnost, da bi se model preveč “navezal” na posamezne komponente, ki so med seboj močno korelirane (npr. vsaka vrsta kopalnice ali del SF).
– Hkrati zmanjšajo število stolpcev, kar pospeši učenje in zmanjša tveganje prekomernega prileganja (overfitting).
Bolj direktna interpretacija za model
– Modelu ni treba sam izračunati starosti hiše ali vedeti, da je polovična kopalnica vredna polovico. S tem, ko te informacije že vnaprej vključimo, model hitreje ujame ključne vzorce.
Zmnožene interakcije in nelinearnosti
– V praksi ima “skupna kvadratura” močnejši korelacijski odnos s ceno kot ločena kvadratura pritličja, nadstropja in kleti.
– Podobno je z “starostjo hiše” in “starostjo po adaptaciji” – obe v naprej izračunani spremenljivki bolje zajameta vpliv na ceno kot surovi stolpci.
Prilagodljivost in lažja generalizacija
– Če bi kakšne hiše nimale npr. kleti (BsmtFinSF1 = 0 in BsmtFinSF2 = 0), jih z “totalSF” vseeno obravnavamo pravilno. Enako pri manjkajočih vrednostih – če nekaj ni prisotno, je ˝0˝, in vseeno smo pokrili logiko, kaj se zgodi z zunanjo površino ali kopalnicami.