# Lab 3 - Statistics

## Excercise 1

### Cohen, J. - The earth is round (1994)

Abstract: Author kritisiert/ hinterfragt die gängige Praxis des Tests von Nullhypothesen anhand des verbeiteten 5% Signifikanzniveaus. Ziel ist es die Probleme dieser Vorgehensweise aufzuzeigen, insbesondere die Fehlinterpretation des P-Werts sowie die Missinterpretation, dass das Verwerfen der Nullhypothese mit der Bestätigung der Alternativhypothese gleichgesetzt werden kann. 

- Behauptung: Die statistischen Methoden die aktuell verwendet werden, insbesondere NHST (=Null Hypothesis Significance Testing), sind Relikte aus einer vergangenen Zeit und nicht mehr zeitgemäß
- Konflikt zwischen eigentlichem Ziel und statistischer Aussagekraft von NHST
- Ziel: Wie hoch ist die Wahrscheinlichkeit, dass H0 wahr ist bei den erhobenen Daten?
- Aussage NHST: Unter der Annahme, dass H0 wahr ist, wie hoch ist die Wahrscheinlichkeit dafür, die erhobenen Daten (oder noch extremere Daten) zu messen?
- Kritisierte Fehlinterpretation von NHST: 
    - Annahme: Wenn H0 wahr ist, würden die folgenden Ergebnisse wahrscheinlich nicht auftreten
    - Messung: Die Ergebnisse sind eingetreten
    - Schluss: H0 ist wahrscheinlich nicht wahr und daher formal ungültig

- P(D|H0) != P(H0|D)
- P(D|H0) -> Wahrscheinlichkeit D zu messen wenn H0 wahr ist
- P(H0|D) -> Wahrscheinlichkeit das H0 wahr ist wenn D gemessen wurde, gemäß des Satz von Bayes muss hierfür die prior Wahrscheinlichkeit P(H0) bekannt sein (i.d.R.unbekannt). Das Problem wird in der Praxis durch Schätzung oder Wahrscheinlichkeitsverteilungen umgangen.

- Beispiel
- Krankheit betrifft 2% der Bevölkerung, Test auf Krankheit hat >= 95% Präzision
- Beim positiven Test (=D) ist die Chance, dass die Person nicht krank ist daher < 5% und somit statistisch signifikant (bei alpha=0,05)
    - > H0, dass Person nicht krank ist, wird verworfen
- P(H0|D): Wahrscheinlichkeit, dass Person nicht krank ist und ein positiver Test vorliegt ist jedoch nicht 5% sondern liegt bei >60% ! Prior Wahscheinlichkeit P(H0) liegt bei 98% wodurch es zu einer enormen Abweichung von dem erwarteten Ergebnissen kommt (hohe Anzahl an falsch positiven Ergebnissen im Verhältnis zu richtig positiven Ergebnissen)

Weitere Kritik: Auch bei richtiger Interpretation/Anwendung ist es unwissenschaftlich, eine HA als bestätigt/erwiesen zu betrachten, nachdem bei einer NHST H0 verworfen wurde. Grund hierführ können z.B. die Stichprobenmenge, Ceteris Paribus Annahmen, Einfluss von anderen (auxillary) Theorien

Vorgeschlagene Vorgehensweise:
1. Es gibt keine universelle Alternative zu NHST 
2. "Detektivarbeit/ Exploration": Einsatz von einfachen/universellen/flexiblen/grafischen Verfahren um Struktur und Aufbau der erhobenen Daten besser einschätzen zu können
3. Effektstärken standardmäßig in Form von Konfidenzintervallen angeben. Diese umfassen neben den Informationen, die auch durch NHST angegeben werden, zusätzliche Aussagekraft



### Hentschke & Stüttgen - Computation of measures of effect size for neuroscience data sets (2011)

Abstract: Kritik am weit verbreiteten Einsatz von Signifikanztests mit Hilfe des P-Werts und des willkürlich gewählten 5% Signifikanzniveau. Als Alternative wird MES (Measure of effect Size) vorgestellt, an mehreren Beispielen soll verdeutlicht werden, warum MES vorteilhaft ggü. NHST ist. Neben einer Auflistung der Vor/Nachteile beider Methoden wird eine Umsetzung der entwickelten Toolboc in der Statistik Software Matlab vorgestellt.

- > 90% aller Paper in der Psychologie nutzen NHST (Zeitpunk 2011)
- Schwierig bis unmöglich Arbeiten zu veröffentlichen, die nicht dem willkürlich gewählten 5% Signifikanztest bestehen
- Empfohlene Alternative: Effektstärke und angemessene Konfidenzintervalle
- Ursache für Einsatz von NHST trotz kontinuierlicher Kritik? Umsetzung aternativer Methoden in vielen gängigen Statistik Programmen nicht/nur umständlich möglich

#### Shortcomings of NHST
 - Verweis auf Paper von Cohen(1994)
 - P Wert als Maß für die Wahrscheinlichkeit das die Differenz zwischen zwei Stichprobenmittelwerten (H0: kein Unterschied zwischen beiden Gruppen) signifikant ist (p<0.05 ? H0 ablehnen)
 - Vorgehensweise ist in vielen gängigen Statistischen Test integriert (Chi Quadrat, t-Test, ANOVA, Mann-Whitney U etc.)
 - Häufige Missinterpretation der Ergebnisse im wisschenschaftlichen Arbeiten 
 - kleiner P-Wert wird mit großer Effektstärke gleichgesetzt, in Wahrheit ist der P-Wert jedoch von mehreren Variablen abhängig (N, Stichprobenart, Abhängigkeiten, Testform)
 
 #### Effect Size
  - Beziffert den Unterschied zwischen den Stichproben aus zwei Testgruppen (H0: kein Unterschied der Mittelwerte)
  - Unstandartisierte MES: Abhängig und ausgedrückt in der betrachteten Zielgröße (z.B. Anzahl richtiger Antworten, Laufzeit etc)
  - -> Pro: Besser verständliche unt intuitive Interpretation des Ergebnisses
  - Standartisierte MES: Unabhängig von der betrachteten Metrik (z.B. Einheiten der Std.Abweichung) -> Im Fokus des Artikels
  - -> Pro: Erlaubt den Vergleich zwischen verschiedenen Studien, Samples etc.


Beispiel: Hedge's g

g = (mittelwert_behandlungsgruppe - mittelwert_kontrollgruppe) / std_abw_kombiniert
 -> Ergebnis ist in einer "universellen/kombinierten" Einheit

 #### Advantages of MES (Erfahrungen der Authoren)
1. Im Gegensatz zum P-Wert ist MES nicht abhängig vom Stichprobenumfang
2. Identifikation von Effekten/Phänomen, die anhand des P-Werts nicht zu erkennen gewesen wären
3. Ergebnisse auf Grundlage des P-Werts sind dichotom (Annahme/Verwerfung), anhand des MES können Ergebnisse quantifiziert werden und ermöglichen eine Quantifizierung
