# Computational Intelligence - Lab 1

Student: Roman T. Höhn
E-Mail: rohoehn@students.uni-mainz.de

## Generelle Informationen

Der Code für die 3 Aufgaben befindet sich in dem Ordner "/code", die Datei "facilities.py" enthält universell benötigte Funktionen und stellt die Entfernungsmatrix bereit. Alle Plots sind in dem Ordner "/plots" gespeichert

## Aufgabe 1

Bruteforce Algorithmus in der Datei e1.py (Lösungen als Konsolenoutput)
Schätzungen der Laufzeit anhand einer OLS Regression, basierend auf den gemessenen Laufzeiten, in der jupyter notebook Datei /predictions/e1_regression.ipynb (Plot wird innerhalb des Notebooks erzeugt)

## Aufgabe 2

Ich habe 2-opt sowohl mit einer Zufalls-Startlösung als auch mit einer Lösung des Next-Neighbour Algorithmus getestet.

Die globale Minimum für die ersten 10 Städte (per Bruteforce gefunden) hat eine Entfernung von 2275km

Der Next-Neighbour Algorithmus findet die optimale Lösung (i.d.R.) nicht, der anschließende 2-Opt Algorithmus schafft es jedoch meistens die Lösung bis auf ein globales Minimum zu optimieren. Bei der Initialisierung von 2-opt mit einer Zufallsroute wird nur bei manchen Durchläufen das globale Minimum gefunden, häufig werden jedoch längere Routen als Output ausgegeben (ebenfalls im Output aufgelistet).

Wenn in Zeile 82 der Wert von n auf 16 erhöht wird, findet der Algorithmus Lösungen mit einer Entfernung im Bereich von ca. 2.800-3.000 KM (also keine Garantie, dass das globale Minimum gefunden wird)

Im Konsolenoutput werden die Touren, Entfernungen und die Anzahl an Schritten im 2-Opt Verfahren ausgegeben
(erst für Random->2opt und anschließend für Greedy->2opt). 
2-opt benötigt für 16 Städte ~0.0005 Sekunden, Brute Force würde basierend auf meiner Vorhersage 163527125875476,5 Sekunden benötigen.

## Aufgabe 3

Simulated Annealing findet ebenfalls in den meisten Durchläufen eine Route mit der Entfernung von 2275 km (für n=10), die Laufzeit liegt bei ~0,02 Sekunden und ist damit etwas länger als das 2-opt Verfahren (ich vermute, das liegt daran, dass deutlich mehr Hilfsfunktionen aufgerufen werden), je nachdem welche Cooling Funktion verwendet wird und wie die Parameter eingestellt werden, schwanken die Kennzahlen durchaus stark.

Für das vorliegende TSP Problem, bringt sowohl die Kombination aus Next-Neighbhour und 2-Opt sowie Simulated Annealing zuverlässig gute Ergebnisse und in den meisten Durchlaufen ein globales Minimum. Next Neighbour und 2-opt haben einen Vorteil was die Laufzeit angeht, Simulated Annealing bringt jedoch durch die Einstellung von Parametern und Hilfsfunktionen eine höhere Flexibilität mit sich (Vorteil - Breitere Anwendungsmöglichkeiten auf andere Probleme; Nachteil - Adjustierung per Hand eventuell aufwändiger).
