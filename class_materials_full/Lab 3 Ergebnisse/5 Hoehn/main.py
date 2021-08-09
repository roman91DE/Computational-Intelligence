from typing import Dict, List
from scipy.stats import (
    shapiro,
    anderson,
    kstest,
    ttest_ind,
    mannwhitneyu,
    ks_2samp,
    rankdata,
)
import math
from statistics import variance, mean
from matplotlib import pyplot as pypl


# read results from lab 3 (
# (i copied the results from the excel file to the file auswertung.csv)
def read_results(path: str) -> Dict[str, List[float]]:

    file = open(path)
    first_line = True
    names, vals = [], []

    for line in file:
        if first_line:
            names = line.split()
            n = len(names)
            for _ in range(n):
                vals.append([])
            first_line = False
            continue
        temp = list(map(float, line.split()))
        for i in range(n):
            vals[i].append(temp[i])
    file.close()

    sequences = {}
    for name, data in zip(names, vals):
        data.sort()
        sequences[name] = data

    return sequences


# sequences = read_results("auswertung.csv")

# Sequences
a = [0.0300, 0.9100, 0.6400, 0.9900, 0.6400, 0.1600, 0.1600, 0.9100, 0.1600, 0.2700]
b = [0.6400, 0.0800, 0.1600, 0.2700, 0.0200, 0.0100, 0.1600, 0.0300, 0.0300, 0.6400]

# Sort data
a_sorted = list(sorted(a))
b_sorted = list(sorted(b))

# Names of sequences
name_a = "Algorithm 1"
name_b = "Algorithm 2"


# Create dict
sequences = {name_a: a_sorted, name_b: b_sorted}

# Set alpha level
alpha = 0.05


def test_all(sequences: Dict, alpha):

    normal_distributed = False

    print(
        "This is an Artificial Statistician for continuous target values and unpaired data!"
    )
    print(
        "Let's compare", len(sequences), "sequences at an alpha-level of", alpha, "\n"
    )
    print("*****************************************")
    print("Let's first have a look at the properties of the data distributions:\n")

    n_samples = len(sequences)

    for name, seq in sequences.items():
        print(name)
        print("sequence: ", seq)

        # normality tests

        # HA:
        normal_distributed = False

        # Shapiro-Wilk
        a_shapiro = shapiro(seq)
        print("{:30s} {:30f}".format("Shapiro-Wilk: p-value", a_shapiro[1]))

        # if p > alpha: normal_distributed = True

        # Kolmogorov-Smirnov
        a_kolmogorov = kstest(seq, "norm")
        print("{:30s} {:30f}".format("Kolmogorov-Smirnov: p-value", a_kolmogorov[1]))

        # Anderson (assumes alpha = 0.05)
        a_anderson = anderson(seq, dist="norm")
        print("{:30s} {:30f}".format("Anderson: test statistic", a_anderson[0]))
        print(
            "{:30s} {:30f}".format("Anderson: critical value", a_anderson[1][2]), "\n"
        )

        # if test_statistic > critical value?   reject h0

        # kolmogorov-smirnov Test habe ich hier ignoriert

        if (a_shapiro[1] > alpha) and (a_anderson[0] <= a_anderson[1][2]):
            normal_distributed = True

        if normal_distributed:
            print("{} is normal distributed!".format(name))
        else:
            print("{} is not normal distributed!".format(name))

    # plotting data
    for key, val in sequences.items():
        # histogram
        pypl.subplot(1, 2, 1)
        pypl.hist(val, density=True, bins="auto")
        pypl.ylabel("Probability")
        pypl.xlabel("Value")
        pypl.title("Histogram {}".format(key))
        # boxplot
        pypl.subplot(1, 2, 2)
        pypl.boxplot(val, notch=True, autorange=True)
        pypl.title("Boxplot {}".format(key))
        pypl.show()

    if n_samples == 2:

        print("*****************************************")
        print("Let's continue with comparing the", len(sequences), "sequences:")

        a = sequences[name_a]
        b = sequences[name_b]

        # significance tests

        if normal_distributed:

            print("\nT-test assuming norm. distr. & equal sigmas")
            ttest_ab = ttest_ind(a, b)
            print("{:30s} {:30f}".format("t-statistic", ttest_ab[0]))
            print("{:30s} {:30f}".format("p-value", ttest_ab[1]))

            if ttest_ab[1] < alpha:
                print(
                    "Result for T-test assuming norm. distr. & equal sigmas: Reject H0"
                )
            else:
                print(
                    "Result for T-test assuming norm. distr. & equal sigmas: Accept H0"
                )

            print("\nT-test assuming norm. distr. & unequal sigmas")
            ttest_ab_unequal = ttest_ind(a, b, equal_var=False)
            print("{:30s} {:30f}".format("t-statistic", ttest_ab_unequal[0]))
            print("{:30s} {:30f}".format("p-value", ttest_ab_unequal[1]))

            if ttest_ab_unequal[1] < alpha:
                print(
                    "Result for T-test assuming norm. distr. & unequal sigmas: Reject H0"
                )
            else:
                print(
                    "Result for T-test assuming norm. distr. & unequal sigmas: Accept H0"
                )

        elif not normal_distributed:

            print("\nMann - Whitney U test/ Wilcoxon rank-sum test")
            mannwhitney = mannwhitneyu(a, b, alternative="two-sided")
            print("{:30s} {:30f}".format("p-value", mannwhitney[1]))

            if mannwhitney[1] < alpha:
                print(
                    "Result for Mann - Whitney U test/ Wilcoxon rank-sum test: Reject H0"
                )
            else:
                print(
                    "Result for Mann - Whitney U test/ Wilcoxon rank-sum test: Accept H0"
                )

            print("\nKolmogorov Smirnov(a,b) test")
            kolmog = ks_2samp(
                a, b
            )  # Compute the Kolmogorov-Smirnov statistic on 2 samples.
            print("{:30s} {:30f}".format("p-value", kolmog[1]))

            if kolmog[1] < alpha:
                print("Result for Kolmogorov Smirnov Test: Reject H0")
            else:
                print("Result for Kolmogorov Smirnov Test: Accept H0")

                # effect measures
        c = list(a) + list(b)
        c_ranked = rankdata(c, method="average")
        a_ranked = list(c_ranked[: len(a)])
        b_ranked = list(c_ranked[len(a) : len(a) + len(b)])
        ranksum_a = sum(a_ranked)
        ranksum_b = sum(b_ranked)

        print("\nVargha’s and Delaney’s A Measure")
        print("0.5=no, 0.56=small, 0.64=medium, 0.71=big effect")
        A = 1 / len(b) * (ranksum_a / len(a) - (len(a) + 1) / 2)
        if A < 0.5:
            A = 1 - A
        print("{:30s} {:30f}".format("A measure", A))

        print("\nCohens d measure")
        print("0.25=small, 0.5=medium 0.75=large effect")
        s_pooled = math.sqrt((variance(a) + variance(b)) / 2)
        d = (mean(a) - mean(b)) / s_pooled
        print("{:30s} {:30f}".format("d measure", d))

        # Hedges g measure (p.344 lecture 2018)
        hedges_g2 = (mean(a) - mean(b)) / (
            (
                (len(a) - 1) * math.sqrt(variance(a)) ** 2
                + (len(b) - 1) * math.sqrt(variance(b)) ** 2
            )
            / ((len(a) + len(b) - 2))
        ) ** 0.5
        print("{:30s} {:31f}".format("\nhedges g", hedges_g2))
        hedges_g1 = d * (1 - (3 / (4 * (len(a) + len(b)) - 9)))
        print("{:30s} {:30f}".format("hedges g (Korrekturfaktor)", hedges_g1))

        # Glass delta measure
        print("\nGlass delta measure")
        if len(a) > len(b):
            glass_delta = (mean(b) - mean(a)) / math.sqrt(variance(a))
            print("{:30s} {:30f}".format("\nGlass delta measure", glass_delta))

        if len(b) > len(a):
            glass_delta = (mean(a) - mean(b)) / math.sqrt(variance(b))
            print("{:30s} {:30f}".format("\nGlass delta measure", glass_delta))

        if len(a) == len(b):
            glass_delta1 = (mean(a) - mean(b)) / math.sqrt(variance(b))
            glass_delta2 = (mean(b) - mean(a)) / math.sqrt(variance(a))
            print("{:30s} {:30f}".format("Calculate with Std.Dev(a)", glass_delta2))
            print("{:30s} {:30f}".format("Calculate with Std.Dev(b)", glass_delta1))

    # hier bin ich nicht mehr fertig geworden...
    if n_samples > 2:
        pass


test_all(sequences, alpha)
