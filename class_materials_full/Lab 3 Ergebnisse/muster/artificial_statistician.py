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
import matplotlib.pyplot as plt

# Sequences
a_ = [0.0300, 0.9100, 0.6400, 0.9900, 0.6400, 0.1600, 0.1600, 0.9100, 0.1600, 0.2700]
b_ = [0.6400, 0.0800, 0.1600, 0.2700, 0.0200, 0.0100, 0.1600, 0.0300, 0.0300, 0.6400]

# Sort data
a_sorted = list(sorted(a_))
b_sorted = list(sorted(b_))

# Names of sequences
name_a = "Algorithm 1"
name_b = "Algorithm 2"

# Set alpha level
alpha_ = 0.05

# Create dict
sequences_ = {name_a: a_sorted, name_b: b_sorted}


def test_all(sequences, alpha):
    print(
        "This is an Artificial Statistician for continuous target values and unpaired data!"
    )
    print(
        "Let's compare", len(sequences), "sequences at an alpha-level of", alpha, "\n"
    )
    print("*****************************************")
    print("Let's first have a look at the properties of the data distributions:\n")

    normal_distributed = list()
    for name, seq in sequences.items():
        normal_distributed.append(0)
        print(name)
        print("sequence: ", seq)

        # normality tests
        # Shapiro-Wilk
        a_shapiro = shapiro(seq)
        print("{:30s} {:30f}".format("Shapiro-Wilk: p-value", a_shapiro[1]))
        if (
            a_shapiro[1] >= 0.05
        ):  # we assume normal distribution if value greater than 0.05
            normal_distributed[-1] += 1

        # Kolmogorov-Smirnov
        a_kolmogorov = kstest(seq, "norm")
        print("{:30s} {:30f}".format("Kolmogorov-Smirnov: p-value", a_kolmogorov[1]))
        if (
            a_kolmogorov[1] >= 0.05
        ):  # we assume normal distribution if value greater than 0.05
            normal_distributed[-1] += 1

        # Anderson (assumes alpha = 0.05)
        a_anderson = anderson(seq, dist="norm")
        print("{:30s} {:30f}".format("Anderson: test statistic", a_anderson[0]))
        print(
            "{:30s} {:30f}".format("Anderson: critical value", a_anderson[1][2]), "\n"
        )
        if (
            a_anderson[0] <= a_anderson[1][2]
        ):  # when test statsitic > critical value (here for 5%) assume normal distr.
            normal_distributed[-1] += 1

        if normal_distributed[-1] >= 2:
            normal_distributed[-1] = True

        # histogram
        if len(sequences) <= 2:
            plt.hist(seq)
            plt.title(f"Histogram der Sequenz {name}")
            plt.show()

    for idx, value in enumerate(normal_distributed):
        print(f"Ergebnis {idx} normalverteilt: {value}")

    for (k, v), nd in zip(sequences.items(), normal_distributed):
        sequences[k] = [v, nd]

    print("*****************************************")
    print("Let's continue with comparing the", len(sequences), "sequences:")

    # box plot
    sequences_values = list(values[0] for values in sequences.values())
    plt.boxplot(sequences_values, labels=list(sequences.keys()), notch=True)
    plt.title(f"Boxplot der Sequenzen")
    plt.show()

    def compare_two(normals, values, result_nrs, less_verbose=False):
        print("\nVergleiche:", result_nrs)

        if sum(normals) == len(normals):
            if variance(values[0]) == variance(values[1]):
                # significance tests
                print("\nT-test assuming norm. distr. & equal sigmas")
                ttest_ab = ttest_ind(values[0], values[1])
                print("{:30s} {:30f}".format("t-statistic", ttest_ab[0]))
                print("{:30s} {:30f}".format("p-value", ttest_ab[1]))
                if ttest_ab[1] >= 0.05:
                    print("Die beiden Verteilungen könnten gleich sein.")
                    if less_verbose:
                        return
                else:
                    print("Die beiden Verteilungen unterscheiden sich signifikant.")
            else:
                print("\nT-test assuming norm. distr. & unequal sigmas")
                ttest_ab_unequal = ttest_ind(values[0], values[1], equal_var=False)
                print("{:30s} {:30f}".format("t-statistic", ttest_ab_unequal[0]))
                print("{:30s} {:30f}".format("p-value", ttest_ab_unequal[1]))
                if ttest_ab_unequal[1] >= 0.05:
                    print("Die beiden Verteilungen könnten gleich sein.")
                    if less_verbose:
                        return
                else:
                    print("Die beiden Verteilungen unterscheiden sich signifikant.")
        else:
            print("mindest eine der beiden verteilung ist nicht normalverteilt:")

            print("\nMann - Whitney U test/ Wilcoxon rank-sum test")
            mannwhitney = mannwhitneyu(values[0], values[1], alternative="two-sided")
            print("{:30s} {:30f}".format("p-value", mannwhitney[1]))
            if mannwhitney[1] <= 0.05:
                print(
                    f"p-value smaller than 0.05, distributions do not differ significantly."
                )
            else:
                print(
                    f"p-value bigger than 0.05, distributions do differ significantly."
                )

            print("\nKolmogorov Smirnov(a,b) test")
            kolmog = ks_2samp(
                values[0], values[1]
            )  # Compute the Kolmogorov-Smirnov statistic on 2 samples.
            print("{:30s} {:30f}".format("p-value", kolmog[1]))

            if kolmog[1] > 0.05:
                print(
                    f"p-value bigger than 0.05, distributions do not differ significantly."
                )
                if less_verbose:
                    return
            else:
                print(
                    f"p-value smaller than 0.05, distributions do differ significantly."
                )

        # effect measures
        c = list(values[0]) + list(values[1])
        c_ranked = rankdata(c, method="average")
        a_ranked = list(c_ranked[: len(values[0])])
        b_ranked = list(c_ranked[len(values[0]) : len(values[0]) + len(values[1])])
        ranksum_a = sum(a_ranked)
        ranksum_b = sum(b_ranked)

        print("\nVargha’s and Delaney’s A Measure")
        print("0.5=no, 0.56=small, 0.64=medium, 0.71=big effect")
        big_a = (
            1 / len(values[1]) * (ranksum_a / len(values[0]) - (len(values[0]) + 1) / 2)
        )
        if big_a < 0.5:
            big_a = 1 - big_a
        print("{:30s} {:30f}".format("A measure", big_a))
        effects_cut = [0.5, 0.56, 0.64, 0.71]
        effects = ["no", "a small", "a medium", "a big"]
        for cut, eff in zip(effects_cut, effects):
            if big_a >= cut:
                effect = eff
        print(f"This A measure correspond to {effect} effect.")

        s_pooled = math.sqrt((variance(values[0]) + variance(values[1])) / 2)
        d = (mean(values[0]) - mean(values[1])) / s_pooled
        if not less_verbose:
            print("\nCohens d measure")
            print("0.25=small, 0.5=medium 0.75=large effect")
            print("{:30s} {:30f}".format("d measure", d))

        # Hedges g measure (p.344 lecture 2018)
        print(
            '\n"Corrected" d-measure, comparable efect size:\n0.25=small, 0.5=medium 0.75=large effect'
        )
        hedges_g2 = (mean(values[0]) - mean(values[1])) / (
            (
                (len(values[0]) - 1) * math.sqrt(variance(values[0])) ** 2
                + (len(values[1]) - 1) * math.sqrt(variance(values[1])) ** 2
            )
            / (len(values[0]) + len(values[1]) - 2)
        ) ** 0.5
        print("{:30s} {:31f}".format("hedges g", hedges_g2))
        hedges_g1 = d * (1 - (3 / (4 * (len(values[0]) + len(values[1])) - 9)))
        print("{:30s} {:30f}".format("hedges g (Korrekturfaktor)", hedges_g1))

        if less_verbose:
            return

        # Glass delta measure
        print("\nGlass delta measure")
        if len(values[0]) > len(values[1]):
            glass_delta = (mean(values[1]) - mean(values[0])) / math.sqrt(
                variance(values[0])
            )
            print("{:30s} {:30f}".format("\nGlass delta measure", glass_delta))

        if len(values[1]) > len(values[0]):
            glass_delta = (mean(values[0]) - mean(values[1])) / math.sqrt(
                variance(values[1])
            )
            print("{:30s} {:30f}".format("\nGlass delta measure", glass_delta))

        if len(values[0]) == len(values[1]):
            glass_delta1 = (mean(values[0]) - mean(values[1])) / math.sqrt(
                variance(values[1])
            )
            glass_delta2 = (mean(values[1]) - mean(values[0])) / math.sqrt(
                variance(values[0])
            )
            print(
                "{:30s} {:30f}".format(
                    "Calculate with Std.Dev(values[0])", glass_delta2
                )
            )
            print(
                "{:30s} {:30f}".format(
                    "Calculate with Std.Dev(values[1])", glass_delta1
                )
            )

    results_nrs = [k for k in sequences.keys()]
    values = [v[0] for v in sequences.values()]
    normals = [v[1] for v in sequences.values()]
    for i in range(len(values)):
        for j in range(len(values)):
            if j > i:
                compare_two(
                    [normals[i], normals[j]],
                    [values[i], values[j]],
                    [results_nrs[i], results_nrs[j]],
                    True,
                )


def compare_eas():
    results_from_file = dict()
    with open("results.txt") as results:
        for line in results:
            if not results_from_file:
                for idx, length in enumerate(line.split()):
                    results_from_file[idx] = [float(length)]
            else:
                for idx, length in enumerate(line.split()):
                    results_from_file[idx].append(float(length))

    test_all(results_from_file, alpha_)


test_all(sequences_, alpha_)
compare_eas()

print(
    "\n\nDie Verteilungen 0,1,8,9 unterscheiden sich dem T-test gemäß nicht signifikant.\n"
    "Die Verteilung unterscheidet sich signifikant von bspw. 2 und 8, die Effektstärke ist jeweils groß."
)
