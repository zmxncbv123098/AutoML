import csv
from statistics import median
bind = ["5АЭ", "3АН", "12А", "9А", "5А ЖДЛ"]

result = {}
weight = {x: {"sum": 0.0, "count": 0, "min": 100000, "max": 0, "all": [], "average": 0, "median": 0} for x in bind}

with open('metal_statistic.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    first_line = spamreader.__next__()
    print(first_line)
    for row in spamreader:
        if row[first_line.index('N_АКТА_M7')] is not "":
            result.setdefault(row[first_line.index('N_АКТА_M7')], [])
            result[row[first_line.index('N_АКТА_M7')]].append(row)
count = 0
for item in result.items():
    if len(item[1]) == 1:
        if item[1][0][first_line.index('ЛОМ_ФАКТ')] in bind:
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["count"] += 1
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["sum"] += float(item[1][0][first_line.index('ВЕС_ФАКТ')].replace(",", "."))
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["all"].append(float(item[1][0][first_line.index('ВЕС_ФАКТ')].replace(",", ".")))
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["min"] = min(weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["min"], float(item[1][0][first_line.index('ВЕС_ФАКТ')].replace(",", ".")))
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["max"] = max(weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["max"], float(item[1][0][first_line.index('ВЕС_ФАКТ')].replace(",", ".")))

for i in weight:
    weight[i]["average"] = round(weight[i]["sum"]/weight[i]["count"], 3)
    print("{}: average = {}; min = {}; max = {}; on {} wagons".format(i,
                                                                      round(weight[i]["sum"]/weight[i]["count"], 3),
                                                                      weight[i]["min"],
                                                                      weight[i]["max"],
                                                                      weight[i]["count"]))
print("# - " * 5)
for i in weight:
    weight[i]["median"] = round(median(sorted(weight[i]["all"])), 3)
    print("{}: median = {}; min = {}; max = {}; on {} wagons".format(i,
                                                                      median(sorted(weight[i]["all"])),
                                                                      weight[i]["min"],
                                                                      weight[i]["max"],
                                                                      weight[i]["count"]))

for j in range(3, 7):
    print("# - # - # " * 5)
    for i in weight:
        count_median = 0
        count_average = 0
        for w in weight[i]["all"]:
            if weight[i]["average"] - j < w < weight[i]["average"] + j:
                count_average += 1
            if weight[i]["median"] - j < w < weight[i]["median"] + j:
                count_median += 1
        print("{}: +- {} near average: {}/{}".format(i, j, count_average, weight[i]["count"]))
        print("{}: +- {} near median: {}/{}".format(i, j, count_median, weight[i]["count"]))
        print()
