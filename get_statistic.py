import csv
bind = ["5АЭ", "3АН", "12А", "9А", "5А ЖДЛ"]

result = {}
weight = {x: {"sum": 0.0, "count": 0, "min": 100000, "max": 0} for x in bind}

with open('metal_statistic.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    first_line = spamreader.__next__()
    print(first_line)
    print(first_line[23])
    for row in spamreader:
        if row[first_line.index('N_АКТА_M7')] is not "":
            result.setdefault(row[first_line.index('N_АКТА_M7')], [])
            result[row[first_line.index('N_АКТА_M7')]].append(row)
count = 0
for item in result.items():
    if len(item[1]) == 1:
        if item[1][0][first_line.index('ЛОМ_ФАКТ')] in bind:
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["count"] += 1
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["sum"] += float(item[1][0][first_line.index('ПР_ЗАСОР_Р')].replace(",", "."))
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["min"] = min(weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["min"], float(item[1][0][first_line.index('ПР_ЗАСОР_Р')].replace(",", ".")))
            weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["max"] = max(weight[item[1][0][first_line.index('ЛОМ_ФАКТ')]]["max"], float(item[1][0][first_line.index('ПР_ЗАСОР_Р')].replace(",", ".")))

for i in weight:
    print("{}: avarage = {}; min = {}; max = {}; on {} wagons".format(i, weight[i]["sum"]/weight[i]["count"], weight[i]["min"], weight[i]["max"], weight[i]["count"]))