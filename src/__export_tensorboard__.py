from tensorflow.python.summary.event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE

x = EventAccumulator("summaries/logs/train/FreksenThinkDeep-5baf7ab",
                     size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
x.Reload()
print(x.Tags())

with open("steps.csv", "w") as file:
    for s in x.Scalars("main_DDQRN_1/extra_stats/steps"):
        file.write("%s,%s,%s\n" % s)

with open("histogram.csv", "w") as file:
    for h in x.Histograms("main_DDQRN/output_layer/prediction/predictions"):
        l = h[2][5]
        b = h[2][6]
        d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(l)):
            idx = round(l[i])
            if idx > 8:
                idx = 8
            d[idx] += int(b[i])

        file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (h[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))
