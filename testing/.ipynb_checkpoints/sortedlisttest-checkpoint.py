import csv
import textwrap
import timeit

DATASIZES = [100, 1000, 10000, 100000, 1000000]
QUERIES = [10, 100, 1000, 10000]

def main():
    data = [] # the entire csv file
    for datasize in DATASIZES:
        times = [] # times for a given model/query combination
        for query in QUERIES:
            setup =\
                """
                import random
                from models.sortedlist import SortedList as sortedlist
                up = 10 * {ds}
                dataset = sortedlist.fromlist([random.randrange(up) for _ in range({ds})])
                largest = dataset[-1]
                """.format(ds=datasize)
            statement =\
                """
                dataset.index(random.randrange(largest))
                """

            trials = timeit.repeat(
                setup=textwrap.dedent(setup),
                stmt=textwrap.dedent(statement),
                repeat=100,
                number=query
            )
            times.append(sum(trials)/len(trials))
        data.append(times)
    
    with open("testing/data/sortedlist.csv", "w") as file:
        fout = csv.writer(file)
        fout.writerow([""] + QUERIES)
        for times, size in zip(data, DATASIZES):
            fout.writerow([size] + times)

if __name__ == "__main__":
    main()