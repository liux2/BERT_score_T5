from corpus_reader.benchmark_reader import Benchmark
from corpus_reader.benchmark_reader import select_files
import csv


class DataGenerator:
    def __init__(self, path):
        """Load datasets."""
        self.b = Benchmark()
        self.b.fill_benchmark(select_files(path))

    def printStats(self):
        """Print stats of the dataset split."""
        print("-------------------------")
        print("Number of entries: ", self.b.entry_count())
        print("Number of texts: ", self.b.total_lexcount())
        print("Number of distinct properties: ", len(list(self.b.unique_p_mtriples())))

    def printEntryInfo(self, row):
        """Print stats of an entry."""
        entry = self.b.entries[row]
        print("-------------------------")
        print("Entry size: ", entry.size)
        print("Entry catagory: ", entry.category)
        print("Entry shape: ", entry.shape)
        print("Entry shape type: ", entry.shape_type)
        print("Entry triples: ", entry.list_triples())
        print("Entry text: ")
        for i in entry.lexs:
            print("    ", i.lex)
        # print("Entry links s: ", entry.links[0].s)
        # print("Entry links o: ", entry.links[0].o)
        # print("Entry links p: ", entry.links[0].p)

    def creating_split(self, path):
        """Create training/dev/testing sets."""
        with open(path, "w+") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["triple", "sentence"])
        with open(path, "a") as f:
            csvwriter = csv.writer(f)
            for row in range(self.b.entry_count()):
                entry = self.b.entries[row]
                for tri in entry.list_triples():
                    for sent in entry.lexs:
                        csvwriter.writerow([tri, sent.lex])


def main():
    # Train set
    train_set = DataGenerator("./datasets/train")
    train_set.creating_split("./datasets/train_set.csv")
    # # Dev set
    dev_set = DataGenerator("./datasets/dev")
    dev_set.creating_split("./datasets/dev_set.csv")
    # test set
    test_set = DataGenerator("./datasets/test")
    test_set.creating_split("./datasets/test_set.csv")


if __name__ == "__main__":
    main()
