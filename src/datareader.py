from corpus_reader.benchmark_reader import Benchmark
from corpus_reader.benchmark_reader import select_files


class DataLoader:
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


def main():
    dset = DataLoader("./datasets/train")
    dset.printStats()
    dset.printEntryInfo(200)


if __name__ == "__main__":
    main()
