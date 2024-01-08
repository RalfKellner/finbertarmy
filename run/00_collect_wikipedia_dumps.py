from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
ds.save("/Users/ralf_kellner/Data/Textdata/wikipedia_dumps")
