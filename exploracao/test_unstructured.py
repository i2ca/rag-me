from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("texts/modern_blast_furnace_ironmaking_an_introduction.pdf", url=None)
print("\n\n".join([str(el) for el in elements][300:500]))