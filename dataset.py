from schnetpack.data.atoms import create_dataset

QM9 = create_dataset(datapath="/home/ubuntu/algo/data/qm9_micro_enriched.pt", format = "tala3 chnia", distance_unit=" tala3ha",property_unit_dict={"energy": "eV", "forces": "eV/angstrom"})

# tala3 
print(QM9)