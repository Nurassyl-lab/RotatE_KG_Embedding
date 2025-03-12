"""
Reduce triplet dataset
"""
import os, sys
import argparse
import csv
import pandas as pd
import random
import shutil


def get_parser():
    parser = argparse.ArgumentParser(
        description='Nothing here!',
    )

    parser.add_argument('--dataset', type=str, default="FB15k", help="Dataset from which you'd subsample")
    parser.add_argument('--n_ent', type=int, default=10, help="Number of entities to remove")
    parser.add_argument('--degree_th', type=int, default=1, help="Remove entities with degree number degree_th")

    return parser.parse_args()


def get_random_sample(x, sample_size=10):
    if not x:
        return []  # Return empty list if input is empty.

    if sample_size > len(x):
        return random.sample(x, len(x)) #return all if sample size is larger than list.

    return random.sample(x, sample_size)


def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)  # Using makedirs and exist_ok for robustness.
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{directory_path}': {error}")


def write2file(read_filename, write_filename, write_filename_removed, entities2remove):
    with open(read_filename, "r") as read_file:
                with open(write_filename, "w") as write_file:
                    with open(write_filename_removed, "w") as write_removed_file:
                        for line in read_file:
                            e1, rel, e2 = line.split()
                            
                            if e1 not in entities2remove and e2 not in entities2remove:
                                write_file.write(line)
                            else:
                                write_removed_file.write(line)
                             
    print(f"File {read_filename} is succesfully re-writen into {write_filename}, and {write_filename_removed}")


# Load triplet dataset
class Dataset:
    def __init__(self):
        self.dataset = {} # keys are entity names, values are degree based on the train set
        self.entities = []
        self.relations = []

    def get_entity_degree(self, key):
        return self.dataset[key]
    
    def sort_dataset(self, reverse=False):
        # Sorts dataset by degree order

        if reverse: # descending order
            self.dataset = dict(sorted(self.dataset.items(), key=lambda item: item[1], reverse=True))
        else: # ascneding order
            self.dataset = dict(sorted(self.dataset.items(), key=lambda item: item[1], reverse=False))

    def save_dataset(self, csv_filename):
        # convert to a proper dict
        dic = {"entities":[], "degree":[]}
        for key, value in self.dataset.items():
            dic["entities"].append(key)
            dic["degree"].append(value)

        df = pd.DataFrame(dic)
        df.to_csv(csv_filename)
        print(f"CSV file '{csv_filename}' created successfully.")

    def find_entity_by_degree(self, target_degree):
        return [key for key, value in self.dataset.items() if value == target_degree]


if __name__ == "__main__":
    args = get_parser()

    path2datasets = os.path.join("./data/", args.dataset)

    original_dataset = Dataset()

    # load every entity from the train dataset and count the degree
    train_filename = "train.txt"
    with open(os.path.join(path2datasets, train_filename), "r") as read_file:
        for line in read_file:
            e1, rel, e2 = line.split()
            
            if e1 not in original_dataset.entities:
                original_dataset.dataset[e1] = 1
                original_dataset.entities.append(e1)
            else:
                original_dataset.dataset[e1] += 1

            if e2 not in original_dataset.entities:
                original_dataset.dataset[e2] = 1
                original_dataset.entities.append(e2)
            else:
                original_dataset.dataset[e2] += 1

    print(f"Total Number of Entities: {len(original_dataset.entities)}")

    # sort the dictionary
    original_dataset.sort_dataset(reverse=True) 

    # save the dictionary
    # Specify the CSV file name
    csv_filename = os.path.join(path2datasets, args.dataset+"_entity_degree.csv")
    original_dataset.save_dataset(csv_filename=csv_filename)

    path2save = os.path.join("./data/", args.dataset+f"_reduced_n{args.n_ent}_deg{args.degree_th}")
    create_directory(path2save)

    # remove entities
    entities2remove = original_dataset.find_entity_by_degree(target_degree=args.degree_th)
    entities2remove = get_random_sample(x=entities2remove, sample_size=args.n_ent)

    with open(os.path.join(path2save, "entities2remove.dict"), "w") as write_file:
        for i, e in enumerate(entities2remove):
            write_file.write(f"{i}\t{e}\n")

    test_filename = "test.txt"
    valid_filename = "valid.txt"
    
    # Save
    print(f"Processing train set...")
    write2file(
         read_filename=os.path.join(path2datasets, train_filename),
         write_filename=os.path.join(path2save, train_filename),
         write_filename_removed=os.path.join(path2save, train_filename.split(".")[0]+"_removed.txt"),
         entities2remove=entities2remove
    ) 
    print(f"Processing test set...")
    write2file(
         read_filename=os.path.join(path2datasets, test_filename),
         write_filename=os.path.join(path2save, test_filename),
         write_filename_removed=os.path.join(path2save, test_filename.split(".")[0]+"_removed.txt"),
         entities2remove=entities2remove
    ) 
    print(f"Processing valid set...")
    write2file(
         read_filename=os.path.join(path2datasets, valid_filename),
         write_filename=os.path.join(path2save, valid_filename),
         write_filename_removed=os.path.join(path2save, test_filename.split(".")[0]+"_removed.txt"),
         entities2remove=entities2remove
    )

    # Copy other files
    shutil.copy(os.path.join(path2datasets, "relations.dict"), os.path.join(path2save, "relations.dict"))

    # remove entities from entities.dict
    with open(os.path.join(path2datasets, "entities.dict"), "r") as read_file:
        with open(os.path.join(path2save, "entities.dict"), "w") as write_file:
            i = 0
            for line in read_file:
                _, ent = line.split()

                if ent not in entities2remove:
                    write_file.write(f"{i}\t{ent}\n")
                    i+=1
