



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--processor", type=str, required=True)
    parser.add_argument("--split", type=str | list[str], required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    args = parser.parse_args()
    
    process_dataset(args.dataset_name, args.processor, args.split, args.repeat, args.num_proc)
    