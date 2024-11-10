import os
import pandas as pd
import csv

def add_header_and_clean_csv_files(directory="./data"):

    # # training file header
    # header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"

    # test file header
    header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,Borda ranking Winner,Plurality ranking Winner,STV Winner,Approval Voting (AV) Winner,Proportional Approval Voting (PAV) Winner,Approval Chamberlin-Courant (CC) Winner,Lexicographic Chamberlin-Courant (lex-CC) Winner,Sequential Approval Chamberlin-Courant (seq-CC) Winner,Monroe's Approval Rule (Monroe) Winner,Greedy Monroe Winner,Minimax Approval Voting (MAV) Winner,Method of Equal Shares (aka Rule X) with Phragmén phase Winner,E Pluribus Hugo (EPH) Winner,Random Serial Dictator Winner,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"

    test_files = True

    # Split header into list for comparison and DataFrame manipulation
    header_list = header.split(",")

    # Loop through all files in the directory that end with ".csv"
    for filename in os.listdir(directory):
        # if filename.endswith("-TRAIN.csv"):
        if filename.endswith("-TEST.csv"):

            if test_files:
                if "committee_size=1" in filename:
                    header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,Borda ranking Winner,Plurality ranking Winner,STV Winner,Approval Voting (AV) Winner,Proportional Approval Voting (PAV) Winner,Approval Chamberlin-Courant (CC) Winner,Lexicographic Chamberlin-Courant (lex-CC) Winner,Sequential Approval Chamberlin-Courant (seq-CC) Winner,Monroe's Approval Rule (Monroe) Winner,Greedy Monroe Winner,Minimax Approval Voting (MAV) Winner,Method of Equal Shares (aka Rule X) with Phragmén phase Winner,E Pluribus Hugo (EPH) Winner,Random Serial Dictator Winner,Anti-Plurality Winner,Two-Approval Winner,Three-Approval Winner,Instant Runoff Winner,Bottom-Two-Runoff Instant Runoff Winner,Benham Winner,PluralityWRunoff PUT Winner,Coombs Winner,Baldwin Winner,Strict Nanson Winner,Weak Nanson Winner,Iterated Removal Condorcet Loser Winner,Raynaud Winner,Tideman Alternative Top Cycle Winner,Knockout Voting Winner,Banks Winner,Condorcet Winner,Copeland Winner,Llull Winner,Uncovered Set Winner,Uncovered Set - Fishburn Winner,Uncovered Set - Bordes Winner,Uncovered Set - McKelvey Winner,Slater Winner,Top Cycle Winner,GOCHA Winner,Bipartisan Set Winner,Minimax Winner,Split Cycle Winner,Ranked Pairs ZT Winner,Ranked Pairs TB Winner,Simple Stable Voting Winner,Stable Voting Winner,Loss-Trimmer Voting Winner,Daunou Winner,Blacks Winner,Condorcet IRV Winner,Condorcet IRV PUT Winner,Smith IRV Winner,Smith-Minimax Winner,Condorcet Plurality Winner,Copeland-Local-Borda Winner,Copeland-Global-Borda Winner,Borda-Minimax Faceoff Winner,Bucklin Winner,Simplified Bucklin Winner,Weighted Bucklin Winner,Bracket Voting Winner,Superior Voting Winner,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"
                else:
                    header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,Borda ranking Winner,Plurality ranking Winner,STV Winner,Approval Voting (AV) Winner,Proportional Approval Voting (PAV) Winner,Approval Chamberlin-Courant (CC) Winner,Lexicographic Chamberlin-Courant (lex-CC) Winner,Sequential Approval Chamberlin-Courant (seq-CC) Winner,Monroe's Approval Rule (Monroe) Winner,Greedy Monroe Winner,Minimax Approval Voting (MAV) Winner,Method of Equal Shares (aka Rule X) with Phragmén phase Winner,E Pluribus Hugo (EPH) Winner,Random Serial Dictator Winner,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"
            else:
                # train header
                header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"

            file_path = os.path.join(directory, filename)

            # Load the CSV file
            with open(file_path, 'r') as file:
                # Read the first line to check for the header
                first_line_string = file.readline()
                first_line = first_line_string.strip().split(",")

            # Load the data, skipping the header if it already exists
            # if first_line == header_list:
            if header in first_line_string:
                continue
                print("Saving without modifying")
                df = pd.read_csv(file_path, skiprows=1, header=None)

                # Remove extraneous commas by trimming excess columns
                df = df.iloc[:, :len(header_list)]
            else:
                df = pd.read_csv(file_path, header=None)

                # Remove extraneous commas by trimming excess columns
                df = df.iloc[:, :len(header_list)]

                # Insert the header row at the top
                df.columns = header_list
                print(f"First line was: {first_line_string}")

            # # Save the modified file, including the header
            # df.to_csv(file_path, index=False, header=header_list)
            print(f"Processed file: {filename}\n")
            exit()


def ensure_consistent_commas(directory="./data", count_only=False):
    def count_commas(value):
        return str(value).count(',')

    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if "mixed" in filename:
                continue
            file_path = os.path.join(directory, filename)

            # Load the CSV file
            df = pd.read_csv(file_path, dtype=str)  # Load all values as strings to check commas

            # Track rows to replace due to inconsistent comma counts
            inconsistent_rows = []

            # Check each column for consistent comma counts across rows
            for col in df.columns:
                # Get the comma counts for each value in the column
                comma_counts = df[col].apply(count_commas)

                # Find the most common comma count in the column
                common_count = comma_counts.mode().iloc[0]

                # Identify rows where the comma count doesn't match the most common count
                mismatched_rows = comma_counts[comma_counts != common_count].index.tolist()
                inconsistent_rows.extend(mismatched_rows)

            # Remove duplicates and sort rows needing replacement
            inconsistent_rows = sorted(set(inconsistent_rows))
            if len(inconsistent_rows) > 0:
                print(f"Processing {filename}")
                print(f"Found these inconsisent rows: {list(inconsistent_rows)}")

                if not count_only:
                    # Replace inconsistent rows with the row above or below
                    for row in inconsistent_rows:
                        if row > 1:
                            # Replace with the row above, not the header
                            df.iloc[row] = df.iloc[row - 1]
                        else:
                            # Replace the first row with the row below if it's the first row
                            df.iloc[row] = df.iloc[row + 1]

                    # Save the modified DataFrame back to the file
                    df.to_csv(file_path, index=False)
                    print(f"Processed file: {filename}")


def check_csv_headers():
    # Path to the data directory
    directory = "./data"

    train_header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Only consider files ending with "-TRAIN.csv"
        if filename.endswith("-TRAIN.csv"):
            filepath = os.path.join(directory, filename)

            # with open(filepath, 'r') as file:
            #     # Read the first line to check for the header
            #     first_line_string = file.readline()
            #
            #     if first_line_string != train_header:
            #         print(f"Header mismatch in file: {filename}")

            # Open and read the CSV file
            with open(filepath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                # Read the first row as the header
                header = next(reader, None)

                # Check if the header matches the expected header
                if header != train_header.split(","):
                    print(f"Header mismatch in file: {filename}")

    print("Finding mismatches in k = 1 TEST data files:")
    print("\n\n")
    test_header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,Borda ranking Winner,Plurality ranking Winner,STV Winner,Approval Voting (AV) Winner,Proportional Approval Voting (PAV) Winner,Approval Chamberlin-Courant (CC) Winner,Lexicographic Chamberlin-Courant (lex-CC) Winner,Sequential Approval Chamberlin-Courant (seq-CC) Winner,Monroe's Approval Rule (Monroe) Winner,Greedy Monroe Winner,Minimax Approval Voting (MAV) Winner,Method of Equal Shares (aka Rule X) with Phragmén phase Winner,E Pluribus Hugo (EPH) Winner,Random Serial Dictator Winner,Anti-Plurality Winner,Two-Approval Winner,Three-Approval Winner,Instant Runoff Winner,Bottom-Two-Runoff Instant Runoff Winner,Benham Winner,PluralityWRunoff PUT Winner,Coombs Winner,Baldwin Winner,Strict Nanson Winner,Weak Nanson Winner,Iterated Removal Condorcet Loser Winner,Raynaud Winner,Tideman Alternative Top Cycle Winner,Knockout Voting Winner,Banks Winner,Condorcet Winner,Copeland Winner,Llull Winner,Uncovered Set Winner,Uncovered Set - Fishburn Winner,Uncovered Set - Bordes Winner,Uncovered Set - McKelvey Winner,Slater Winner,Top Cycle Winner,GOCHA Winner,Bipartisan Set Winner,Minimax Winner,Split Cycle Winner,Ranked Pairs ZT Winner,Ranked Pairs TB Winner,Simple Stable Voting Winner,Stable Voting Winner,Loss-Trimmer Voting Winner,Daunou Winner,Blacks Winner,Condorcet IRV Winner,Condorcet IRV PUT Winner,Smith IRV Winner,Smith-Minimax Winner,Condorcet Plurality Winner,Copeland-Local-Borda Winner,Copeland-Global-Borda Winner,Borda-Minimax Faceoff Winner,Bucklin Winner,Simplified Bucklin Winner,Weighted Bucklin Winner,Bracket Voting Winner,Superior Voting Winner,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Only consider files ending with "-TRAIN.csv"
        if filename.endswith("-TEST.csv"):
            filepath = os.path.join(directory, filename)

            if "committee_size=1" not in filename:
                continue

            # Open and read the CSV file
            with open(filepath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                # Read the first row as the header
                header = next(reader, None)

                # Check if the header matches the expected header
                if header != test_header.split(","):
                    print(f"Header mismatch in file: {filename}")

    print("Finding mismatches in k > 1 TEST data files:")
    print("\n\n")
    test_header = "Profile,min_violations-committee,min_violations,max_violations-committee,max_violations,Borda ranking Winner,Plurality ranking Winner,STV Winner,Approval Voting (AV) Winner,Proportional Approval Voting (PAV) Winner,Approval Chamberlin-Courant (CC) Winner,Lexicographic Chamberlin-Courant (lex-CC) Winner,Sequential Approval Chamberlin-Courant (seq-CC) Winner,Monroe's Approval Rule (Monroe) Winner,Greedy Monroe Winner,Minimax Approval Voting (MAV) Winner,Method of Equal Shares (aka Rule X) with Phragmén phase Winner,E Pluribus Hugo (EPH) Winner,Random Serial Dictator Winner,candidate_pairs,candidate_pairs-normalized-no_diagonal,binary_pairs-no_diagonal,rank_matrix,rank_matrix-normalized"
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Only consider files ending with "-TRAIN.csv"
        if filename.endswith("-TEST.csv"):
            filepath = os.path.join(directory, filename)

            if "committee_size=1" in filename:
                continue


            try:
                # Open and read the CSV file
                with open(filepath, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    # Read the first row as the header
                    header = next(reader, None)

                    # Check if the header matches the expected header
                    if header != test_header.split(","):
                        print(f"Header mismatch in file: {filename}")
            except Exception as e:
                print(f"Caught exception: {e}")
                with open(filepath, 'r') as file:
                    # Read the first line to check for the header
                    first_line_string = file.readline()

                    if first_line_string != train_header:
                        print(f"Header mismatch in file: {filename}")


if __name__ == "__main__":
    check_csv_headers()
