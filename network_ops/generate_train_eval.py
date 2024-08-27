
import os
import sys


from random import randint
import train_networks as tn
import evaluate_networks as en



def generate_train_eval():
    # Define what we want to make 
    pref_models = [
        #'stratification__args__weight=0.5',
        #'stratification__args__weight=0.5',
        #'URN-R',
        #'URN-R',
        #'IC',
        'IC',
        'MALLOWS-RELPHI-R',
        'MALLOWS-RELPHI-R',
        'single_peaked_conitzer',
        'single_peaked_conitzer'
    ]


    profile_counts = [randint(3000, 8000) for _ in range(len(pref_models))]
    prefs_per_profile = [randint(20, 100) for _ in range(len(pref_models))]
    candidate_sizes = [randint(3, 7) for _ in range(len(pref_models))]
    winners_sizes = [randint(2, candidate_sizes[i] - 1) for i in range(len(pref_models))]
    
    # We don't need to generate the data, as training the network with these params
    # will generate the data for us

    for i, (distribution, profile_count, num_voters, num_candidates, num_winners) in enumerate(zip(pref_models, profile_counts, prefs_per_profile, candidate_sizes, winners_sizes)):
        # Train the networks
        tn.train_networks(train_size=profile_count,
                          n=num_voters,
                          m=num_candidates,
                          num_winners=num_winners,
                          pref_dist=distribution)

        print("")
        print("FINISHED TRAINING")
        print("NOW EVALUATING")
        print("")    

        # Evaluate the networks
        en.save_accuracies_of_all_network_types(test_size=profile_count,
                                                n=num_voters,
                                                m=num_candidates,
                                                num_winners=num_winners,
                                                pref_dist=distribution)


generate_train_eval()


   