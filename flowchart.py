from graphviz import Digraph
import os

def create_flowchart():
    # set the desired output path
    output_path = r'C:\Users\main\OneDrive\Documents\ddos cloud flood\datset\modified'

    dot = Digraph(format='png')

    # Start
    dot.node('1', 'Start', shape='ellipse')

    # Input Handling (Parallelogram for Input/Output)
    dot.node('2', 'Input Handling\nLoad datasets:\n- Syn-training-balanced.csv\n- Syn-testing.csv\nDisplay Info:\n- Unique labels\n- Class counts', shape='parallelogram')

    # Data Preprocessing (Rectangle for Process)
    dot.node('3', 'Data Preprocessing\n- Separate Features/Labels\n- Encode Labels\n- Standardize Features\n- Backup Test Data', shape='rectangle')

    # Model Initialization (Rectangle for Process)
    dot.node('4', 'Model Initialization', shape='rectangle')
    dot.node('4A', '(A) LS-SVM\n- Linear Kernel', shape='ellipse')
    dot.node('4B', '(B) Naive Bayes\n- GaussianNB', shape='ellipse')
    dot.node('4C', '(C) KNN\n- n_neighbors=5', shape='ellipse')
    dot.node('4D', '(D) MLP\n- 100 Neurons\n- max_iter=300', shape='ellipse')

    # Model Training & Evaluation (Rectangle for Process)
    dot.node('5', 'Model Training & Evaluation\n- Train & Predict\n- Accuracy\n- Confusion Matrix\n- Classification Report\n- Store Results', shape='rectangle')

    # Prevention Mechanism (Rectangle for Process)
    dot.node('6', 'Prevention Mechanism\n- Identify Source Features\n- Flag Malicious Entries\n- Consensus Blacklist', shape='rectangle')

    # Cross-Validation (Parallelogram for Input/Output)
    dot.node('7', '3-Fold Cross-Validation\n- Mean ± Std of Accuracy', shape='parallelogram')

    # Output Results (Parallelogram for Input/Output)
    dot.node('8', 'Output Results\n- Accuracy\n- Confusion Matrix\n- Blocked Packets\n- Consensus Blacklist', shape='parallelogram')

    # End
    dot.node('9', 'End', shape='ellipse')

    # Edges (Flow)
    dot.edges(['12', '23', '34', '45', '56', '67', '78', '89'])

    # Connect Model Initialization to Training
    dot.edge('4', '4A')
    dot.edge('4', '4B')
    dot.edge('4', '4C')
    dot.edge('4', '4D')

    dot.edge('4A', '5')
    dot.edge('4B', '5')
    dot.edge('4C', '5')
    dot.edge('4D', '5')

    # ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # save flowchart to the specified directory
    output_file = os.path.join(output_path, 'syn_flood_flowchart')
    dot.render(output_file, cleanup=False)

    print(f"Flowchart saved at: {output_file}.png")

create_flowchart()
