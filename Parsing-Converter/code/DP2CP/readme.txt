To install dependencies:
    ./install.sh

To run:
    python3 dp_to_cp.py
or
    ./run.sh

File structure:
    dp_to_cp.py - contains main code for the tool.
    utils.py - contains a utility function to handle unwanted prints by the stanfordnlp package

Methodology:
    We follow a rule-based approach here. We handle sentences which consist of determiners, numeral modifiers, adjectives, nouns, pronouns, verbs, adverbs. The tool asks to input a sentence, then internally uses the stanfordnlp package to get its dependency parse, then converts it to DP and prints it to the screen.
 
