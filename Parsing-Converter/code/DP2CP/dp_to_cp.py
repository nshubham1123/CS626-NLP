import stanfordnlp as s
from utils import *
# s.download('en')

PRONOUNS = ["i", "you", "he", "she", "it", "we", "they", "myself", "yourself",\
            "himself", "herself", "itself", "ourselves", "yourselves", "theirselves",\
            "who", "what", "somebody", "something", "anybody", "anything", "everybody",\
            "everything", "nobody", "nothing", "mine", "yours", "his", "hers",\
            "its", "ours", "theirs", "him", "her", "them"]

@dont_print
def dependency_parse(sent):
    # get dependency parse string of sentence from stanfordnlp api.
    nlp = s.Pipeline()
    doc = nlp(sent)
    return doc.sentences[0].dependencies_string()

def format_dependency_parse(parse):
    # convert parse string to list of tuples.
    parse = parse.strip().split("\n")
    
    parse_list = []
    for dependency in parse:
        parse_item = dependency.strip("()").split(",")
        parse_item = list(map(lambda x: x.strip(" '"), parse_item))
        parse_list.append(parse_item)

    return parse_list

def parse_list_to_parse_dict(parse_list):
    """
    convert parse list to a dictionary of dictionaries indexed by 
    position id of word in sentence.
    """
    parse_dict = {(id + 1): {"rel": rel, "head": int(head), "word": word, "tagged": False} 
                    for id, [word, head, rel] in enumerate(parse_list)}
    return parse_dict

def parse_list_to_relations_dict(parse_list):
    # mapping from words to relations on it.
    word_relations_dict = {i:[] for i in range(len(parse_list) + 1)}

    for _, head, rel in parse_list:
        parent_id = int(head)
        word_relations_dict[parent_id].append(rel)
    
    return word_relations_dict

def prettyprint(p):
    tabc = 0
    st = ""
    for c in p:
        if c not in [")", "("]:
            st += c
        elif c == "(":
            st += "\n"
            for i in range(tabc):
                st += "\t"
            st += c
            tabc += 1
        else:
            st += c
            tabc -= 1
    print(st)

def handle_dets(parse_dict, word_relations_dict):
    """
    there is only one determiner applied to a phrase. so, order of 
    traversal shouldn't matter.
    """

    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]

        if v["rel"] == "det":
            # tag the word as det.
            if not v["tagged"]:
                tagged_word = "( DET " + v["word"] + " )"
            else:
                tagged_word = v["word"]
            
            parent_id = v["head"]
            parent = parse_dict[parent_id]

            # tag parent.
            if not parent["tagged"]:
                tagged_parent = "( NOUN " + parent["word"] + " )"
            else:
                tagged_parent = parent["word"]

            # tag the phrase.
            tagged_det = "( NP " + tagged_word + " " + tagged_parent + " )"

            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_det
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("det")
            processed_words.append(id)
    
    for id in processed_words:
        del parse_dict[id]

def handle_adj(parse_dict, word_relations_dict):
    """
    Link adjectives to corresponding noun. We traverse the
    list in reverse word order to maintain order of adjectives
    in case of multiple adjectives.
    """
    processed_words = []
    keys = list(parse_dict.keys())
    keys.reverse()
    
    for id in keys:
        v = parse_dict[id]
        
        if v["rel"] == "amod":
            if not v["tagged"]:
                tagged_word = "( ADJ " + v["word"] + " )"
            else:
                tagged_word = v["word"]
        
            parent_id = v["head"]
            parent_relations = word_relations_dict[parent_id]
            parent = parse_dict[parent_id]
            
            if not parent["tagged"]:
                tagged_parent = "( NOUN " + parent["word"] + " )"
            else:
                tagged_parent = parent["word"]

            # tag the phrase.
            if "det" not in parent_relations and "nummod" not in parent_relations and parent_relations.count("amod") == 1:
                tagged_adj = "( NP " + tagged_word + " " + tagged_parent + " )"
            else:
                tagged_adj = tagged_word + " " + tagged_parent

            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_adj
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("amod")
            processed_words.append(id)
    
    for id in processed_words:
        del parse_dict[id]

def handle_obj(parse_dict, word_relations_dict):
    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]
        
        if v["rel"] == "obj":
            if not v["tagged"]:
                if v["word"].lower() not in PRONOUNS:
                    tagged_word = "( NP ( NOUN " + v["word"] + " ) )"
                else:
                    tagged_word = "( NP ( PRON " + v["word"] + " ) )"
            else:
                tagged_word = v["word"]
            
            parent_id = v["head"]
            parent = parse_dict[parent_id]
            if not parent["tagged"]:
                tagged_parent = "( VERB " + parent["word"] +  " )"
            else:
                tagged_parent = parent["word"]
            
            if "advmod" not in word_relations_dict[parent_id]:
                tagged_obj = "( VP " + tagged_parent + " " + tagged_word +  " )"
            else:
                tagged_obj = tagged_parent + " " + tagged_word

            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_obj
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("obj")
            processed_words.append(id)
    for id in processed_words:
        del parse_dict[id]

def handle_nsubj(parse_dict, word_relations_dict):
    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]
        
        if v["rel"] == "nsubj":
            if not v["tagged"]:
                if v["word"].lower() not in PRONOUNS:
                    tagged_word = "( NP ( NOUN " + v["word"] + " ) )"
                else:
                    tagged_word = "( NP ( PRON " + v["word"] + " ) )"
            else:
                tagged_word = v["word"]
            
            parent_id = v["head"]
            parent = parse_dict[parent_id]
            
            if not parent["tagged"]:
                tagged_parent = "( VP ( VERB " + parent["word"] + " ) )"
            else:
                tagged_parent = parent["word"]
            
            tagged_nsubj = tagged_word + " " + tagged_parent
            
            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_nsubj
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("nsubj")
            processed_words.append(id)
    for id in processed_words:
        del parse_dict[id]

def link_advmods(parse_dict, word_relations_dict):
    processed_words = []

    keys = list(parse_dict.keys())
    keys.reverse()

    for id in keys:
        v = parse_dict[id]
        if v["rel"] == "advmod":
            parent_id = v["head"]
            parent = parse_dict[parent_id]

            if parent["rel"] == "advmod":
                if not v["tagged"]:
                    tagged_word = "( ADV " + v["word"] + " )"
                else:
                    tagged_word = v["word"]
            
                if not parent["tagged"]:
                    tagged_parent = "( ADV " + parent["word"] + " )"
                else:
                    tagged_parent = parent["word"]
                
                tagged_adv = tagged_word + " " + tagged_parent

                # housekeeping
                parse_dict[parent_id]["word"] = tagged_adv
                parse_dict[parent_id]["tagged"] = True
                word_relations_dict[parent_id].remove("advmod")
                processed_words.append(id)
    
    for id in processed_words:
        del parse_dict[id]

def handle_advmods(parse_dict, word_relations_dict):
    # we have linked the adverbs at this point. For the accepted formats,
    # there will only be one set of adverbs now.
    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]
        if v["rel"] == "advmod":
            if not v["tagged"]:
                # there was only a single averb.
                tagged_word = "( ADV " + v["word"] + " )"
            else:
                tagged_word = v["word"]

            parent_id = v["head"]
            parent = parse_dict[parent_id]
            if not parent["tagged"]:
                # no obj. so verb was not tagged.
                tagged_parent = "( VERB " + parent["word"] + " )"
            else:
                tagged_parent = parent["word"]

            if v["head"] > id:
                # this set of adverbs comes before the verb.
                # needn't do any higher level tagging since that will be done at nsubj level.
                tagged_adv = "( ADVP " + tagged_word + " ) ( VP " + tagged_parent + " )"

            else:
                # this set of adverbs come after the verb, at end of sentence.
                # handled right after handling the object
                tagged_adv = "( VP " + tagged_parent + " ( ADVP " + tagged_word + " ) )"
            
            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_adv
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("advmod")
            processed_words.append(id)
    
    for id in processed_words:
        del parse_dict[id]

def handle_nummod(parse_dict, word_relations_dict):
    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]
        if v["rel"] == "nummod":
            tagged_word = "( NUM " + v["word"] + " )"

            parent_id = v["head"]
            parent = parse_dict[parent_id]
            
            if not parent["tagged"]:
                tagged_parent = "( NOUN " + parent["word"] + " )"
            else:
                tagged_parent = parent["word"]
            
            parent_relations = word_relations_dict[parent_id]
            if "det" not in parent_relations:
                # we can do the higher level tagging as well.
                tagged_result = "( NP " + tagged_word + " " + tagged_parent + " )"
            else:
                tagged_result = tagged_word + " " + tagged_parent
            # housekeeping
            parse_dict[parent_id]["word"] = tagged_result
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("nummod")
            processed_words.append(id)
    
    for id in processed_words:
        del parse_dict[id]

def handle_punct(parse_dict, word_relations_dict):
    processed_words = []
    for id in parse_dict:
        v = parse_dict[id]
        if v["rel"] == "punct":
            if not v["tagged"]:
                tagged_word = "( . " + v["word"] + " )"
            else:
                tagged_word = v["word"]
            
            parent_id = v["head"]
            parent = parse_dict[parent_id]
            
            tagged_parent = parent["word"]
            
            tagged_punct = "( S " + tagged_parent + " " + tagged_word + " )"
            
            # housekeeping.
            parse_dict[parent_id]["word"] = tagged_punct
            parse_dict[parent_id]["tagged"] = True
            word_relations_dict[parent_id].remove("punct")
            processed_words.append(id)
    for id in processed_words:
        del parse_dict[id]

def dp_to_cp(sent):
    try:
        # print("hello")
        parse = dependency_parse(sent)
        # print("hello")
        print("\nDependency parse (generated by StanfordNLP api.): ")
        print(parse)
        print("\n")
    except:
        print("StanfordNLP Parser threw error while parsing this sentence. Couln't get a Dependency Parse for sentence.\n")
        raise Exception
    parse_list = format_dependency_parse(parse)
    parse_dict = parse_list_to_parse_dict(parse_list)
    word_relations_dict = parse_list_to_relations_dict(parse_list)
    
    handle_adj(parse_dict, word_relations_dict)
    handle_nummod(parse_dict, word_relations_dict)
    handle_dets(parse_dict, word_relations_dict)
    link_advmods(parse_dict, word_relations_dict)
    handle_obj(parse_dict, word_relations_dict)
    handle_advmods(parse_dict, word_relations_dict)
    handle_nsubj(parse_dict, word_relations_dict)
    handle_punct(parse_dict, word_relations_dict)

    keys = list(parse_dict.keys())
    
    if len(keys) != 1:
        print("Parsing incomplete. Our tool doesn't support this sentence type yet.\n")
        print("parse_dict:\n", parse_dict)
        print("\nword_relations_dict:\n", word_relations_dict)
        raise Exception
    
    cp = parse_dict[keys[0]]["word"]
    cp = "( ROOT " + cp + " )"

    return cp

if __name__ == "__main__":
    while 1:
        # cp = dp_to_cp(sent)
        # print("Corresponding Constituency Parse: ")
        # prettyprint(cp)
        # print("\n")
        try:
            sent = input("Write sentence:\n")
            cp = dp_to_cp(sent)
            print("Corresponding Constituency Parse: ")
            prettyprint(cp)
            print("\n")
        except:
            print("\n")
            raise Exception