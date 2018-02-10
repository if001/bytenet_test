
def get_word_lists(file_path):
    print("make wordlists")
    lines = open(file_path).read().split("\n")
    wordlists = []
    for line in lines:
        wordlists.append(line.split(" "))

    print("wordlist num:", len(wordlists))
    return wordlists[:-1]


def main():
    file_path = "./data/test.txt"
    word_list = get_word_lists(file_path)
    print(word_list)


if __name__ == "__main__":
    main()
