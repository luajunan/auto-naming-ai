def unknown4(matrix):
    flattened = [item for sublist in matrix for item in sublist]
    return sum(flattened) / len(flattened)


def unknown5(string_list):
    counts = {}
    for string in string_list:
        counts[string] = counts.get(string, 0) + 1
    return counts


def unknown6(matrix):
    result = []
    for row in matrix:
        row_sum = sum(row)
        row_avg = row_sum / len(row)
        result.append(row_avg)
    return result


def unknown7(number):
    if number < 0:
        return "Negative"
    elif number == 0:
        return "Zero"
    elif number % 2 == 0:
        return "Even"
    else:
        return "Odd"


def unknown8(string):
    vowels = "aeiouAEIOU"
    vowel_count = 0
    for char in string:
        if char in vowels:
            vowel_count += 1
    return vowel_count


def unknown9(word_list):
    word_lengths = {}
    for word in word_list:
        word_lengths[word] = len(word)
    return word_lengths
