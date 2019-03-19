import re


pattern= r'(\w+|\!+|\.+|\,+|\?+)'
def findmatches(sentences):
    output = []
    for sen in sentences:
        match= re.findall(pattern, sen)
        cleaned = []
        for word in match:
            if word.find('.')>-1 :
                cleaned.append('.')
            elif word.find(',')>-1  :
                cleaned.append(',')
            elif word.find('?')>-1  :
                cleaned.append('?')
            elif word.find('!')>-1  :
                cleaned.append('!')
            else:
                cleaned.append(word)

        output.append(cleaned)
    return output


