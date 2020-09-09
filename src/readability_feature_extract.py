import readability


def readability_attributes(dataset: list):

    for sample in dataset:
        text = sample['essay_sent']
        results = readability.getmeasures(text, lang='en')
        sample['syll_per_word'] = results['sentence info']['syll_per_word']
        sample['type_token_ratio'] = results['sentence info']['type_token_ratio']
        sample['syllables'] = results['sentence info']['syllables']
        sample['wordtypes'] = results['sentence info']['wordtypes']
        sample['long_words'] = results['sentence info']['long_words']
        sample['complex_words'] = results['sentence info']['complex_words']
        sample['complex_words_dc'] = results['sentence info']['complex_words_dc']

        sample['tobeverb'] = results['word usage']['tobeverb']
        sample['auxverb'] = results['word usage']['auxverb']
        sample['pronoun'] = results['word usage']['pronoun']
        sample['preposition'] = results['word usage']['preposition']
        sample['nominalization'] = results['word usage']['nominalization']

        sample['sentence_beginning_pronoun'] = results['sentence beginnings']['pronoun']
        sample['sentence_beginning_interrogative'] = results['sentence beginnings']['interrogative']
        sample['sentence_beginning_article'] = results['sentence beginnings']['article']
        sample['sentence_beginning_subordination'] = results['sentence beginnings']['subordination']
        sample['sentence_beginning_conjunction'] = results['sentence beginnings']['conjunction']
        sample['sentence_beginning_preposition'] = results['sentence beginnings']['preposition']

        sample['FleschReadingEase'] = results['readability grades']['FleschReadingEase']
        sample['Kincaid'] = results['readability grades']['Kincaid']
        sample['ARI'] = results['readability grades']['ARI']
        sample['Coleman-Liau'] = results['readability grades']['Coleman-Liau']
        sample['GunningFogIndex'] = results['readability grades']['GunningFogIndex']
        sample['LIX'] = results['readability grades']['LIX']
        sample['SMOGIndex'] = results['readability grades']['SMOGIndex']
        sample['RIX'] = results['readability grades']['RIX']
        sample['DaleChallIndex'] = results['readability grades']['DaleChallIndex']
