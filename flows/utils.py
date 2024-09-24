import spacy


def spacy_preprocess(text, nlp):
    """
    Apply spaCy preprocessing to the text.
    """
    doc = nlp(text)
    return " ".join(
        [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    )
