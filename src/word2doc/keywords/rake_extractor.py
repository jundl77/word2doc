from rake_nltk import Rake


class RakeExtractor:

    def __init__(self):
        self.rake = Rake()

    def extract(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()
