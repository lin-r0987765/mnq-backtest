class Panel:
    @staticmethod
    def fit(text, **kwargs):
        import re
        return re.sub(r'\[/?[^\]]*\]', '', text)
